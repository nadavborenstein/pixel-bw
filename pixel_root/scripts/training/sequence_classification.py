#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""

import argparse
import logging
import os
import random
import sys

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset, load_metric
from transformers import (
    EvalPrediction,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from configs.all_configs import FineTuningDatasetArguments, FineTuningModelArguments
from dataset_synthesis.real_dataset import HistoricDatasetForPretraining
from pixel import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PIXELForSequenceClassification,
    PIXELTrainer,
    PIXELTrainingArguments,
    PoolingMode,
)

check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install ./datasets")


logger = logging.getLogger(__name__)


def get_model_and_config(model_args: argparse.Namespace, dataset_args, num_labels: int):
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token if model_args.use_auth_token else None,
    }

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        attention_probs_dropout_prob=model_args.dropout_prob,
        hidden_dropout_prob=model_args.dropout_prob,
        **config_kwargs,
    )

    logger.info(f"Using dropout with probability {model_args.dropout_prob}")

    if config.model_type in ["bert", "roberta"]:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            **config_kwargs,
        )
    elif config.model_type in ["vit_mae", "pixel"]:
        model = PIXELForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            pooling_mode=PoolingMode.from_string(model_args.pooling_mode),
            add_layer_norm=model_args.pooler_add_layer_norm,
            **config_kwargs,
        )
    else:
        raise ValueError(f"Model type {config.model_type} not supported.")

    image_height = dataset_args.figure_size[0]
    image_width = dataset_args.figure_size[1]
    model.config.image_size = (image_height, image_width)
    model.image_size = (image_height, image_width)
    model.vit.config.image_size = (image_height, image_width)
    model.vit.embeddings.patch_embeddings.image_size = (image_height, image_width)
    return model, config


def get_datasets(args: FineTuningDatasetArguments, seed: int):
    base_dataset = load_dataset(args.dataset_name, cache_dir=args.dataset_cache_dir)

    categories = list(set(base_dataset["train"][args.label_column_name]).union(set(base_dataset["test"][args.label_column_name])))
    categories.sort()
    args.number_of_labels = len(categories)
    label_mapper = {label: i for i, label in enumerate(categories)}

    def map_labels(example):
        example["label"] = label_mapper[example[args.label_column_name]]
        return example

    rng = np.random.RandomState(seed)
    train_dataset = HistoricDatasetForPretraining(args, base_dataset["train"], rng, labeled=True, preprocessing_function=map_labels)
    test_dataset = HistoricDatasetForPretraining(args, base_dataset["test"], rng, labeled=True, preprocessing_function=map_labels)
    return train_dataset, test_dataset


def get_collator(
        is_regression: bool = False
):
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        if "label" in examples[0]:
            if is_regression:
                labels = torch.FloatTensor([example["label"] for example in examples])
            else:
                labels = torch.LongTensor([example["label"] for example in examples])
            return {"pixel_values": pixel_values, "attention_mask": attention_mask, "labels": labels}
        return {"pixel_values": pixel_values, "attention_mask": attention_mask}

    return collate_fn

def main():

    parser = HfArgumentParser((FineTuningModelArguments, FineTuningDatasetArguments, PIXELTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    train_dataset, test_dataset = get_datasets(data_args, training_args.seed)
    # Load pretrained model and config
    model, config = get_model_and_config(model_args, data_args, data_args.number_of_labels)



    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]['label']}.")

    if training_args.do_eval:
        for index in random.sample(range(len(test_dataset)), 3):
            logger.info(f"Sample {index} of the eval set: {test_dataset[index]['label']}.")

    # Get the metric function
    metric = load_metric("accuracy")
    is_regression = False  # TODO change if needed

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Initialize our Trainer
    trainer = PIXELTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=test_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=get_collator(is_regression=is_regression),
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)]
        # if training_args.early_stopping
        # else None,  TODO fix this
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)

        outputs = trainer.evaluate(eval_dataset=test_dataset)
        logger.info("***** Eval results *****")
        for key, value in outputs.items():
            logger.info(f"  {key} = {value}")
        trainer.log_metrics("eval", outputs)


    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()