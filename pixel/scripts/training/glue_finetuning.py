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
from dataclasses import dataclass

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset, load_metric, concatenate_datasets
from transformers import (
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from configs.all_configs import FineTuningDatasetArguments, RenderingArguments, FineTuningModelArguments
from dataset_synthesis.document_syntesis import DocumentSynthesizer
from dataset_synthesis.glue_dataset import GlueDataset
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

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

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
        finetuning_task=dataset_args.task_name,
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


def get_datasets(args: FineTuningDatasetArguments, rendering_args: RenderingArguments, seed: int):
    base_dataset = load_dataset("glue", args.task_name, cache_dir=args.dataset_cache_dir)
    if args.task_name == "mnli":
        base_dataset["validation"] = concatenate_datasets([base_dataset["validation_matched"], base_dataset["validation_mismatched"]])

    rng = np.random.RandomState(seed)
    ds = DocumentSynthesizer(rendering_args, rng=rng)

    train_dataset = GlueDataset(args, task=args.task_name, data=base_dataset["train"], document_synthesizer=ds)
    test_dataset = GlueDataset(args, task=args.task_name, data=base_dataset["validation"], document_synthesizer=ds)
    return train_dataset, test_dataset, base_dataset


def main():

    parser = HfArgumentParser((FineTuningModelArguments, FineTuningDatasetArguments, PIXELTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    parser = HfArgumentParser((RenderingArguments))
    rendering_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]

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

    train_dataset, eval_dataset, raw_datasets = get_datasets(data_args, rendering_args, training_args.seed)
    # Load pretrained model and config
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # Load pretrained model and config
    model, config = get_model_and_config(model_args=model_args, dataset_args=data_args, num_labels=num_labels)

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]['label']}.")

    if training_args.do_eval:
        for index in random.sample(range(len(eval_dataset)), 3):
            logger.info(f"Sample {index} of the eval set: {eval_dataset[index]['label']}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    training_args.dataloader_num_workers = 16
    training_args.per_device_train_batch_size = 16
    training_args.evaluation_strategy = "steps"
    training_args.fp16 = True
    training_args.half_precision_backend = "amp"
    training_args.report_to = "wandb"


    # Initialize our Trainer
    trainer = PIXELTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=get_collator(is_regression=is_regression),
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

        outputs = trainer.evaluate(eval_dataset=eval_dataset)
        logger.info("***** Eval results *****")
        for key, value in outputs.items():
            logger.info(f"  {key} = {value}")
        trainer.log_metrics("eval", outputs)


    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()