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


from pixel_datasets.glue_dataset_generator import GlueDatasetForPixel
from pixel_datasets.dataset_transformations import (
    SyntheticDatasetTransform,
    SimpleTorchTransform,
)
from pixel import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PIXELForSequenceClassification,
    PIXELTrainer,
    PIXELTrainingArguments,
    PoolingMode,
)

import wandb
from wandb.sdk.wandb_config import Config
from configs.utils import (
    read_args,
    evaluate_config_before_update,
    evaluate_config_after_update,
    update_config,
    update_config_key,
    generate_training_args_from_config,
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


def get_model_and_config(args: Config, num_labels: int):
    config_kwargs = {
        "cache_dir": args.model_cache_dir,
        "revision": args.model_revision,
        "use_auth_token": args.use_auth_token if args.use_auth_token else None,
    }

    config = AutoConfig.from_pretrained(
        args.model_config_name if args.model_config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        attention_probs_dropout_prob=args.dropout_prob,
        hidden_dropout_prob=args.dropout_prob,
        **config_kwargs,
    )

    logger.info(f"Using dropout with probability {args.dropout_prob}")

    if config.model_type in ["bert", "roberta"]:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            **config_kwargs,
        )
    elif config.model_type in ["vit_mae", "pixel"]:
        model = PIXELForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            pooling_mode=PoolingMode.from_string(args.pooling_mode),
            add_layer_norm=args.pooler_add_layer_norm,
            **config_kwargs,
        )
    else:
        raise ValueError(f"Model type {config.model_type} not supported.")

    return model, config


def get_collator(is_regression: bool = False):
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        attention_mask = torch.stack(
            [example["attention_mask"] for example in examples]
        )
        if "label" in examples[0]:
            if is_regression:
                labels = torch.FloatTensor([example["label"] for example in examples])
            else:
                labels = torch.LongTensor([example["label"] for example in examples])
            return {
                "pixel_values": pixel_values,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        return {"pixel_values": pixel_values, "attention_mask": attention_mask}

    return collate_fn


def get_datasets(args: Config, seed: int):
    rng = np.random.RandomState(seed)
    transform = SimpleTorchTransform(args, rng)

    train_dataset = GlueDatasetForPixel(
        config=args, task=args.task_name, split="train", transform=transform, rng=rng
    )
    test_dataset = GlueDatasetForPixel(
        config=args, task=args.task_name, split="validation", transform=transform, rng=rng
    )

    return train_dataset, test_dataset


def main(args: Config):
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
        f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}"
        + f"distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(args.seed)

    train_dataset, eval_dataset = get_datasets(args, args.seed)

    # Load pretrained model and config
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    raw_dataset = train_dataset.text_dataset
    is_regression = args.task_name == "stsb"
    if not is_regression:
        label_list = raw_dataset.features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # Load pretrained model and config
    model, config = get_model_and_config(args, num_labels=num_labels)

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    # Log a few random samples from the training set:
    if args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]['label']}."
            )

    if args.do_eval:
        for index in random.sample(range(len(eval_dataset)), 3):
            logger.info(
                f"Sample {index} of the eval set: {eval_dataset[index]['label']}."
            )

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        breakpoint()
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Initialize our Trainer
    trainer = PIXELTrainer(
        model=model,
        args=generate_training_args_from_config(args),
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=get_collator(is_regression=is_regression),
    )

    # Training
    if args.do_train:
        checkpoint = None
        if args.resume_from_checkpoint is not None:
            checkpoint = args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            args.max_train_samples
            if args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)

        outputs = trainer.evaluate(eval_dataset=eval_dataset)
        logger.info("***** Eval results *****")
        for key, value in outputs.items():
            logger.info(f"  {key} = {value}")
        trainer.log_metrics("eval", outputs)

    kwargs = {
        "finetuned_from": args.model_name_or_path,
        "tasks": "text-classification",
    }
    if args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = args.task_name
        kwargs["dataset"] = f"GLUE {args.task_name.upper()}"

    if args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    command_line_args = read_args()
    wandb.init(
        project="pixel",
        config="configs/glue_config.yaml",
        mode="disabled",
        name=command_line_args["run_name"],
        save_code=True,
    )
    assert evaluate_config_before_update(wandb.config)
    update_config(wandb.config, command_line_args)
    assert evaluate_config_after_update(wandb.config)

    main(wandb.config)
