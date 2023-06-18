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

import logging
import os
import sys
from dataclasses import dataclass

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset, load_metric, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import (
    EvalPrediction,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from datasets import load_dataset
from pixel import (
    AutoConfig,
    PIXELForTokenClassification,
    PIXELTrainer,
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
from pixel_datasets.squad_dataset_for_pixel import SquadDatasetFromDisk
from pixel.scripts.training.callbacks import SquadVisualizationCallback

from PIL import Image

check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install ./datasets")


logger = logging.getLogger(__name__)


def get_model_and_config(args: Config, num_labels: int = 2):
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

    model = PIXELForTokenClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        **config_kwargs,
    )

    return model, config


def get_collator(is_regression: bool = False):
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        attention_mask = torch.stack(
            [example["attention_mask"] for example in examples]
        )
        if "label" in examples[0]:
            labels = torch.stack([example["label"] for example in examples])
            return {
                "pixel_values": pixel_values,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        return {"pixel_values": pixel_values, "attention_mask": attention_mask}

    return collate_fn


def get_datasets(args: Config, seed: int):
    # Load data features from cache or dataset file
    try:
        dataset = load_dataset(args.dataset_name, cache_dir=args.dataset_cache_dir)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
    except DatasetGenerationError:
        dataset = load_from_disk(args.dataset_name)
        train_dataset = SquadDatasetFromDisk(base_dataest=dataset["train"], config=args)
        test_dataset = SquadDatasetFromDisk(base_dataest=dataset["test"], config=args)

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

    num_labels = args.num_labels if args.num_labels is not None else 2

    # Load pretrained model and config
    model, config = get_model_and_config(args, num_labels=num_labels)

    # Some models have set the order of the labels to use, so let's make sure we do use it.

    label_list = ["Answer", "NotAnswer"]
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        def compute_single_acc(pred, label):
            summed = label + pred
            if max(summed) == 0:
                return 1.0
            else:
                num_missmatched = np.sum(summed == 1)
                num_matched = np.sum(summed == 2) / 2
                return num_matched / (num_missmatched + num_matched)

        def compute_has_match(pred, label):
            summed = label + pred
            return 1 if np.max(summed) == 2 else 0

        def false_negative(pred, label):
            return 1 if np.max(label) == 1 and np.max(pred) == 0 else 0

        def false_positive(pred, label):
            return 1 if np.max(pred) == 1 and np.max(label) == 0 else 0

        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=2)
        labels = p.label_ids

        accuracies = [compute_single_acc(p, l) for p, l in zip(preds, labels)]
        has_answer = np.mean(np.max(preds, axis=1))
        has_match = [compute_has_match(p, l) for p, l in zip(preds, labels)]
        false_negatives = [false_negative(p, l) for p, l in zip(preds, labels)]
        false_positives = [false_positive(p, l) for p, l in zip(preds, labels)]
        return {
            "accuracy": np.mean(accuracies),
            "has_match": np.sum(has_match),
            "has_answer": has_answer,
            "false_negatives": np.sum(false_negatives),
            "false_positives": np.sum(false_positives),
            "num_samples": len(accuracies),
        }

    # Initialize our Trainer
    trainer = PIXELTrainer(
        model=model,
        args=generate_training_args_from_config(args),
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=get_collator(),
    )

    if args.do_eval and "wandb" in args.report_to:
        logger.info(f"adding visualization callback")
        trainer.add_callback(SquadVisualizationCallback(visualize_train=False))

    # Training
    if args.do_train:
        checkpoint = None
        if args.resume_from_checkpoint is not None:
            checkpoint = args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # # Evaluation
    # if args.do_eval:
    #     logger.info("*** Evaluate ***")

    #     # Loop to handle MNLI double evaluation (matched, mis-matched)

    #     outputs = trainer.evaluate(eval_dataset=eval_dataset)
    #     logger.info("***** Eval results *****")
    #     for key, value in outputs.items():
    #         logger.info(f"  {key} = {value}")
    #     trainer.log_metrics("eval", outputs)

    # kwargs = {"finetuned_from": args.model_name_or_path, "tasks": "question-answering"}
    # if args.task_name is not None:
    #     kwargs["language"] = "en"
    #     kwargs["dataset_tags"] = "pixel-squad"
    #     kwargs["dataset_args"] = args.task_name
    #     kwargs["dataset"] = args.dataset_name

    # if args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    command_line_args = read_args()
    wandb.init(
        project="pixel",
        config="configs/squad_config.yaml",
        mode="online",
        name=command_line_args["run_name"],
        save_code=True,
    )
    # assert evaluate_config_before_update(wandb.config)
    update_config(wandb.config, command_line_args)
    assert evaluate_config_after_update(wandb.config)

    main(wandb.config)
