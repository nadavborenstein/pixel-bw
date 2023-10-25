#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
import logging
import os
import sys
from collections import defaultdict
from dataclasses import asdict

import wandb

from typing import Any, Dict
import numpy as np
import datasets
import torch
import transformers
from datasets import interleave_datasets, load_dataset

from pixel_datasets.pixel_dataset_generator import PretrainingDataset
from pixel_datasets.dataset_transformations import (
    SimpleTorchTransform,
    SyntheticDatasetTransform,
)

from pixel import (
    PIXELConfig,
    PIXELForPreTraining,
    PIXELTrainerForPretraining,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from pixel.scripts.training.callbacks import VisualizationCallback
from wandb.sdk.wandb_config import Config
from configs.utils import (
    read_args,
    evaluate_config_before_update,
    evaluate_config_after_update,
    update_config,
    update_config_key,
    generate_training_args_from_config,
)

""" Pre-training a PIXEL model as an MAE (masked autoencoder)"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")
require_version("datasets>=1.8.0", "To fix: pip install ./datasets")


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    attention_mask = torch.stack([example["attention_mask"] for example in examples])
    inputs = {"pixel_values": pixel_values, "attention_mask": attention_mask}
    if "patch_mask" in examples[0]:
        patch_mask = torch.stack([example["patch_mask"] for example in examples])
        inputs.update({"patch_mask": patch_mask})
    return inputs


def get_datasets(args: Config):
    rng = np.random.RandomState(args.seed)
    train_text_datasets = [
        load_dataset(
            d_name,
            d_config,
            split=d_split,
            use_auth_token=args.use_auth_token,
            cache_dir=args.dataset_cache_dir,
        )
        for d_name, d_config, d_split in zip(
            args.train_dataset_names,
            args.train_dataset_configs,
            args.train_splits,
        )
    ]
    if args.generate_test_data_from_train:
        test_text_datasets = []
        for i in range(len(train_text_datasets)):
            ds = train_text_datasets[i]
            ds_size = len(ds)
            test_size = ds_size // 500
            random_test_indices = rng.choice(ds_size, test_size, replace=False)
            random_train_indices = np.setdiff1d(np.arange(ds_size), random_test_indices)
            test_text_datasets.append(ds.select(random_test_indices))
            train_text_datasets[i] = ds.select(random_train_indices)
            assert (
                len(train_text_datasets[i]) + len(test_text_datasets[i]) == ds_size
            ), "Dataset sizes don't add up: {} + {} != {}".format(
                len(train_text_datasets[i]), len(test_text_datasets[i]), ds_size
            )

        test_text_dataset = datasets.concatenate_datasets(test_text_datasets)
        test_text_dataset = test_text_dataset.shuffle(seed=args.seed)
    else:
        test_text_dataset = load_dataset("wikipedia", "20220301.simple")["train"]
        test_text_dataset = test_text_dataset.shuffle(seed=args.seed)

    dataset_sizes = [len(ds) for ds in train_text_datasets]
    combined_size = sum(dataset_sizes)
    dataset_sampling_probs = [d_size / combined_size for d_size in dataset_sizes]
    train_text_dataset = interleave_datasets(
        train_text_datasets,
        probabilities=dataset_sampling_probs,
        seed=args.seed,
    )
    train_transform = SyntheticDatasetTransform(args, rng=rng)
    test_transform = SyntheticDatasetTransform(args, rng=rng)

    train_dataset = PretrainingDataset(
        config=args,
        text_dataset=train_text_dataset,
        transform=train_transform,
        rng=rng,
    )

    test_dataset = PretrainingDataset(
        config=args,
        text_dataset=test_text_dataset,
        transform=test_transform,
        rng=rng,
        max_steps=args.num_test_steps,
    )
    return train_dataset, dataset_sampling_probs, test_dataset


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

    logger.info(f"Configuration: {args}")

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

    # Initialize our datasets

    train_dataset, dataset_sampling_probs, validation_dataset = get_datasets(args)
    logger.info("***** Interleaving training datasets *****")
    for d_name, d_config, d_split, d_sampling_prob in zip(
        args.train_dataset_names,
        args.train_dataset_configs,
        args.train_splits,
        dataset_sampling_probs,
    ):
        logger.info(
            f"\tDataset name = {d_name}, config = {d_config}, split = {d_split}, "
            f"sampling probability = {d_sampling_prob:.3f}, cache = {args.dataset_cache_dir}"
        )

    config_kwargs = {
        "cache_dir": args.model_cache_dir,
        "revision": args.model_revision,
        "use_auth_token": args.use_auth_token,
    }
    logger.info(f"Using dropout with probability {args.dropout_prob}")

    if args.model_config_name:
        config = PIXELConfig.from_pretrained(
            args.model_config_name,
            attention_probs_dropout_prob=args.dropout_prob,
            hidden_dropout_prob=args.dropout_prob,
            **config_kwargs,
        )
    elif args.model_name_or_path:
        config = PIXELConfig.from_pretrained(
            args.model_name_or_path,
            attention_probs_dropout_prob=args.dropout_prob,
            hidden_dropout_prob=args.dropout_prob,
            **config_kwargs,
        )
    else:
        config = PIXELConfig(
            attention_probs_dropout_prob=args.dropout_prob,
            hidden_dropout_prob=args.dropout_prob,
            **config_kwargs,
        )
        logger.warning("You are instantiating a new config instance from scratch.")
        if args.config_overrides is not None:
            logger.info(f"Overriding config: {args.config_overrides}")
            config.update_from_string(args.config_overrides)
            logger.info(f"New config: {config}")

    # Adapt config
    config.update(
        {
            "mask_ratio": args.mask_ratio,
            "norm_pix_loss": args.norm_pix_loss,
            "architectures": [PIXELForPreTraining.__name__],
        }
    )

    # Create model
    if args.model_name_or_path:
        logger.info(f"Training from pretrained model {args.model_name_or_path}")
        model = PIXELForPreTraining.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            **config_kwargs,
        )
    else:
        logger.info("Training new model from scratch")
        model = PIXELForPreTraining(config)

    logger.info("***** Final model config *****")
    logger.info(config)

    total_params = sum([p.numel() for p in model.parameters()])
    logger.info(f"Total parameters count: {total_params}")
    encoder_params = sum([p.numel() for p in model.vit.parameters()])
    logger.info(f"Encoder parameters count: {encoder_params}")
    encoder_embedding_params = sum(
        [p.numel() for p in model.vit.embeddings.parameters()]
    )
    logger.info(f"Encoder embeddings parameters count: {encoder_embedding_params}")
    decoder_params = sum([p.numel() for p in model.decoder.parameters()])
    logger.info(f"Decoder parameters count: {decoder_params}")

    # calulates the real batch size and learning rate
    total_train_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps * args.n_gpu
    )
    if args.base_learning_rate is not None:
        update_config_key(
            args,
            "learning_rate",
            args.base_learning_rate * total_train_batch_size / 256,
        )

        # Initialize our trainer
        logger.info(f"LEN OF EVAL DATASET: {args.num_test_steps * max(1, args.n_gpu)}")

    trainer = PIXELTrainerForPretraining(
        model=model,
        args=generate_training_args_from_config(args),
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=validation_dataset,
        data_collator=collate_fn,
    )

    if args.do_eval and "wandb" in args.report_to:
        logger.info(f"adding visualization callback")
        trainer.add_callback(VisualizationCallback(visualize_train=False))

    # Training
    if args.do_train:
        checkpoint = None
        if args.resume_from_checkpoint is not None:
            checkpoint = args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
            logger.info(f"Resuming from Checkpoint: {checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        # Also save feature extractor together with model and text renderer
        # feature_extractor.save_pretrained(args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    print("eval metrics:", metrics)
    # Write model card and (optionally) push to hub
    kwargs = {
        "tasks": "masked-auto-encoding",
        "dataset": "wikipedia + bookcorpus",
        "tags": ["masked-auto-encoding"],
    }
    if args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    command_line_args = read_args()
    wandb.init(
        project="pixel",
        config="configs/pretraining_config.yaml",
        mode="online",
        name=command_line_args["run_name"],
        save_code=True,
    )
    assert evaluate_config_before_update(wandb.config)
    update_config(wandb.config, command_line_args)
    assert evaluate_config_after_update(wandb.config)

    main(wandb.config)
