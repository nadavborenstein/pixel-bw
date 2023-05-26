"""
Script that takes in an input string, renders it, masks out patches, and let's PIXEL reconstruct the image.
Adapted from ViT-MAE demo: https://github.com/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb

Example usage:
python visualize_pixel.py \
  --input_str="Cats conserve energy by sleeping more than most animals, especially as they grow older." \
  --model_name_or_path="Team-PIXEL/pixel-base" \
  --span_mask \
  --mask_ratio=0.25 \
  --max_seq_length=256

"""

import argparse
import logging
import math
import sys

import torch
from datasets import load_dataset, interleave_datasets
import numpy as np

import wandb
from PIL import Image

from configs.all_configs import RenderingArguments, VisualizationArguments, ModelArguments, DataTrainingArguments, \
    CustomTrainingArguments
from dataset_synthesis.document_syntesis import DocumentSynthesizer
from dataset_synthesis.synthetic_dataset import SyntheticDatasetTransform, SyntheticDatasetTorch
from pixel import (
    AutoConfig,
    PIXELForPreTraining,
    resize_model_embeddings,
    truncate_decoder_pos_embeddings, PIXELEmbeddings,
)
from transformers import set_seed, HfArgumentParser

logger = logging.getLogger(__name__)


def get_eval_dataset(model_args, data_args, rendering_args, training_args):
    train_text_datasets = [
        load_dataset(
            d_name,
            d_config,
            split=d_split,
            use_auth_token=model_args.use_auth_token,
            cache_dir=d_cache,
        )
        for d_name, d_config, d_split, d_cache in zip(
            data_args.train_dataset_names,
            data_args.train_dataset_configs,
            data_args.train_splits,
            data_args.dataset_caches,
        )
    ]
    dataset_sizes = [ds._info.splits.total_num_examples for ds in train_text_datasets]
    combined_size = sum(dataset_sizes)
    dataset_sampling_probs = [d_size / combined_size for d_size in dataset_sizes]
    train_text_dataset = interleave_datasets(
        train_text_datasets,
        probabilities=dataset_sampling_probs,
        seed=training_args.seed,
    )
    rng = np.random.RandomState(training_args.seed)
    transform = SyntheticDatasetTransform(rendering_args, rng=rng)
    ds = DocumentSynthesizer(rendering_args, rng=rng)
    train_dataset = SyntheticDatasetTorch(
        train_text_dataset,
        transform=transform,
        args=rendering_args,
        document_synthesizer=ds,
        overfit=training_args.overfit,
        rng=rng
    )
    train_dataset.max_step = training_args.max_steps
    train_dataset.warmup_steps = training_args.warmup_steps
    train_dataset.randomness_intensity_update_interval = training_args.randomness_intensity_update_interval

    return train_dataset, train_dataset.get_evaluation_set(10)


def get_datasets(rendering_args):
    transform = SyntheticDatasetTransform(rendering_args)
    ds = DocumentSynthesizer(rendering_args)
    train_dataset = SyntheticDatasetTorch(
        None,
        transform=transform,
        args=rendering_args,
        document_synthesizer=ds,
    )
    return train_dataset



def clip(x: torch.Tensor):
    x = torch.einsum("chw->hwc", x)
    x = torch.clip(x * 255, 0, 255)
    x = torch.einsum("hwc->chw", x)
    return x


def log_image(img: torch.Tensor, img_name: str, do_clip: bool = True):
    if do_clip:
        img = clip(img)
    wandb.log({img_name: wandb.Image(img)})


def main():
    parser = HfArgumentParser(
        (
            VisualizationArguments,
            ModelArguments,
            DataTrainingArguments,
            CustomTrainingArguments,
            RenderingArguments,
        )
    )
    args, model_args, data_args, training_args, rendering_args = parser.parse_args_into_dataclasses()
    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logger.setLevel(log_level)

    set_seed(args.seed)

    wandb.init()
    wandb.run.name = args.revision

    config_kwargs = {
        "use_auth_token": args.auth_token if args.auth_token else None,
        "revision": args.revision,
    }


    # Load Dataset
    train_dataset, dataset = get_eval_dataset(model_args, data_args, rendering_args, training_args)

    # Load model
    config = AutoConfig.from_pretrained(args.vis_model_path, **config_kwargs)


    model = PIXELForPreTraining.from_pretrained(args.vis_model_path, config=config, **config_kwargs)

    # Resize position embeddings in case we use shorter sequence lengths
    resize_model_embeddings(model, train_dataset.num_patches)
    truncate_decoder_pos_embeddings(model, train_dataset.num_patches)

    logger.info("Running PIXEL masked autoencoding with pixel reconstruction")


    # Render input
    inputs = dataset[0]
    image_height = inputs["pixel_values"].shape[-2]
    image_width = inputs["pixel_values"].shape[-1]
    model.config.image_size = (image_height, image_width)
    model.image_size = (image_height, image_width)
    model.vit.embeddings = PIXELEmbeddings(model.config)

    # Run MAE
    model.eval()
    with torch.inference_mode():
        outputs = model(pixel_values=inputs["pixel_values"].unsqueeze(0),
                        attention_mask=inputs["attention_mask"].unsqueeze(0),
                        patch_mask=inputs["patch_mask"].unsqueeze(0))

    predictions = model.unpatchify(outputs["logits"]).detach().cpu().squeeze()

    # visualize the mask
    mask = outputs["mask"].detach().cpu()

    # Log mask
    mask = mask.unsqueeze(-1).repeat(1, 1, 16 ** 2 * 3)
    mask = model.unpatchify(mask).squeeze()  # 1 is removing, 0 is keeping
    log_image(mask, "mask")

    # Log attention mask
    attention_mask = inputs["attention_mask"].unsqueeze(-1).repeat(1, 1, 16 ** 2 * 3)
    attention_mask = model.unpatchify(attention_mask).squeeze()
    log_image(attention_mask, "attention_mask")

    # Log original image
    original_img = model.unpatchify(model.patchify(inputs["pixel_values"].unsqueeze(0))).squeeze()
    log_image(original_img, "original")

    # Log masked image
    im_masked = original_img * (1 - mask)
    log_image(im_masked, "masked")

    # Log predictions
    log_image(predictions, "predictions", do_clip=False)

    # Log masked predictions
    masked_predictions = predictions * mask * attention_mask
    log_image(masked_predictions, "masked_predictions", do_clip=False)

    # Log MAE reconstruction pasted with visible patches
    reconstruction = (
        original_img * (1 - (torch.bitwise_and(mask == 1, attention_mask == 1)).long())
        + predictions * mask * attention_mask
    )
    log_image(reconstruction, "reconstruction", do_clip=False)


if __name__ == "__main__":
    main()
