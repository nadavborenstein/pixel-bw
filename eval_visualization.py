import numpy as np
from datasets import load_dataset, load_from_disk
from pixel_datasets.dataset_transformations import (
    SyntheticDatasetTransform,
    SimpleTorchTransform,
)
from pixel_datasets.pixel_dataset_generator import PretrainingDataset
from pixel_datasets.glue_dataset_generator import GlueDatasetForPixel
from pixel_datasets.utils.utils import (
    mask_single_word_from_scan,
    convert_torch_tensor_to_image,
    plot_arrays,
    merge_mask_with_image,
    get_mask_edges,
    convert_patch_mask_to_pixel_mask,
)

import wandb
from tqdm.auto import tqdm
from pixel.utils.inference import load_model_for_pretraining, predict, parse_outputs
from PIL import Image
import matplotlib.pyplot as plt
import torch


def random_masking():
    visualization_dataset = load_dataset("multi_news", split="validation")
    visualization_dataset = visualization_dataset.rename_column("document", "text")

    # visualization_dataset = load_dataset("wikipedia", "20220301.simple")["train"]
    print(visualization_dataset[0])

    rng = np.random.RandomState(42)
    wandb.init(
        config="/home/knf792/PycharmProjects/pixel-2/configs/inference_config.yaml",
        mode="disabled",
    )
    transform = SyntheticDatasetTransform(wandb.config, rng=rng)
    simple_transform = SimpleTorchTransform(wandb.config, rng=rng)

    dataset = PretrainingDataset(
        config=wandb.config,
        text_dataset=visualization_dataset,
        transform=transform,
        rng=rng,
    )

    visualization_examples = []
    counter = 9
    for sample in tqdm(dataset):
        if counter == 0:
            break
        counter -= 1
        visualization_examples.append(sample)

    model = load_model_for_pretraining(
        wandb.config,
        "Nadav/PretrainedPHD-v2",
    )

    predictions = []
    for example in tqdm(visualization_examples):
        outputs = predict(model, example["pixel_values"], example["patch_mask"])
        prediction = parse_outputs(outputs, model, example["pixel_values"])
        predictions.append(prediction)

    for i in range(len(predictions)):
        mask = visualization_examples[i]["patch_mask"]
        mask = mask.numpy()
        pixel_mask = convert_patch_mask_to_pixel_mask(mask)
        only_edges = get_mask_edges(pixel_mask, 3)
        merged = merge_mask_with_image(
            only_edges, np.array(predictions[i]), colour=(0, 0, 0), alpha=0.1
        )
        predictions[i] = merged

    final = plot_arrays(predictions)
    final.save("evaluations/final.png")
    
    
def mask_a_word():
    visualization_dataset = load_dataset("multi_news", split="validation")
    visualization_dataset = visualization_dataset.rename_column("document", "text")

    # visualization_dataset = load_dataset("wikipedia", "20220301.simple")["train"]
    print(visualization_dataset[0])

    rng = np.random.RandomState(42)
    wandb.init(
        config="/home/knf792/PycharmProjects/pixel-2/configs/inference_config.yaml",
        mode="disabled",
    )
    transform = SyntheticDatasetTransform(wandb.config, rng=rng)
    simple_transform = SimpleTorchTransform(wandb.config, rng=rng)

    dataset = PretrainingDataset(
        config=wandb.config,
        text_dataset=visualization_dataset,
        transform=transform,
        rng=rng,
    )

    visualization_examples = []
    counter = 9
    for sample in tqdm(dataset):
        if counter == 0:
            break
        counter -= 1
        visualization_examples.append(sample)

    for sample in visualization_examples:
        im = convert_torch_tensor_to_image(sample["pixel_values"])
        pixel_mask = mask_single_word_from_scan(im)
        patch_mask = pixel_mask[::16, ::16]
        sample["patch_mask"] = torch.from_numpy(pixel_mask).flatten()
        
        
    model = load_model_for_pretraining(
        wandb.config,
        "Nadav/PretrainedPHD-v2",
    )

    predictions = []
    for example in tqdm(visualization_examples):
        outputs = predict(model, example["pixel_values"], example["patch_mask"])
        prediction = parse_outputs(outputs, model, example["pixel_values"])
        predictions.append(prediction)

    for i in range(len(predictions)):
        mask = visualization_examples[i]["patch_mask"]
        mask = mask.numpy()
        pixel_mask = convert_patch_mask_to_pixel_mask(mask)
        only_edges = get_mask_edges(pixel_mask, 3)
        merged = merge_mask_with_image(
            only_edges, np.array(predictions[i]), colour=(0, 0, 0), alpha=0.1
        )
        predictions[i] = merged

    final = plot_arrays(predictions)
    final.save("evaluations/random_word.png")
    
    
if __name__ == "__main__":
    mask_a_word()