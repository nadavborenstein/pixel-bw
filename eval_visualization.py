import numpy as np
from datasets import load_dataset, load_from_disk, interleave_datasets
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
from pixel_datasets.utils.squad_utils import (
    convert_pixel_mask_to_patch_mask,
)
import wandb
from tqdm.auto import tqdm
from pixel.utils.inference import (
    load_model_for_pretraining,
    predict,
    parse_outputs,
    get_inference_font,
)
from PIL import Image
import matplotlib.pyplot as plt
import torch
import pickle
import platform


def random_masking():
    visualization_dataset = load_dataset("wikipedia", "20220301.simple", split="train")
    visualization_dataset = visualization_dataset.filter(
        lambda x: len(x["text"].split()) > 200
    )

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


def generate_single_mask_scans():
    visualization_dataset = load_dataset("wikipedia", "20220301.simple", split="train")
    visualization_dataset = visualization_dataset.filter(
        lambda x: len(x["text"].split()) > 200
    )

    # visualization_dataset = load_dataset("wikipedia", "20220301.simple")["train"]
    print(visualization_dataset[0])

    rng = np.random.RandomState(42)
    transform = SyntheticDatasetTransform(wandb.config, rng=rng)
    simple_transform = SimpleTorchTransform(wandb.config, rng=rng)

    dataset = PretrainingDataset(
        config=wandb.config,
        text_dataset=visualization_dataset,
        transform=transform,
        rng=rng,
    )
    font = get_inference_font()
    font.font_size = 20
    visualization_examples = []
    for text_sample in tqdm(visualization_dataset["text"][:9]):
        image = dataset.generate_inference_image(
            text_sample, split_text=True, clean_text=True, font=font
        )
        image = simple_transform(image)

        visualization_examples.append({"pixel_values": image})

    for sample in visualization_examples:
        im = convert_torch_tensor_to_image(sample["pixel_values"])
        pixel_mask, ocred_image, random_word = mask_single_word_from_scan(im)
        patch_mask = convert_pixel_mask_to_patch_mask(pixel_mask, 16, 0.3)
        sample["patch_mask"] = (
            torch.from_numpy(patch_mask).flatten().type(torch.float32)
        )
        sample["ocr"] = ocred_image
        sample["random_word"] = random_word

    pickle.dump(
        visualization_examples, open("evaluations/visualization_examples.pkl", "wb")
    )
    print("Saved visualization examples to disk")


def mask_a_word(generate=False):
    if generate:
        generate_single_mask_scans()
    else:
        visualization_examples = pickle.load(
            open("evaluations/visualization_examples.pkl", "rb")
        )
        model = load_model_for_pretraining(
            wandb.config,
            "Nadav/PretrainedPHD-v3",
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

        final = plot_arrays(
            predictions,
            titles=[example["random_word"] for example in visualization_examples],
        )
        final.save("evaluations/random_word.png")


def save_dataset_samples(args):
    """
    Saves a random sample of fake images from the dataset
    """
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
    dataset_sizes = [ds._info.splits.total_num_examples for ds in train_text_datasets]
    combined_size = sum(dataset_sizes)
    dataset_sampling_probs = [d_size / combined_size for d_size in dataset_sizes]
    train_text_dataset = interleave_datasets(
        train_text_datasets,
        probabilities=dataset_sampling_probs,
        seed=args.seed,
    )
    rng = np.random.RandomState(112)
    train_transform = SimpleTorchTransform(args, rng=rng)

    train_dataset = PretrainingDataset(
        config=args,
        text_dataset=train_text_dataset,
        transform=train_transform,
        rng=rng,
    )

    figures = []
    counter = 0
    for batch in train_dataset:
        if counter == 16:
            break
        im = batch["pixel_values"].numpy().transpose(1, 2, 0)
        im = (im * 255).astype(np.uint8)
        Image.fromarray(im).save(
            f"/home/knf792/PycharmProjects/pixel-2/pixel_datasets/results/samples/synthetics_no_noise_{counter}.png"
        )
        figures.append(im)
        counter += 1

    im = plot_arrays(figures)
    im.save(
        "/home/knf792/PycharmProjects/pixel-2/pixel_datasets/results/samples/synthetics_no_noise_sample_grid.png"
    )


if __name__ == "__main__":
    wandb.init(
        config="/home/knf792/PycharmProjects/pixel-2/configs/pretraining_config.yaml",
        mode="disabled",
    )
    save_dataset_samples(wandb.config)
