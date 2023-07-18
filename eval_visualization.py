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
from pixel_datasets.utils.dataset_utils import generate_patch_mask
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
    parse_squad_outputs,
    predict_squad,
    load_model_for_squad,
)
from PIL import Image
import matplotlib.pyplot as plt
import torch
import pickle
import platform


def load_real_scans(args, n=100):
    visualization_dataset = load_dataset("Nadav/CaribbeanScans", split="test", cache_dir=args.dataset_cache_dir)
    visualization_dataset = visualization_dataset.shuffle().select(range(n))
    visualization_dataset = visualization_dataset.filter(lambda x: x["image"].size == (368, 368))
    rng = np.random.RandomState(42)
    visualization_examples = []
    for sample in tqdm(visualization_dataset):
        sample["patch_mask"] = generate_patch_mask(args, rng, (368, 368))[1]
        sample["pixel_values"] = np.array(sample["image"])
        visualization_examples.append(sample)
    return visualization_examples


def random_masking_real_samples(args, save_all_figs=False):
    visualization_examples = load_real_scans(args, 16)

    model = load_model_for_pretraining(
        wandb.config,
        "Nadav/Pixel-real-scans-v3",
    )

    predictions = []
    for example in tqdm(visualization_examples):
        outputs = predict(model, example["pixel_values"], example["patch_mask"])
        prediction = parse_outputs(outputs, model, example["pixel_values"])
        predictions.append(prediction)

    for i in range(len(predictions)):
        mask = visualization_examples[i]["patch_mask"]
        pixel_mask = convert_patch_mask_to_pixel_mask(mask)
        only_edges = get_mask_edges(pixel_mask, 3)
        merged = merge_mask_with_image(
            only_edges, np.array(predictions[i]), colour=(0, 0, 0), alpha=0.1
        )
        if save_all_figs:
            original = Image.fromarray(visualization_examples[i]["pixel_values"])
            pixel_mask = Image.fromarray((pixel_mask * 255).astype(np.uint8))
            predicted = Image.fromarray(predictions[i]) 
            merged_image = Image.fromarray(merged)
            for name, im in zip(["original", "pixel_mask", "predicted", "merged"], [original, pixel_mask, predicted, merged_image]):
                im.save(f"evaluations/sample_scans/real_scans_{i}_{name}.png")
        
        predictions[i] = merged

    final = plot_arrays(predictions)
    final.save("evaluations/completions_results/real_validation_samples.png")


def random_masking(args):
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


def generate_single_mask_real_scans(args):
    visualization_dataset = load_real_scans(args, n=64)
    filtered_dataset = []
    for sample in tqdm(visualization_dataset):
        im = np.array(sample["pixel_values"])
        try:
            pixel_mask, ocred_image, random_word = mask_single_word_from_scan(im)
        except ValueError:
            continue
        patch_mask = convert_pixel_mask_to_patch_mask(pixel_mask, 16, 0.3)
        sample["patch_mask"] = (
            torch.from_numpy(patch_mask).flatten().type(torch.float32)
        )
        sample["ocr"] = ocred_image
        sample["random_word"] = random_word
        filtered_dataset.append(sample)

    pickle.dump(
        filtered_dataset, open("evaluations/visualization_examples_real.pkl", "wb")
    )
    print("Saved visualization examples to disk")
    

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
        visualization_examples_all = pickle.load(
            open("evaluations/visualization_examples_real.pkl", "rb")
        )
        model = load_model_for_pretraining(
            wandb.config,
            "Nadav/Pixel-real-scans-v3",
        )
        predictions = []
        visualization_examples = []
        for example in tqdm(visualization_examples_all):
            try:
                outputs = predict(model, example["pixel_values"], example["patch_mask"])
                prediction = parse_outputs(outputs, model, example["pixel_values"])
                predictions.append(prediction)
                visualization_examples.append(example)
            except ValueError:
                continue

        for i in range(len(predictions)):
            mask = visualization_examples[i]["patch_mask"]
            random_word = visualization_examples[i]["random_word"]
            mask = mask.numpy()
            pixel_mask = convert_patch_mask_to_pixel_mask(mask)
            only_edges = get_mask_edges(pixel_mask, 3)
            merged = merge_mask_with_image(
                only_edges, np.array(predictions[i]), colour=(0, 0, 0), alpha=0.1
            )
            predictions[i] = merged
            merged = Image.fromarray(merged)
            merged.save(f"evaluations/completions_results/real_scans_{random_word}.png")
            
        final = plot_arrays(
            predictions,
            titles=[example["random_word"] for example in visualization_examples],
        )
        final.save("evaluations/completions_results/random_words_real_scans.png")


def save_synthetic_dataset_samples(args):
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


def visualise_squad(args):
    model = load_model_for_squad(args, "/projects/copenlu/data/nadav/pixel/pixel_squad_mixed_with_hist/checkpoint-2400/")
    dataset = load_from_disk("/projects/copenlu/data/nadav/Datasets/runaways_visual/dataset")
    dataset = dataset["test"]
    dataset = dataset.shuffle()
    dataset = dataset.filter(lambda x: np.max(x["label"]) == 1)

    for i in tqdm(range(0, 32)):
        instance = dataset[i]
        image = np.asarray(instance["image"].copy().convert("RGB"))
        label = np.array(instance["label"])
        prediction = predict_squad(model, image)
        parsed_predictions = parse_squad_outputs(prediction, image, label, method="saliency")
        
        pixel_mask = convert_patch_mask_to_pixel_mask(label)
        only_edges = get_mask_edges(pixel_mask, 3)
        merged = merge_mask_with_image(
            only_edges, parsed_predictions, colour=(0, 0, 0), alpha=0.1
        )
        merged = Image.fromarray(merged.astype(np.uint8))
        merged.save(f"evaluations/runaways/saliency_{i}.png")
        


if __name__ == "__main__":
    wandb.init(
        config="/home/knf792/PycharmProjects/pixel-2/configs/inference_config.yaml",
        mode="disabled",
    )
    random_masking_real_samples(wandb.config)
