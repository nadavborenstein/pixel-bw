from pixel_datasets.glue_dataset_generator import GlueDatasetForPixel
from pixel_datasets.dataset_transformations import (
    SyntheticDatasetTransform,
    SimpleTorchTransform,
)
import numpy as np
import pandas as pd
from PIL import Image
import datasets
import wandb
import glob
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
import sys


TASKS = ["sst2", "wnli", "cola", "mnli", "mrpc", "qnli", "qqp", "rte", "stsb"]


def get_datasets(args, seed=42):
    rng = np.random.RandomState(seed)
    transform = SyntheticDatasetTransform(args, rng)

    train_dataset = GlueDatasetForPixel(
        config=args, task=args.task_name, split="train", transform=transform, rng=rng
    )
    test_dataset = GlueDatasetForPixel(
        config=args,
        task=args.task_name,
        split="validation",
        transform=transform,
        rng=rng,
    )

    return train_dataset, test_dataset


def convert_instance_to_dataset(instance):
    new_instance = {}
    image = instance["pixel_values"].numpy()
    image = (image * 255).astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    new_instance["image"] = Image.fromarray(image)
    new_instance["label"] = instance["label"]
    return new_instance


def main(task, split):
    wandb.init(
        config="/home/knf792/PycharmProjects/pixel-2/configs/glue_config_low_noise.yaml",
        mode="disabled",
    )
    task = TASKS[int(task)]
    print(task)
    wandb.config.update({"task_name": task}, allow_val_change=True)
    train_dataset, test_dataset = get_datasets(wandb.config)

    new_test_dataset_as_dict = {"image": [], "label": []}
    new_train_dataset_as_dict = {"image": [], "label": []}

    if split != "train":
        for i in tqdm(range(len(test_dataset))):
            instance = convert_instance_to_dataset(test_dataset[i])
            new_test_dataset_as_dict["image"].append(instance["image"])
            new_test_dataset_as_dict["label"].append(instance["label"])

    if split != "test":
        for i in tqdm(range(len(train_dataset))):
            instance = convert_instance_to_dataset(train_dataset[i])
            new_train_dataset_as_dict["image"].append(instance["image"])
            new_train_dataset_as_dict["label"].append(instance["label"])

    num_labels = len(set(new_test_dataset_as_dict["label"]))
    print(f"Number of labels: {num_labels}")

    feature = (
        datasets.Value("float32")
        if task == "stsb"
        else datasets.ClassLabel(num_classes=num_labels)
    )

    new_test_dataset = Dataset.from_dict(
        new_test_dataset_as_dict,
        features=datasets.Features(
            {
                "image": datasets.Image(),
                "label": feature,
            }
        ),
    )
    new_train_dataset = Dataset.from_dict(
        new_train_dataset_as_dict,
        features=datasets.Features(
            {
                "image": datasets.Image(),
                "label": feature,
            }
        ),
    )
    if split == "train":
        dataset = DatasetDict({"train": new_train_dataset})
    elif split == "test":
        dataset = DatasetDict({"validation": new_test_dataset})
    else:
        dataset = DatasetDict(
            {"train": new_train_dataset, "validation": new_test_dataset}
        )
    print(dataset)

    dataset.save_to_disk(
        f"/projects/copenlu/data/nadav/Datasets/pixel_glue_{task}_low_noise/dataset"
    )
    dataset.push_to_hub(
        f"pixel_glue_{task}_low_noise", token="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"
    )


if __name__ == "__main__":
    task_id, split = sys.argv[1:]
    main(task_id, split)