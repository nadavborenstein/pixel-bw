from pixel_datasets.glue_dataset_generator import GlueDatasetForPixel
from pixel_datasets.dataset_transformations import (
    SyntheticDatasetTransform,
    SimpleTorchTransform,
)
from pixel_datasets.utils.dataset_utils import simple_ocr
import numpy as np
import pandas as pd
from PIL import Image
import datasets
import wandb
import glob
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from wandb.sdk.wandb_config import Config
import sys
import pickle


TASKS = ["sst2", "wnli", "cola", "mnli", "mrpc", "qnli", "qqp", "rte", "stsb"]


def get_datasets(args: Config):
    seed = args["seed"]
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
    ocred_image = simple_ocr(image)
    new_instance["text"] = ocred_image
    new_instance["label"] = instance["label"]
    return new_instance


def main(task_id, chunk, total_chunk):
    wandb.init(
        config="/home/knf792/PycharmProjects/pixel-2/configs/glue_config.yaml",
        mode="disabled",
    )
    task = TASKS[task_id]
    print(task)
    wandb.config.update({"task_name": task}, allow_val_change=True)

    train_dataset, test_dataset = get_datasets(wandb.config)
    new_test_dataset_as_dict = {"text": [], "label": []}
    new_train_dataset_as_dict = {"text": [], "label": []}

    if chunk == 0:
        for i in tqdm(range(len(test_dataset))):
            instance = convert_instance_to_dataset(test_dataset[i])
            new_test_dataset_as_dict["text"].append(instance["text"])
            new_test_dataset_as_dict["label"].append(instance["label"])
        pickle.dump(new_test_dataset_as_dict, open(f"test_{task}.pkl", "wb"))
    else:
        train_dataset.text_dataset = train_dataset.text_dataset.select(
            range(chunk - 1, len(train_dataset), total_chunk - 1)
        )
        counter = 0
        for epoch in range(5):
            for i in tqdm(range(len(train_dataset))):
                instance = convert_instance_to_dataset(train_dataset[i])
                new_train_dataset_as_dict["text"].append(instance["text"])
                new_train_dataset_as_dict["label"].append(instance["label"])
                counter += 1

        pickle.dump(new_train_dataset_as_dict, open(f"train_{task}_{chunk}.pkl", "wb"))
    print("Done")
    exit()
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
                "text": datasets.Value("string"),
                "label": feature,
            }
        ),
    )
    new_train_dataset = Dataset.from_dict(
        new_train_dataset_as_dict,
        features=datasets.Features(
            {
                "text": datasets.Value("string"),
                "label": feature,
            }
        ),
    )
    dataset = DatasetDict({"train": new_train_dataset, "validation": new_test_dataset})
    print(dataset)

    dataset.save_to_disk(
        f"/projects/copenlu/data/nadav/Datasets/pixel_glue_{task}_noisy_ocr/dataset"
    )
    dataset.push_to_hub(
        f"pixel_glue_{task}_noisy_ocr", token="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"
    )


if __name__ == "__main__":
    import time
    task_id, chunk, total_chunk = list(map(int, sys.argv[1:]))
    main(task_id, chunk, total_chunk)
