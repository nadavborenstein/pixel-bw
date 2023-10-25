import os
import sys
import matplotlib.pyplot as plt
import skimage.io as io
from glob import glob
import random
from typing import List, Tuple
import numpy as np
from datasets import Dataset, DatasetDict
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
import pandas as pd


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append("/home/knf792/PycharmProjects/pixel-2/")

from pixel.utils.inference import encode_images, load_general_model
from pixel import PIXELForSequenceClassification
import wandb


def resize_image(image: np.ndarray, size: Tuple[int, int] = (368, 368)):
    """
    A function to resize an image.
    """
    if image.shape[0] != size[0] or image.shape[1] != size[1]:
        white_image = np.ones((size[0], size[1], 3)) * 255
        white_image[: image.shape[0], : image.shape[1], :] = image
        image = white_image
    return image


def load_images(root_path: str, n_images: int, test_ratio: float = 0.2):
    """
    A function to load images from a folder.
    """
    assert os.path.exists(root_path), "root_path does not exist"
    assert os.path.exists(
        os.path.join(root_path, "train")
    ), "train folder does not exist"
    assert os.path.exists(
        os.path.join(root_path, "evaluation")
    ), "evaluation folder does not exist"

    num_test_images = int(n_images * test_ratio)
    num_train_images = n_images - num_test_images

    train_paths = glob(os.path.join(root_path, "train", "*.png"))
    random.Random(42).shuffle(train_paths)
    train_paths = train_paths[:num_train_images]

    test_paths = glob(os.path.join(root_path, "evaluation", "*.png"))
    random.Random(42).shuffle(test_paths)
    test_paths = test_paths[:num_test_images]

    train_images = [(io.imread(path), path) for path in train_paths]
    test_images = [(io.imread(path), path) for path in test_paths]

    train_images = [(resize_image(image), path) for image, path in train_images]
    test_images = [(resize_image(image), path) for image, path in test_images]

    return train_images, test_images


def construct_dataset(
    model: PIXELForSequenceClassification,
    images: Tuple[np.ndarray, str],
    top_ten_newspapers: List[str] = None,
):
    """
    A function to construct a dataset from a list of images.
    """
    encodings = encode_images(model, [image[0] for image in images], batch_size=32)
    encodings = np.split(encodings, encodings.shape[0])
    encodings = [encoding[0] for encoding in encodings]
    df = pd.DataFrame({"path": [image[1] for image in images], "encoding": encodings})
    df["year"] = df["path"].apply(lambda x: int(re.findall(r"\d\d\d\d", x)[0]))
    df["decade"] = df["year"].apply(lambda x: str(x // 10) + "0")
    df["century"] = df["year"].apply(lambda x: str(x // 100) + "00")
    df["newspaper"] = df["path"].apply(
        lambda x: x[x.rfind("/") + 1 : x.find("published") - 1]
    )

    if top_ten_newspapers is None:
        top_ten_newspapers = df["newspaper"].value_counts()[:10].index.tolist()

    dataset: Dataset = Dataset.from_pandas(df)

    # filter out samples where one of the labels is missing
    dataset = dataset.filter(
        lambda x: x["year"] != 0 and x["decade"] != 0 and x["century"] != 0
    )
    dataset = dataset.filter(lambda x: x["newspaper"] != "")
    dataset = dataset.filter(lambda x: x["newspaper"] in top_ten_newspapers)

    top_ten_newspapers = dataset["newspaper"]
    return dataset, top_ten_newspapers


def get_datasets(
    model: PIXELForSequenceClassification,
    root_path: str,
    n_images: int,
    test_ratio: float = 0.2,
):
    """
    A function to load images from a folder and construct a dataset.
    """
    train_images, test_images = load_images(root_path, n_images, test_ratio)
    train_dataset, top_10_newspapers = construct_dataset(model, train_images)
    test_dataset, _ = construct_dataset(model, test_images, top_10_newspapers)

    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    return dataset


def train_linear_model(dataset, target: str = "century"):
    model = LogisticRegression()
    model.fit(dataset["train"]["encoding"], dataset["train"][target])
    return model


def evaluate_model(model, dataset, target: str = "century"):
    preds = model.predict(dataset["test"]["encoding"])
    print(classification_report(dataset["test"][target], preds))
    return preds


def main():
    MODEL_NAME = "Nadav/Pixel-real-scans-v3"
    DATASET_PATH = "/projects/copenlu/data/nadav/pixel/data/Caribbean_scans_dataset/"
    print("Loading model...")
    model = load_general_model(
        wandb.config, MODEL_NAME, model_type=PIXELForSequenceClassification
    )
    print("Loading dataset...")
    dataset = get_datasets(model, DATASET_PATH, 20000)
    print("Training model...")
    model = train_linear_model(dataset, target="newspaper")
    print("Evaluating model...")
    preds = evaluate_model(model, dataset, target="newspaper")
    return model, dataset, preds


if __name__ == "__main__":
    wandb.offline = True
    wandb.init(
        config="/home/knf792/PycharmProjects/pixel-2/configs/inference_config.yaml",
        project="pixel",
    )

    main()
