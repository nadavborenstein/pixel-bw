import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append("/home/knf792/PycharmProjects/pixel-2/")

import numpy as np

from datasets import load_dataset

from pixel_datasets.dataset_transformations import (
    SimpleTorchTransform,
)
from pixel_datasets.pixel_dataset_generator import PretrainingDataset

from pixel import PIXELForSequenceClassification
from glob import glob

import wandb
from tqdm.auto import tqdm

from pixel.utils.inference import load_general_model, encode_image, get_inference_font
from PIL import Image
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.cluster import SpectralClustering


MODEL_NAME = "Nadav/Pixel-real-scans-v3"
IMAGES_PATH = "/projects/copenlu/data/nadav/pixel/images_to_encode/reviews/"


def load_text_dataset(seed=42):
    text_dataset = load_dataset("amazon_reviews_multi", "en", split="validation")
    text_dataset = text_dataset.shuffle(seed=seed)
    text_dataset = text_dataset.filter(
        lambda x: len(x["review_body"].split()) > 20
        and len(x["review_body"].split()) < 30
    )
    text_dataset = text_dataset.rename_column("review_body", "text")
    return text_dataset


def load_image_dataset(seed=42):
    dataset = load_dataset(
        "Nadav/CaribbeanScans",
        split="test",
        cache_dir="/projects/copenlu/data/nadav/cache",
    )
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(1000))
    dataset = dataset.filter(lambda x: x["image"].size == (368, 368))
    dataset = dataset["image"]
    dataset = [image.copy().convert("RGB") for image in dataset]
    return dataset


def dump_reviews_images_to_disk():
    config = wandb.config
    print("Starting to dump images to disk...")
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)
    seed = config["seed"]
    rng = np.random.RandomState(seed)
    text_dataset = load_text_dataset(seed)

    transform = SimpleTorchTransform(config, rng=rng)
    dataset = PretrainingDataset(
        config=config, text_dataset=text_dataset, transform=transform, rng=rng
    )

    inference_font = get_inference_font()
    for i, instance in tqdm(enumerate(text_dataset)):
        image = dataset.generate_inference_image(
            instance["text"], inference_font, split_text=False
        )
        image = Image.fromarray(image)
        image_name = instance["review_id"].replace(" ", "_") + ".png"
        image.save(os.path.join(IMAGES_PATH, image_name))

    print(f"Done dumping {i} images!")


def get_images_to_encode(config):
    images_path = glob(IMAGES_PATH + "/*.png")
    print(f"Found {len(images_path)} images to encode!")
    images = [
        Image.open(image_path).copy().convert("RGB") for image_path in images_path
    ]
    names = [image_path.split("/")[-1].split(".")[0] for image_path in images_path]
    return images, names


def reduce_dim(embeddings):
    tsne = TSNE(n_components=2, random_state=0)
    vectors_2d = tsne.fit_transform(embeddings)
    return vectors_2d


def cluster(embeddings):
    n_clusters = 2

    # Create an instance of the SpectralClustering class
    sc = SpectralClustering(n_clusters=n_clusters)
    # Fit the model to the data
    sc.fit(embeddings)
    # Get the cluster labels for each vector
    labels = sc.labels_
    return labels


def plot(vectors_2d, lengths):
    df = pd.DataFrame({"x": vectors_2d[:, 0], "y": vectors_2d[:, 1], "length": lengths})
    ax = sns.scatterplot(data=df, x="x", y="y", hue="length", palette="flare")
    norm = plt.Normalize(min(lengths), max(lengths))
    sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
    sm.set_array([])

    ax.get_legend().remove()
    ax.figure.colorbar(sm, label="length")
    plt.savefig("review_length.png", dpi=300)


def plot_nothing(vectors_2d):
    df = pd.DataFrame({"x": vectors_2d[:, 0], "y": vectors_2d[:, 1]})
    ax = sns.scatterplot(data=df, x="x", y="y")
    plt.savefig("evaluations/real_scans.png", dpi=300)


def plot_class(vectors_2d, classes):
    df = pd.DataFrame({"x": vectors_2d[:, 0], "y": vectors_2d[:, 1], "class": classes})
    ax = sns.scatterplot(data=df, x="x", y="y", hue="class")

    plt.savefig("evaluations/review_class.png", dpi=300)


# def main():
#     model = load_general_model(
#         wandb.config, MODEL_NAME, model_type=PIXELForSequenceClassification
#     )
#     text_dataset = load_dataset("amazon_reviews_multi", "en", split="validation")
#     id_to_class = {
#         id_: class_
#         for id_, class_ in zip(
#             text_dataset["review_id"],
#             text_dataset["product_category"],
#         )
#     }
#     images_to_encode, names = get_images_to_encode(wandb.config)
#     embeddings = []
#     for image in tqdm(images_to_encode):
#         embeddings.append(encode_image(model, image))
#     embeddings = np.array(embeddings)
#     classes = [id_to_class[id_] for id_ in names]

#     embeddings_2d = reduce_dim(embeddings)
#     plot_class(embeddings_2d, classes)


def log_to_wandb(embeddings, images):
    # Create a "target" column
    df = pd.DataFrame({"image": [], "embeddings": []})
    
    df["embeddings"] = [list(embedding) for embedding in embeddings]
    df["image"] = images
    df.to_pickle("evaluations/embeddings.pkl")
    df["image"] = [wandb.Image(image.resize((46, 46))) for image in images]
    
    wandb.log({"embeddings": df})
    wandb.finish()


def main():
    model = load_general_model(
        wandb.config, MODEL_NAME, model_type=PIXELForSequenceClassification
    )

    images_to_encode = load_image_dataset(wandb.config["seed"])
    embeddings = []
    for image in tqdm(images_to_encode):
        embeddings.append(encode_image(model, image))
    embeddings = np.array(embeddings)

    embeddings_2d = reduce_dim(embeddings)
    plot_nothing(embeddings_2d)
    log_to_wandb(embeddings, images_to_encode)


if __name__ == "__main__":
    wandb.init(
        config="/home/knf792/PycharmProjects/pixel-2/configs/inference_config.yaml",
        project="pixel",
    )

    main()
