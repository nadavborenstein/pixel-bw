import os
import sys
import matplotlib.pyplot as plt
import skimage.io as io
from glob import glob
import random
from typing import List, Tuple
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
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
from PIL import Image
from tqdm.auto import tqdm
import pickle
import time
from sklearn.cluster import KMeans
import multiprocessing as mp
from sklearn.metrics.pairwise import cosine_distances

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append("/home/knf792/PycharmProjects/pixel-2/")


from pixel.utils.inference import encode_images, load_general_model
from pixel import PIXELForSequenceClassification
import wandb

DATASET_PATH = "/projects/copenlu/data/nadav/pixel/data/real_scans/"


def resize_caribbean_image(image: np.ndarray, size: Tuple[int, int] = (368, 368)):
    """
    A function to resize an image.
    """
    if image.shape[0] != size[0] or image.shape[1] != size[1]:
        white_image = np.ones((size[0], size[1], 3)) * 255
        white_image[: image.shape[0], : image.shape[1], :] = image
        image = white_image
    return image


def read_images_multiprocessing(paths):
    # Create a pool of workers with the number of available CPUs
    pool = mp.Pool(mp.cpu_count())
    # Use pool.map to apply the read_image function to each path in the list and get the results as a list of images
    images = pool.map(io.imread, paths)
    # Close the pool and wait for the workers to finish
    pool.close()
    pool.join()
    # Return the list of images
    return images


def get_caribbean_paths(root_path: str, n_images):
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

    images_paths = glob(os.path.join(root_path, "train", "*.png"))
    random.Random(42).shuffle(images_paths)
    images_paths = images_paths[:n_images]

    return images_paths


def construct_caribbean_dataset(
    model: PIXELForSequenceClassification,
    num_images: int = 10000,
    processing_batch_size: int = 1024,
    batch_size=32,
):
    """
    A function to construct a dataset from a list of images.
    """
    encodings = []
    paths = get_caribbean_paths(DATASET_PATH, num_images)
    pbar = tqdm(total=num_images)
    for batch_id in range(0, num_images, processing_batch_size):
        batch_paths = paths[batch_id : batch_id + processing_batch_size]
        time_before = time.time()
        batch_images = read_images_multiprocessing(batch_paths)
        time_after = time.time()

        batch_images = [resize_image(wandb.config, image) for image in batch_images]
        excecution_time = time_after - time_before

        batch_encodings = encode_images(model, batch_images, batch_size=batch_size)
        encodings.append(batch_encodings)
        pbar.set_description(f"excecution time: {excecution_time:.2f}")
        pbar.update(processing_batch_size)

    encodings = np.concatenate(encodings)
    return encodings, paths


def resize_image(args, image):
    """
    Resize the image to the specified size.
    """
    if type(image) == np.ndarray:
        image = Image.fromarray(image)

    width, length = image.size
    if (
        width != args.image_width
    ):  # resize the image if the width is not the same as the specified width, without changing the aspect ratio
        ratio = width / args.image_width
        new_length = int(length / ratio)
        image = image.resize((args.image_width, new_length), Image.LANCZOS)

    if image.size[1] > args.image_height:  # crop the image if it's too long
        image = image.crop((0, 0, image.size[0], args.image_height))

    if image.size[1] < args.image_width:  # pad the image if it's too short
        embedded_image = Image.new(
            "RGB",
            (args.image_width, args.image_height),
            (255, 255, 255),
        )
        embedded_image.paste(image, (0, 0))
        image = embedded_image

    return image


def load_ad(image_id):
    """
    load an ad image.
    """
    try:
        image = Image.open(
            f"/projects/copenlu/data/nadav/pixel/runaways_scans/{image_id}.png"
        )
    except FileNotFoundError:
        return None
    resized_image = resize_image(wandb.config, image)
    resized_image = np.array(resized_image)
    return resized_image


def load_text_runaway_ad_dataset():
    """
    load the runaway ad dataset.
    """
    text_dataset = load_from_disk("/projects/copenlu/data/nadav/pixel/runaway_dataset")
    all_ids = list(set(map(lambda x: x["ID"][1:5], text_dataset)))

    random.shuffle(all_ids)
    test_size = int(len(all_ids) * 0.2)
    test_ids = all_ids[:test_size]
    train_ids = all_ids[test_size:]

    train_images = [load_ad(image_id) for image_id in train_ids]
    test_images = [load_ad(image_id) for image_id in test_ids]
    train_images = [image for image in train_images if image is not None]
    test_images = [image for image in test_images if image is not None]

    return train_images, test_images


def load_shipping_ads():
    paths = glob("pixel_datasets/shipping_ads/*.jpg")
    images = [io.imread(path) for path in paths]
    resized_images = [resize_image(wandb.config, image) for image in images]
    resized_images = [np.array(image) for image in resized_images if image is not None]
    return resized_images


def embed_images(model, images, batch_size=32):
    """
    embed images using the specified model.
    """
    encodings = encode_images(model, images, batch_size=batch_size)
    return encodings


def k_means(vectors, n=5):
    # Create a KMeans object with n clusters
    kmeans = KMeans(n_clusters=n, random_state=42)
    # Fit the KMeans object to the vectors
    kmeans.fit(vectors)
    # Return the cluster centers
    return kmeans.cluster_centers_


def encode_runaway_images(model, dataset, n_clusters=5, batch_size=32):
    """
    encode the runaway ad images.
    """
    train_encodings = embed_images(model, dataset, batch_size=batch_size)
    cluster = k_means(train_encodings, n=n_clusters)
    return cluster


def load_one_image(path: str):
    image = Image.open(path)
    resized_image = resize_image(wandb.config, image)
    resized_image = np.array(resized_image)
    return resized_image


def semantic_search():
    MODEL_NAME = "Nadav/Pixel-real-scans-v3"
    N_CLUSTERS = 2
    BATCH_SIZE = 24

    print("Loading model...")
    model = load_general_model(
        wandb.config, MODEL_NAME, model_type=PIXELForSequenceClassification
    )
    print("Loading dataset...")
    if not os.path.exists("caribbean_dataset_embedding.npy"):
        dataset_embedding, dataset_paths = construct_caribbean_dataset(
            model, num_images=500000, batch_size=BATCH_SIZE
        )
        np.save("caribbean_dataset_embedding.npy", dataset_embedding)
        pickle.dump(dataset_paths, open("caribbean_dataset_paths.pkl", "wb"))
    else:
        dataset_embedding = np.load("caribbean_dataset_embedding.npy")
        dataset_paths = pickle.load(open("caribbean_dataset_paths.pkl", "rb"))

    print(dataset_embedding.shape)
    print("Loading runaway dataset...")

    target_image = load_one_image("/home/knf792/PycharmProjects/pixel-2/pixel_datasets/shipping_ads/WhatsApp Image 2023-10-19 at 16.11.33.jpeg")
    clusters = embed_images(model, [target_image])
    print(clusters.shape)

    print("Calculating distances...")
    for cluster in clusters:
        distances = cosine_distances(dataset_embedding, [cluster]).flatten()
        print(distances.shape, max(distances), min(distances))
        sorted_indices = np.argsort(distances)
        sorted_paths = np.array(dataset_paths)[sorted_indices]
        sorted_distances = distances[sorted_indices]

        print("most similar images:")
        with open("similar_images.txt", "w") as f:
            for i in range(10):
                f.write(f"'{sorted_paths[i]}',\n")
                print(sorted_paths[i], sorted_distances[i])


if __name__ == "__main__":
    wandb.offline = True
    wandb.init(
        config="/home/knf792/PycharmProjects/pixel-2/configs/inference_config.yaml",
        project="pixel",
    )

    semantic_search()
