from pixel_datasets.squad_dataset_for_pixel import SquadDatasetForPixel
from pixel_datasets.dataset_transformations import SyntheticDatasetTransform, SimpleTorchTransform
import numpy as np
import pandas as pd
from PIL import Image
import datasets
import wandb
import glob
from pixel_datasets.utils.dataset_utils import CustomFont
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import sys

dataset_root_path = "/projects/copenlu/data/nadav/Datasets/pixel_squad_cannon/"

font_list = pd.read_csv("/home/knf792/PycharmProjects/pixel-2/pixel_datasets/fonts/antique_fonts.csv")
index = 6
random_font = font_list["path"][index]
random_font = random_font.replace(" ", "_")  # fixing spaces in the path
font_name = random_font.split(".")[0].split("/")[1]

font_size = font_list["base_size"][index]
font_size = int(font_size // 1.6)
custom_font = CustomFont(
    file_name=random_font, font_name=font_name.title(), font_size=font_size
)


def main(epoch):
    wandb.init(config="configs/squad_config.yaml", mode="disabled")
    wandb.config.update({"randomize_font": False}, allow_val_change=True)

    rng = np.random.RandomState(2)

    train_transform = SimpleTorchTransform(wandb.config, rng=rng)
    test_transform = SimpleTorchTransform(wandb.config, rng=rng)

    train_dataset = SquadDatasetForPixel(
        config=wandb.config, transform=train_transform, rng=rng, split="train", font=custom_font
    )

    test_dataset = SquadDatasetForPixel(
        config=wandb.config, transform=test_transform, rng=rng, split="validation", font=custom_font
    )
    train_dataset.set_epoch(0)
    counter = 0

    print("Saving train dataset")
    for batch in tqdm(train_dataset, total=len(train_dataset.text_dataset)):
        im = batch["pixel_values"].numpy().astype("uint8").transpose(1, 2, 0)
        mask = batch["label_mask"].numpy().astype("uint8")
        np.save(f"{dataset_root_path}/train/labels/{epoch}-{counter}.npy", mask)

        im = Image.fromarray(im)
        im.save(f"{dataset_root_path}/train/images/{epoch}-{counter}.png")

        counter += 1
    print("Done saving train dataset")


if __name__ == "__main__":
    epoch = int(sys.argv[1])
    main(epoch)
