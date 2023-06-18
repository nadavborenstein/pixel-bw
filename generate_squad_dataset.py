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

dataset_root_path = "/projects/copenlu/data/nadav/Datasets/pixel_squad/"


def main(epoch):
    wandb.init(config="configs/squad_config.yaml", mode="disabled")
    wandb.config.update({"font_list_path": "pixel_datasets/fonts/antique_fonts.csv"}, allow_val_change=True)

    rng = np.random.RandomState(wandb.config["seed"])

    train_transform = SyntheticDatasetTransform(wandb.config, rng=rng)
    test_transform = SyntheticDatasetTransform(wandb.config, rng=rng)

    train_dataset = SquadDatasetForPixel(
        config=wandb.config, transform=train_transform, rng=rng, split="train"
    )

    test_dataset = SquadDatasetForPixel(
        config=wandb.config, transform=test_transform, rng=rng, split="validation"
    )
    train_dataset.set_epoch(epoch)
    counter = 0

    if not os.path.exists(f"{dataset_root_path}/train/"):
        os.makedirs(f"{dataset_root_path}/train/images")
        os.makedirs(f"{dataset_root_path}/train/labels")
    if not os.path.exists(f"{dataset_root_path}/test/"):
        os.makedirs(f"{dataset_root_path}/test/images")
        os.makedirs(f"{dataset_root_path}/test/labels")
        
    if epoch == 0:
        print("Saving test dataset")
        for batch in tqdm(test_dataset, total=len(test_dataset.text_dataset)):
            im = batch["pixel_values"].numpy().astype("uint8").transpose(1, 2, 0)
            mask = batch["label_mask"].numpy().astype("uint8")
            np.save(f"{dataset_root_path}/test/labels/{counter}.npy", mask)

            im = Image.fromarray(im)
            im.save(f"{dataset_root_path}/test/images/{counter}.png")

            counter += 1
        print("Done saving test dataset")
    else:
        print("Saving train dataset")
        for batch in tqdm(train_dataset, total=len(train_dataset.text_dataset)):
            if os.path.exists(f"{dataset_root_path}/train/images/{epoch}-{counter}.png"):
                counter += 1
                continue
            else:
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
