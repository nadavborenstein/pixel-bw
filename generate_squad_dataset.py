from squad_dataset_for_pixel import SquadDatasetForPixel
from dataset_transformations import SyntheticDatasetTransform, SimpleTorchTransform
import numpy as np
from PIL import Image
import wandb
from tqdm.auto import tqdm
import sys

dataset_root_path = "/projects/copenlu/data/nadav/Datasets/pixel_squad/"


def main(epoch):
    wandb.init(
        config="/home/knf792/PycharmProjects/pixel-2/configs/squad_config.yaml",
        mode="disabled",
    )
    rng = np.random.RandomState(2)

    train_transform = SyntheticDatasetTransform(wandb.config, rng=rng)

    train_dataset = SquadDatasetForPixel(
        config=wandb.config, transform=train_transform, rng=rng, split="train"
    )

    train_dataset.set_epoch(epoch)
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
