from collections import defaultdict
from typing import Callable

import datasets
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pixel import get_attention_mask


class HistoricDatasetForPretraining(Dataset):
    def __init__(
        self,
        args,
        data: datasets.Dataset,
        rng: np.random.RandomState,
        preprocessing_function: Callable = None,
        transform=None,
        labeled=False,
    ):
        self.args = args
        data = data.shuffle(rng.randint(0, 2**32 - 1))
        if labeled:
            data = data.filter(lambda x: x[self.args.label_column_name] is not None)
        self.data = data
        if preprocessing_function is not None:
            self.data = data.map(preprocessing_function)
        elif labeled:
            preprocessing_function = self.get_preprocessing_function()
            self.data = data.map(preprocessing_function)
        self.transform = transform
        self.rng = rng
        self.torch_transform = ToTensorV2()
        self.labeled = labeled
        self.num_patches = 529

    def __len__(self):
        return len(self.data)

    def get_preprocessing_function(self):
        """
        Return the preprocessing function for the dataset.
        """
        categories = list(set(self.data[self.args.label_column_name]))
        categories.sort()
        label_map = defaultdict(
            lambda: self.args.number_of_labels - 1
        )  # if in test or train dataset we don't have a class, we can still return a label
        for i, category in enumerate(categories):
            label_map[category] = i

        def _categorical_to_int(example):
            example["label"] = label_map[example[self.args.label_column_name]]
            return example

        return _categorical_to_int

    def resize_image(self, image):
        """
        Resize the image to the specified size.
        """
        if type(image) == np.ndarray:
            image = Image.fromarray(image)

        width, length = image.size
        if (
            width != self.args.image_width
        ):  # resize the image if the width is not the same as the specified width, without changing the aspect ratio
            ratio = width / self.args.image_width
            new_length = int(length / ratio)
            image = image.resize((self.args.image_width, new_length), Image.ANTIALIAS)

        if image.size[1] > self.args.image_height:  # crop the image if it's too long
            image = image.crop((0, 0, image.size[0], self.args.image_height))

        if self.args.embed_real_image:
            if image.size[1] < self.args.image_width:
                embedded_image = Image.new(
                    "RGB",
                    (self.args.image_width, self.args.image_height),
                    (255, 255, 255),
                )
                embedded_image.paste(image, (0, 0))
                image = embedded_image
            num_patches = self.num_patches

        else:  # we add black pixels to the image to make it square, and change the attention mask accordingly
            if image.size[1] == self.args.image_width:
                num_patches = self.num_patches
            else:
                num_patches = (
                    (self.args.image_width - image.size[1])
                    // self.args.patch_base_size[0]
                ) * (self.args.image_height // self.args.patch_base_size[1])
                embedded_image = Image.new(
                    "RGB",
                    (self.args.image_width, self.args.image_height),
                    (255, 255, 255),
                )
                embedded_image.paste(image, (0, 0))
                image = embedded_image
        return image, num_patches

    def _torch_transform(self, image):
        image = np.asarray(image).astype("float32")
        image = image / 255.0
        image = self.torch_transform(image=image)["image"]
        return image

    def generate_random_mask(self, image_size):
        """
        Generate a random mask for the image.
        """
        mask = np.zeros(image_size)
        pixels_masked = 0
        while (
            pixels_masked / (image_size[0] * image_size[1])
        ) < self.args.mask_block_probability:
            patch_height = (
                self.rng.randint(
                    self.args.mask_min_merged_blocks_size[0],
                    self.args.mask_max_merged_blocks_size[0] + 1,
                )
                * self.args.mask_block_size[0]
            )
            patch_width = (
                self.rng.randint(
                    self.args.mask_min_merged_blocks_size[1],
                    self.args.mask_max_merged_blocks_size[1] + 1,
                )
                * self.args.mask_block_size[1]
            )

            for i in range(10):
                random_mask_location_x = self.rng.choice(
                    np.arange(
                        0, image_size[0] - patch_height, self.args.mask_block_size[0]
                    )
                )
                random_mask_location_y = self.rng.choice(
                    np.arange(
                        0, image_size[1] - patch_width, self.args.mask_block_size[1]
                    )
                )

                slice = mask[
                    random_mask_location_x : random_mask_location_x + patch_height,
                    random_mask_location_y : random_mask_location_y + patch_width,
                ]
                if np.sum(slice) > 0:
                    continue
                else:
                    mask[
                        random_mask_location_x : random_mask_location_x + patch_height,
                        random_mask_location_y : random_mask_location_y + patch_width,
                    ] = 1

                    pixels_masked += patch_height * patch_width
                    break

        small_mask = mask[
            :: self.args.patch_base_size[0], :: self.args.patch_base_size[1]
        ].flatten()
        return mask, small_mask

    def __getitem__(self, item):
        original_image = self.data[item]["image"]
        pixel_values, num_patches = self.resize_image(original_image)
        attention_mask = get_attention_mask(num_patches)
        if not self.labeled:
            _, patch_mask = self.generate_random_mask(
                (self.args.image_width, self.args.image_height)
            )
            patch_mask = torch.tensor(patch_mask, dtype=torch.float32)
            label = None
        else:
            patch_mask = None
            label = self.data[item]["label"]

        if self.transform:
            pixel_values = np.asarray(pixel_values).astype("float32")
            pixel_values = self.transform(pixel_values)
        else:
            pixel_values = self._torch_transform(pixel_values)

        #     breakpoint()
        if self.labeled:
            assert (
                label in range(self.args.number_of_labels) or label is None
            ), "label is not in range: {}".format(label)

        inputs = {
            "pixel_values": pixel_values,
            "patch_mask": patch_mask,
            "num_patches": num_patches,
            "attention_mask": attention_mask,
            "label": label,
        }
        return inputs
