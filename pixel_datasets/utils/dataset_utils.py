from dataclasses import dataclass
from typing import List

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from weasyprint import HTML, CSS
from weasyprint.fonts import FontConfiguration
from cairocffi import FORMAT_ARGB32
import cv2


@dataclass
class CustomFont:
    """
    A class to represent a custom font
    """

    file_name: str
    font_name: str
    font_size: int

    def __str__(self) -> str:
        return f"Name: {self.font_name}\nSize: {self.font_size}\nPath: {self.file_name}"

    def __getitem__(self, key: str) -> str:
        if key == "file_name":
            return self.file_name
        elif key == "font_name":
            return self.font_name
        elif key == "font_size":
            return self.font_size
        else:
            raise KeyError(f"Invalid key {key}")


def render_html_as_image(
    html_text: str, image_resolution: int = 96, channel: str = "GRAYSCALE"
):
    """
    A function to render an HTML text as an image
    """
    font_config = FontConfiguration()  # TODO define once outside the function
    html = HTML(string=html_text, base_url=".")
    doc = html.render(font_config=font_config)
    surface, width, height = doc.write_image_surface(resolution=image_resolution)
    img_format = surface.get_format()

    # This is BGRA channel in little endian (reverse)
    if img_format != FORMAT_ARGB32:
        raise RuntimeError(
            f"Expect surface format to be 'cairocffi.FORMAT_ARGB32', but got {img_format}."
            + "Please check the underlining implementation of 'weasyprint.document.Document.write_image_surface()'"
        )

    img_buffer = surface.get_data()
    # Returns image array in "BGRA" channel
    img_array = np.ndarray(shape=(height, width, 4), dtype=np.uint8, buffer=img_buffer)
    if channel == "GRAYSCALE":
        return cv2.cvtColor(img_array, cv2.COLOR_BGRA2GRAY)
    elif channel == "RGBA":
        return cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGBA)
    elif channel == "RGB":
        return cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
    elif channel == "BGRA":
        return np.copy(img_array)
    elif channel == "BGR":
        return cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
    else:
        valid_channels = ["GRAYSCALE", "RGB", "RGBA", "BGR", "BGRA"]
        raise ValueError(
            f"Invalid channel code {channel}. Valid values are: {valid_channels}."
        )


def get_random_custom_font(font_list, rng) -> CustomFont:
    """
    A method that returns a random custom font from the font list
    """
    random_index = rng.randint(0, font_list.shape[0])
    random_font = font_list["path"][random_index]
    random_font = random_font.replace(" ", "_")  # fixing spaces in the path
    font_name = random_font.split(".")[0].split("/")[1]

    font_size = font_list["base_size"][random_index]
    font_size = int(font_size * rng.uniform(0.65, 1.15))
    custom_font = CustomFont(
        file_name=random_font, font_name=font_name.title(), font_size=font_size
    )
    return custom_font


def generate_patch_mask(config, rng, image_size):
    """
    Generate a random mask for the image.
    """
    mask_shape = image_size / np.array(config.patch_base_size)
    assert mask_shape[0] == int(mask_shape[0]), "Mask shape is not an integer"
    assert mask_shape[1] == int(mask_shape[1]), "Mask shape is not an integer"

    mask_shape = mask_shape.astype(int)
    mask = np.zeros(mask_shape)
    patches_masked = 0
    while (
        patches_masked / (mask_shape[0] * mask_shape[1])
    ) < config.mask_block_probability:
        patch_height = rng.randint(
            config.mask_min_merged_blocks_size[0],
            config.mask_max_merged_blocks_size[0] + 1,
        )

        patch_width = rng.randint(
            config.mask_min_merged_blocks_size[1],
            config.mask_max_merged_blocks_size[1] + 1,
        )

        for _ in range(10):
            random_mask_location_x = rng.randint(mask_shape[0] - patch_height + 1)
            random_mask_location_y = rng.randint(mask_shape[1] - patch_width + 1)

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

                patches_masked += patch_height * patch_width
                break

    pixel_mask = np.kron(mask, np.ones(config.patch_base_size))
    return pixel_mask, mask
