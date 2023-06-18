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
from wandb.sdk.wandb_config import Config
import platform
import pytesseract


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


def get_random_custom_font(
    font_list, rng, min_size_factor=0.65, max_size_factor=1.15
) -> CustomFont:
    """
    A method that returns a random custom font from the font list
    """
    random_index = rng.randint(0, font_list.shape[0])
    random_font = font_list["path"][random_index]
    random_font = random_font.replace(" ", "_")  # fixing spaces in the path
    font_name = random_font.split(".")[0].split("/")[1]

    font_size = font_list["base_size"][random_index]
    font_size = int(font_size * rng.uniform(min_size_factor, max_size_factor))
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


def find_text_contures(np_image: np.ndarray, sensitivity=32):
    if len(np_image.shape) == 2:
        np_image = np.stack([np_image, np_image, np_image], axis=2)
    if np_image.max() <= 1:
        np_image = np_image * 255
    np_image = np_image.astype(np.uint8)
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    kernel = np.ones((5, 5), np.uint8)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    connected = connected / 255
    summed = np.sum(connected, axis=1)
    try:
        max_black_line = np.max(np.where(summed > sensitivity)[0])
    except ValueError:
        return np_image.shape[0] -1
    return max_black_line


def calculate_num_patches(np_image: np.ndarray, config: Config, noisy=False) -> int:
    """
    A function to calculate the number of patches in an image
    """
    assert type(np_image) == np.ndarray, "Image must be a numpy array"
    if len(np_image.shape) == 4:
        np_image = np_image[0, 0]
    if noisy:
        max_black_line = find_text_contures(np_image)
    else:
        if len(np_image.shape) == 3:
            np_image = np_image[0]
        max_value = np_image.max()
        max_value = np_image.max()
        black_lines = np.any(np_image != max_value, axis=1)
        max_black_line = np.max(np.where(black_lines))
    max_black_line = max_black_line // config.patch_base_size[0]
    num_patches = (max_black_line + 1) * 23
    return num_patches


def simple_ocr(image: np.ndarray) -> str:
    """
    A function to perform OCR on an image
    """
    assert type(image) == np.ndarray, "Image must be a numpy array"
    assert (
        platform.python_version() == "3.7.16"
    ), "Don't forget to run `conda activate Genalog`!"

    ocred_image = pytesseract.image_to_data(
        image, lang="eng", output_type=pytesseract.Output.DICT
    )
    ocred_image = " ".join(ocred_image["text"])
    return ocred_image
