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
