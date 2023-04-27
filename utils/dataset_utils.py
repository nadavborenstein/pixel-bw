from dataclasses import dataclass
from typing import List

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd



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

