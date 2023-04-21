from dataclasses import dataclass
from typing import List

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd



def add_frame(self, np_image: np.ndarray, border_x1, border_y1, border_x2, border_y2):
    """
    Adds a frame to the image
    """
    if self.rng.rand() < self.arguments.frame_probability and border_x1 > 0 and border_y1 > 0:
        number_of_lines = self.rng.randint(1, 9)
        number_of_lines = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 3, 7: 4, 8: 4}[number_of_lines]
        lines_to_draw = self.rng.choice(["x1", "x2", "y1", "y2"], size=number_of_lines, replace=False)

        for line in lines_to_draw:
            frame_width = self.rng.randint(self.arguments.frame_width_min, self.arguments.frame_width_max)
            if line == "x1":
                loc = self.rng.randint(0, border_x1)
                start = max(0, self.rng.randint(-(np_image.shape[1] // 2) , (np_image.shape[1] // 2) - 1))
                end = min(np_image.shape[1], np.random.randint((np_image.shape[1] // 2) + 1, np_image.shape[1] + (np_image.shape[1] // 2)))
                np_image[loc:loc + frame_width, start:end] = 0
            elif line == "x2":
                loc = self.rng.randint(border_x2, np_image.shape[0])
                start = max(0, self.rng.randint(- (np_image.shape[1] // 2), (np_image.shape[1] // 2) - 1))
                end = min(np_image.shape[1], self.rng.randint((np_image.shape[1] // 2) + 1, np_image.shape[1] + (np_image.shape[1] // 2)))
                np_image[loc:loc + frame_width, start:end] = 0
            if line == "y1":
                loc = self.rng.randint(0, border_y1)
                start = max(0, self.rng.randint(- (np_image.shape[1] // 2) , (np_image.shape[1] // 2) - 1))
                end = min(np_image.shape[0], self.rng.randint((np_image.shape[0] // 2) + 1, np_image.shape[0] + np_image.shape[0] // 2))
                np_image[start:end, loc:loc + frame_width] = 0
            elif line == "y2":
                loc = self.rng.randint(border_y2, np_image.shape[1])
                start = max(0, self.rng.randint(-(np_image.shape[0] // 2), (np_image.shape[0] // 2) - 1))
                end = min(self.rng.randint((np_image.shape[0] // 2) + 1, np_image.shape[0] + (np_image.shape[0] // 2)), np_image.shape[0])
                np_image[start:end, loc:loc + frame_width] = 0

    return np_image