from dataclasses import dataclass
from typing import List

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)




class DocumentSynthesizer(object):
    """
    Generates a text image using a random font
    """
    def __init__(self, args, rng):
        self.font_dir = args.font_dir
        self.font_df = pd.read_csv(os.path.join(self.font_dir, args.font_list_path))
        self.font_list = self.font_df['path'].tolist()
        self.font_base_size = {row["path"]: row["base_size"] for index, row in self.font_df.iterrows()}
        self.arguments = args
        self.rng = rng

    def get_font_by_name(self, font_name: str, font_size: int = 10) -> ImageFont:
        """
        Returns the font with the given name
        """
        return ImageFont.truetype(os.path.join(self.font_dir, font_name), font_size)

    def get_random_font(self):
        """
        Returns a random font from the font list, with a random size
        """
        font_name = self.font_list[self.rng.randint(0, len(self.font_list))]
        font_size = self.font_base_size[font_name]
        font_size *= self.rng.uniform(self.arguments.font_size_min, self.arguments.font_size_max)
        spacing = self.rng.uniform(self.arguments.spacing_min, self.arguments.spacing_max) * font_size
        font = ImageFont.truetype(os.path.join(self.font_dir, font_name), int(font_size))
        return font, font_size, spacing

    def break_text_to_lines(self, text, line_length_in_chars, number_of_lines):
        """
        Breaks the text into lines, with a maximum length of line_length_in_chars
        """
        lines = []
        text = text.replace("\n", " ").replace("  ", " ").split(" ")
        next_word = ""

        while len(lines) < number_of_lines:
            line = ""
            while len(line) + len(next_word) < line_length_in_chars:
                line += " " + next_word
                if len(text) == 0:
                    next_word = ""
                    break
                next_word = text.pop(0)
            lines.append(line.strip())
            if next_word == "":
                break

        return lines

    def draw_font(self, font: ImageFont, text: str) -> np.ndarray:
        """
        Draws the text in the base image
        """
        image = Image.new(mode='L', size=(500 + 50 * max(2, len(text)), 500 + 2 * 50))
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), text, font=font, fill=255)
        np_image = np.array(image)
        return np_image

    def crop_generated_image(self, np_image: np.ndarray) -> np.ndarray:
        """
        Crops the generated image to remove the black borders
        """
        horzP = np.max(np_image,axis=0)
        minX=first_nonzero(horzP,0)
        maxX=last_nonzero(horzP,0)
        vertP = np.max(np_image,axis=1)
        minY=first_nonzero(vertP,0)
        maxY=last_nonzero(vertP,0)
        return np_image[minY:maxY, minX:maxX]

    def evaluate_font_size(self, font: ImageFont, text: str, spacing: int):
        """
        Evaluates the font size for the text
        :param font: the font to evaluate
        :param text: a string to evaluate the font size for
        :param spacing: how much space to put between each line
        :return: how many chars should be in each line, and how many lines should be in the text
        """
        text = text.replace("\n", " ")
        eval_length = self.arguments.font_size_eval_length
        random_start = self.rng.randint(0, len(text) - eval_length - 1)
        span = text[random_start: random_start + eval_length]

        np_image = self.draw_font(font, span)
        np_image = self.crop_generated_image(np_image)
        line_width, line_size = np_image.shape
        if line_size == 0 or line_width == 0:
            raise ValueError("Font size evaluation failed")
        random_margin = self.rng.randint(self.arguments.figure_margin_min, self.arguments.figure_margin_max)
        self.arguments.figure_margin_selected = random_margin

        text_in_figure_size = [self.arguments.figure_size[i] - random_margin for i in [0, 1]]
        line_length_in_chars = text_in_figure_size[1] / (line_size / eval_length)
        number_of_lines = text_in_figure_size[0] / (line_width + spacing)
        return line_length_in_chars, number_of_lines

    def generate_base_image(self, text: str, font: ImageFont, spacing: int, deterministic: bool = False) -> np.ndarray:
        """
        Generates a base image for the text, with a random size and random rotation
        """
        if deterministic:
            eval_text = " ".join([text] * 100)
        else:
            eval_text = text
        line_length_in_chars, number_of_lines = self.evaluate_font_size(font, eval_text, spacing)

        if self.rng.rand() >= self.arguments.overflow_probability or deterministic:
            lines = self.break_text_to_lines(text, line_length_in_chars, number_of_lines)
            lines = "\n".join(lines)
            image_size = (self.arguments.figure_size[0] - self.arguments.figure_margin_selected,
                          self.arguments.figure_size[1] - self.arguments.figure_margin_selected)
            image = Image.new(mode='L', size=image_size)
            draw = ImageDraw.Draw(image)
            draw.multiline_text((0, 0), lines, font=font, fill=255, spacing=spacing)
            np_image = np.array(image).astype("float32")
        else:
            lines = self.break_text_to_lines(text, line_length_in_chars, number_of_lines + 10)
            lines = "\n".join(lines)
            image_size = (self.arguments.figure_size[0] - self.arguments.figure_margin_selected,
                          self.arguments.figure_size[1] + self.arguments.figure_margin_selected)
            image = Image.new(mode='L', size=image_size)
            draw = ImageDraw.Draw(image)
            draw.multiline_text((0, 0), lines, font=font, fill=255, spacing=spacing)

            np_image = np.array(image).astype("float32")
            np_image = np_image[self.arguments.figure_margin_selected // 2:
                                self.arguments.figure_size[0] + self.arguments.figure_margin_selected // 2, :]
        horzP = np.max(np_image,axis=0)
        minX=first_nonzero(horzP,0)
        maxX=last_nonzero(horzP,0)
        vertP = np.max(np_image,axis=1)
        minY=first_nonzero(vertP,0)
        maxY=last_nonzero(vertP,0)

        np_image = np_image[minY:maxY, minX:maxX]
        np_image = 1 - (np_image / 255)

        np_image = self.embed_image(np_image, deterministic=deterministic)
        return np_image

    def embed_image(self, np_image: np.ndarray, add_frame=True, deterministic=False) -> np.ndarray:
        """
        Place the text image inside a frame of the correct dimensionality
        """
        large_image = np.ones(self.arguments.figure_size)
        start_x = ((large_image.shape[0] - np_image.shape[0]) // 2)
        start_y = ((large_image.shape[1] - np_image.shape[1]) // 2)
        if add_frame and not deterministic:
            large_image = self.add_frame(large_image, start_x, start_y, np_image.shape[0] + start_x, np_image.shape[1] + start_y)
        large_image[start_x:start_x + np_image.shape[0], start_y:start_y + np_image.shape[1]] = np_image
        return large_image

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
