from weasyprint import HTML, CSS
from weasyprint.fonts import FontConfiguration
from dataset_transformations import SyntheticDatasetTransform
from cairocffi import FORMAT_ARGB32
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML
from PIL import Image, ImageDraw as D
from datasets import load_dataset, Dataset
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Dict, Callable, Optional
from wandb.sdk.wandb_config import Config
from dataclasses import dataclass

from utils.utils import crop_image, concatenate_images, embed_image, plot_arrays
from utils.dataset_utils import CustomFont
from torch.utils.data import IterableDataset, get_worker_info

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytesseract
import fuzzysearch
import logging
import wandb
import torch
import copy
import cv2

logging.basicConfig(level=logging.INFO)


class ImageGenerator(object):
    DEFAULT_COMBINATIONS = {
        "language": "en_US",
        "font_family": "Garamond",
        "font_size": "18px",
        "text_align": "justify",
        "hyphenate": False,
        "width": "368px",
        "height": "368px",
        "top_margin": "0px",
        "right_margin": "0px",
        "bottom_margin": "0px",
        "left_margin": "0px",
        "line_spacing": ".5",
        "word_break": None,
        "line_break": None,
        "line_height_pretraining": None,
    }

    def __init__(self, config: Config, rng: np.random.RandomState) -> None:
        """
        :param config: The wandb config
        :param rng: The random number generator
        """
        self.config = config
        self.rng = rng
        self.font_list = pd.read_csv(config.font_list_path)
        self.template = self._preload_template()

    def _preload_template(self):
        dummy_fonts = CustomFont(
            file_name="DUMMY_FILE", font_name="DUMMY_NAME", font_size="DUMMY_SIZE"
        )
        dummy_margin = ["DUMMY_LEFT", "DUMMY_RIGHT", "DUMMY_TOP", "DUMMY_BOTTOM"]
        dummy_style = self._get_updated_style_config([dummy_fonts], dummy_margin)
        html_template = self._generate_html_text(
            self.config.html_template, dummy_style, "DUMMY_TEXT"
        )
        return html_template

    def update_template(self, font: CustomFont, text: str, margins: List[int]):
        """
        A function to update the template with new font and text
        """
        html_template = self.template.replace("DUMMY_FILE", font.file_name)
        html_template = html_template.replace("DUMMY_NAME", font.font_name)
        html_template = html_template.replace("DUMMY_SIZE", str(font.font_size))
        html_template = html_template.replace("DUMMY_LEFT", str(margins[0]))
        html_template = html_template.replace("DUMMY_RIGHT", str(margins[1]))
        html_template = html_template.replace("DUMMY_TOP", str(margins[2]))
        html_template = html_template.replace("DUMMY_BOTTOM", str(margins[3]))
        html_template = html_template.replace("DUMMY_TEXT", text)
        return html_template

    def _get_updated_style_config(self, custom_fonts: List[CustomFont], margins: List):
        """
        A function to get the updated style config from wandb
        """
        style = copy.deepcopy(self.DEFAULT_COMBINATIONS)
        style["custom_fonts"] = custom_fonts
        style["font_family_pretraining"] = custom_fonts[0].font_name
        style["font_size_pretraining"] = f"{custom_fonts[0].font_size}px"
        style["left_margin"] = f"{margins[0]}px"
        style["right_margin"] = f"{margins[1]}px"
        style["top_margin"] = f"{margins[2]}px"
        style["bottom_margin"] = f"{margins[3]}px"

        for key in style:
            if key in wandb.config:
                style[key] = [wandb.config[key]]
        return style

    def _get_random_custom_font(self) -> CustomFont:
        """
        A method that returns a random custom font from the font list
        """
        random_index = self.rng.randint(0, self.font_list.shape[0])
        random_font = self.font_list["path"][random_index]
        random_font = random_font.replace(" ", "_")  # fixing spaces in the path
        font_name = random_font.split(".")[0].split("/")[1]

        font_size = self.font_list["base_size"][random_index]
        font_size = int(font_size / 1.4) + self.rng.randint(-4, 5, 1)[0]

        custom_font = CustomFont(
            file_name=random_font, font_name=font_name.title(), font_size=font_size
        )

        return custom_font

    def _get_random_margins(self) -> List[int]:
        if self.rng.rand() < self.config.margins_probability:
            margins = self.rng.randint(
                np.zeros(4, dtype=int),
                list(map(lambda x: max(1, x), self.config.max_margins)),
                4,
            )
        else:
            margins = np.zeros(4, dtype=int)
        return margins

    def _generate_html_text(
        self, template: str, style: Dict, pretraining_text: str
    ) -> str:
        env = Environment(
            loader=FileSystemLoader("./templates/"),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.get_template(template)
        html_text = template.render(pretraining_text=pretraining_text, **style)
        return html_text

    def _render_html_as_image(self, html_text: str, channel: str = "GRAYSCALE"):
        """
        A function to render an HTML text as an image
        """
        font_config = FontConfiguration()  # TODO define once outside the function
        html = HTML(string=html_text, base_url=".")
        doc = html.render(font_config=font_config)
        surface, width, height = doc.write_image_surface(
            resolution=self.config.image_resolution
        )
        img_format = surface.get_format()

        # This is BGRA channel in little endian (reverse)
        if img_format != FORMAT_ARGB32:
            raise RuntimeError(
                f"Expect surface format to be 'cairocffi.FORMAT_ARGB32', but got {img_format}."
                + "Please check the underlining implementation of 'weasyprint.document.Document.write_image_surface()'"
            )

        img_buffer = surface.get_data()
        # Returns image array in "BGRA" channel
        img_array = np.ndarray(
            shape=(height, width, 4), dtype=np.uint8, buffer=img_buffer
        )
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

    def _remove_leading_whitespace(self, img_array: np.ndarray) -> np.ndarray:
        non_white_pixels = np.argwhere(img_array < 255)
        min_indices = non_white_pixels.min(axis=0)
        img_array = img_array[min_indices[0] - 2 :, :]
        img_array = np.concatenate(
            [
                img_array,
                np.full((min_indices[0] - 2, *img_array.shape[1:]), 255, dtype="uint8"),
            ],
            axis=0,
        )
        return img_array

    def generate(self, text, font: CustomFont = None):
        """
        Generate an image from the given text and font
        :param text: The text to be rendered
        :param font: The font to be used
        """
        if font is None:
            font = self._get_random_custom_font()
        margins = self._get_random_margins()
        html_text = self.update_template(font, text, margins)
        img_array = self._render_html_as_image(html_text, channel=self.config.channel)
        img_array = self._remove_leading_whitespace(img_array)
        img_array = img_array[: self.config.image_height, :]
        return img_array, font

    def _crop_image(self, img):
        """
        Crops an image to the smallest possible size containing all non-white pixels.
        Parameters:
            img (np.ndarray): A numpy array representing the image to crop.
        Returns:
            np.ndarray: The cropped image.
        """
        # Find the indices of all non-white pixels.
        non_white_pixels = np.argwhere(img < 255)

        # Find the minimum and maximum indices in each dimension.
        max_indices = non_white_pixels.max(axis=0)

        # Crop the image to the smallest possible size containing all non-white pixels.
        cropped_img = img[: max_indices[0] + 2, :]
        return cropped_img

    def check_if_can_concatenate(self, img):
        non_white_pixels = np.argwhere(img < 255)
        # Find the minimum and maximum indices in each dimension.
        max_indices = non_white_pixels.max(axis=0)
        return (
            self.config.image_height - max_indices[0]
            > self.config.maximal_white_space_in_image
        )

    def concatenate_images(self, image1, image2):
        """
        A method that concatenates two images vertically
        """
        concatenated = np.concatenate((self._crop_image(image1), image2), axis=0)
        concatenated = concatenated[: self.config.image_height, :]
        return concatenated

    def get_attention_mask(self, num_text_patches: int):
        """
        Creates an attention mask of size [1, seq_length]
        The mask is 1 where there is text or a [SEP] black patch and 0 everywhere else
        """
        n = min(
            num_text_patches + 1, self.config.max_seq_length
        )  # Add 1 for [SEP] token (black patch)
        zeros = torch.zeros(self.config.max_seq_length)
        ones = torch.ones(n)
        zeros[:n] = ones
        return zeros

    def generate_random_mask(self, image_size):
        """
        Generate a random mask for the image.
        """
        mask = np.zeros(image_size)
        pixels_masked = 0
        while (
            pixels_masked / (image_size[0] * image_size[1])
        ) < self.config.mask_block_probability:
            patch_height = (
                self.rng.randint(
                    self.config.mask_min_merged_blocks_size[0],
                    self.config.mask_max_merged_blocks_size[0] + 1,
                )
                * self.config.mask_block_size[0]
            )
            patch_width = (
                self.rng.randint(
                    self.config.mask_min_merged_blocks_size[1],
                    self.config.mask_max_merged_blocks_size[1] + 1,
                )
                * self.config.mask_block_size[1]
            )

            for i in range(10):
                random_mask_location_x = self.rng.choice(
                    np.arange(
                        0, image_size[0] - patch_height, self.config.mask_block_size[0]
                    )
                )
                random_mask_location_y = self.rng.choice(
                    np.arange(
                        0, image_size[1] - patch_width, self.config.mask_block_size[1]
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
            :: self.config.patch_base_size[0], :: self.config.patch_base_size[1]
        ].flatten()
        return mask, small_mask


class PretrainingDataset(IterableDataset):
    def __init__(
        self,
        config: Config,
        text_dataset: Dataset,
        transform: Optional[Callable],
        rng: np.random.RandomState,
    ) -> None:
        """
        :param config: Config object, wandb.config
        :param text_dataset: textual dataset, made out of (long) strings
        :param transform: transform function for the generated images
        :param rng: random number generator
        """
        super().__init__()
        self.config = config
        if transform:
            self.transform = transform
        else:
            tensor_transform = ToTensorV2()
            self.transform = lambda x: tensor_transform(image=x)["image"]
        self.text_dataset = text_dataset
        self.rng = rng
        self.image_generator = ImageGenerator(config, rng)
        self.attention_mask = self.image_generator.get_attention_mask(
            self.config.num_patches
        )

    def set_epoch(self, epoch):
        """
        A method that sets the epoch for the dataset
        """
        info = get_worker_info()
        self.rng = np.random.RandomState(epoch + info.id if info else epoch)
        logging.info(
            f"randomizing dataset with worker id={info.id if info else 0} and epoch={epoch}"
        )

    def _clean_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """
        Removes empty paragraphs and paragraphs that are too short
        :param paragraphs: list of paragraphs
        :return: list of cleaned paragraphs
        """
        paragraphs = [p.strip() for p in paragraphs]
        paragraphs = [
            p for p in paragraphs if len(p) > self.config.min_paragraph_length
        ]  # TODO add to config file
        return paragraphs

    def _get_random_snippet(self, random_loc_within_paragraph: bool = True) -> str:
        """
        A method that returns a random snippet from a random paragraph in a random document
        :param random_loc_within_paragraph: whether to return a random location within the paragraph or the beginning
        """
        paragraphs = []
        while len(paragraphs) == 0:
            doc_index = self.rng.randint(0, len(self.text_dataset))
            doc = self.text_dataset[doc_index]["text"]
            paragraphs = doc.split("\n")
            paragraphs = self._clean_paragraphs(paragraphs)

        assert len(paragraphs) > 0, f"no paragraphs"

        paragraph_index = self.rng.randint(0, len(paragraphs))
        paragraph = paragraphs[paragraph_index]

        random_loc = (
            self.rng.randint(0, len(paragraph) - self.config.min_paragraph_length)
            if random_loc_within_paragraph
            else 0
        )
        return paragraph[
            random_loc : self.config.max_snippet_length
        ]  # TODO add to config file

    def __iter__(self):
        while True:
            snippet = self._get_random_snippet()
            image, font = self.image_generator.generate(snippet)
            while self.image_generator.check_if_can_concatenate(image):
                if self.rng.rand() > self.config.random_font_probability:
                    font = None
                snippet_2 = self._get_random_snippet(random_loc_within_paragraph=False)
                image_2, font = self.image_generator.generate(snippet_2, font)
                image = self.image_generator.concatenate_images(image, image_2)

            if self.transform:
                image = self.transform(image)

            mask, patch_mask = self.image_generator.generate_random_mask(
                image.shape[1:]
            )
            inputs = {
                "pixel_values": image,
                "patch_mask": torch.tensor(patch_mask, dtype=torch.float32),
                "num_patches": self.config.num_patches,
                "attention_mask": self.attention_mask,
            }

            yield inputs


def main():
    wandb.init(config="configs/config.yaml", mode="disabled")
    rng = np.random.RandomState(2)
    text_dataset = load_dataset("wikipedia", "20220301.simple")

    transform = SyntheticDatasetTransform(wandb.config, rng=rng)
    train_dataset = PretrainingDataset(
        wandb.config, text_dataset["train"], transform, rng=rng
    )
    figures = []
    for i in range(3):
        train_dataset.set_epoch(i)
        counter = 0
        for batch in train_dataset:
            if counter == 3:
                break
            im = batch["pixel_values"].numpy().astype("uint8").transpose(1, 2, 0)
            figures.append(im)
            counter += 1

    im = plot_arrays(figures)
    im.save("results/sample.png")


if __name__ == "__main__":
    main()
    # print(timeit.timeit(main, number=10))
