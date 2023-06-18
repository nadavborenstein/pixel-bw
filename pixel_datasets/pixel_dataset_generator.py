from .dataset_transformations import SyntheticDatasetTransform
from cairocffi import FORMAT_ARGB32
from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image, ImageDraw as D
from datasets import load_dataset, Dataset, interleave_datasets
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Dict, Callable, Optional
from wandb.sdk.wandb_config import Config

from .utils.utils import crop_image, concatenate_images, embed_image, plot_arrays
from .utils.dataset_utils import (
    CustomFont,
    render_html_as_image,
    get_random_custom_font,
    generate_patch_mask,
)
from torch.utils.data import IterableDataset, get_worker_info

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
            loader=FileSystemLoader("pixel_datasets/templates/"),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.get_template(template)
        html_text = template.render(pretraining_text=pretraining_text, **style)
        return html_text

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

    def generate(self, text, font: CustomFont = None, margins: List = None):
        """
        Generate an image from the given text and font
        :param text: The text to be rendered
        :param font: The font to be used
        """
        if font is None:
            font = get_random_custom_font(self.font_list, self.rng)
        if margins is None:
            margins = self._get_random_margins()
        html_text = self.update_template(font, text, margins)
        img_array = render_html_as_image(
            html_text, self.config.image_resolution, channel=self.config.channel
        )
        img_array = self._remove_leading_whitespace(img_array)
        img_array = img_array[: self.config.image_height, :]
        return img_array, font

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
        concatenated = np.concatenate((crop_image(image1), image2), axis=0)
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


class PretrainingDataset(IterableDataset):
    def __init__(
        self,
        config: Config,
        text_dataset: Dataset,
        transform: Optional[Callable],
        rng: np.random.RandomState,
        max_steps: int = None,
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
        self.max_steps = max_steps
        self.steps_taken = 0

    def set_epoch(self, epoch):
        """
        A method that sets the epoch for the dataset
        """
        info = get_worker_info()
        self.rng = np.random.RandomState(epoch + info.id if info else epoch)
        logging.info(
            f"randomizing dataset with worker id={info.id if info else 0} and epoch={epoch}"
        )
        self.steps_taken = 0

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

    def _get_random_snippet(
        self, random_loc_within_paragraph: bool = True
    ) -> Tuple[str, List[str]]:
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
        return (
            paragraph[random_loc : self.config.max_snippet_length],
            paragraphs[paragraph_index + 1 :],
        )

    def _should_stop(self):
        if self.max_steps is None:
            return False
        else:
            return self.steps_taken >= self.max_steps

    def _update_steps(self):
        self.steps_taken += 1

    def generate_inference_image(
        self,
        text: str,
        font: CustomFont = None,
        split_text: bool = True,
        clean_text: bool = True,
    ):
        """
        A method that generates an image from a random snippet
        """
        margins = [1, 1, 1, 1]
        if split_text:
            if clean_text:
                paragraphs = self._clean_paragraphs(text.split("\n"))
            else:
                paragraphs = text.split("\n")
        else:
            paragraphs = [text]
        image, font = self.image_generator.generate(paragraphs.pop(0), font, margins)
        while (
            self.image_generator.check_if_can_concatenate(image) and len(paragraphs) > 0
        ):
            snippet = paragraphs.pop(0)
            image_2, font = self.image_generator.generate(snippet, font, margins)
            image = self.image_generator.concatenate_images(image, image_2)

        assert image.shape[0] == self.config.image_height
        assert image.shape[1] == self.config.image_width
        return image

    def generate_image(self):
        """
        A method that generates an image from a random snippet
        """
        snippet, next_snippets = self._get_random_snippet()
        image, font = self.image_generator.generate(snippet)
        while self.image_generator.check_if_can_concatenate(image):
            if self.rng.rand() > self.config.random_font_probability:
                font = None
            if (
                self.rng.rand() > self.config.random_snippet_probability
                or len(next_snippets) == 0
            ):
                snippet_2, next_snippets = self._get_random_snippet(
                    random_loc_within_paragraph=False
                )
            else:
                snippet_2 = next_snippets.pop(0)
            image_2, font = self.image_generator.generate(snippet_2, font)
            image = self.image_generator.concatenate_images(image, image_2)

        assert image.shape[0] == self.config.image_height
        assert image.shape[1] == self.config.image_width
        return image

    def __iter__(self):
        while not self._should_stop():
            try:
                image = self.generate_image()
            except Exception as e:
                logging.info(f"Failed to generate image. Skipping image")
                continue

            if self.transform:
                image = self.transform(image)
            image = image / 255.0
            mask, patch_mask = generate_patch_mask(
                self.config, self.rng, image.shape[1:]
            )
            patch_mask = patch_mask.flatten()

            inputs = {
                "pixel_values": image,
                "patch_mask": torch.tensor(patch_mask, dtype=torch.float32),
                "num_patches": self.config.num_patches,
                "attention_mask": self.attention_mask,
            }
            self._update_steps()
            yield inputs

