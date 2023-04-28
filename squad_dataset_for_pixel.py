from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML, CSS
from weasyprint.fonts import FontConfiguration
from cairocffi import FORMAT_ARGB32
from torch.utils.data import IterableDataset, get_worker_info
from datasets import load_dataset
from typing import Callable, List, Tuple, Dict
from wandb.sdk.wandb_config import Config
from utils.dataset_utils import CustomFont
import numpy as np
import pandas as pd
import logging
import torch
import wandb
import copy
import cv2


class SquadImageGenerator(object):
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
        "font_family_question": "Garamond",
        "font_size_question": "18px",
        "line_height_question": None,
        "font_family_context": "Garamond",
        "font_size_context": "14px",
        "line_height_context": None,
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
        dummy_fonts = [CustomFont(
            file_name=f"DUMMY_FILE_{i}", font_name=f"DUMMY_NAME_{i}", font_size=f"DUMMY_SIZE_{i}"
        ) for i in [0,1]]  # one for question and one for context
        dummy_margin = ["DUMMY_LEFT", "DUMMY_RIGHT", "DUMMY_TOP", "DUMMY_BOTTOM"]
        dummy_style = self._get_updated_style_config([dummy_fonts], dummy_margin)
        html_template = self._generate_html_text(
            self.config.html_template, dummy_style, "DUMMY_QUESTION", "DUMMY_CONTEXT"
        )
        return html_template

    def update_template(self, fonts: List[CustomFont], question: str, context: str, margins: List[int]):
        """
        A function to update the template with new font and texts
        """
        for i in range(2):
            html_template = self.template.replace(f"DUMMY_FILE_{i}", fonts[i].file_name)
            html_template = html_template.replace(f"DUMMY_NAME_{i}", fonts[i].font_name)
            html_template = html_template.replace(f"DUMMY_SIZE_{i}", str(fonts[i].font_size))
            
        html_template = html_template.replace("DUMMY_LEFT", str(margins[0]))
        html_template = html_template.replace("DUMMY_RIGHT", str(margins[1]))
        html_template = html_template.replace("DUMMY_TOP", str(margins[2]))
        html_template = html_template.replace("DUMMY_BOTTOM", str(margins[3]))
        html_template = html_template.replace("DUMMY_QUESTION", question)
        html_template = html_template.replace("DUMMY_CONTEXT", context)
        return html_template

    def _get_updated_style_config(self, custom_fonts: List[CustomFont], margins: List):
        """
        A function to get the updated style config from wandb
        """
        style = copy.deepcopy(self.DEFAULT_COMBINATIONS)
        style["custom_fonts"] = custom_fonts
        
        style["font_family_question"] = custom_fonts[0].font_name
        style["font_size_question"] = f"{custom_fonts[0].font_size}px"
        style["font_family_context"] = custom_fonts[1].font_name
        style["font_size_context"] = f"{custom_fonts[2].font_size}px"
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
        font_size = int(font_size / 1.5) + self.rng.randint(-4, 5, 1)[0]

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
        self, template: str, style: Dict, question: str, context: str
    ) -> str:
        env = Environment(
            loader=FileSystemLoader("./templates/"),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.get_template(template)
        html_text = template.render(question=question, context=context, **style)
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


class SquadDatasetForPixel(IterableDataset):
    def __init__(
        self,
        config: Config,
        split: str = "train",
        transform: Callable = None,
        rng: np.random.RandomState = None,
    ) -> None:
        super().__init__()
        self.text_dataset = load_dataset("squad_v2", split=split)
        self.config = config
        self.transform = transform
        self.rng = rng
        self.image_generator = SquadImageGenerator(config, rng)
        self.attention_mask = self.image_generator.get_attention_mask(config.num_patches)

    def set_epoch(self, epoch):
        """
        A method that sets the epoch for the dataset
        """
        info = get_worker_info()
        self.rng = np.random.RandomState(epoch + info.id if info else epoch)
        logging.info(
            f"randomizing dataset with worker id={info.id if info else 0} and epoch={epoch}"
        )

    def _generate_scans_fron_sample(self, instance: Dict):
        question = instance["question"]
        context = instance["context"]
        generated_scans = self.image_generator.generate(question, context)
        label_masks = self.image_generator.generate_label_mask(generated_scans, instance['answers']['text'][0])
        return generated_scans, label_masks

    def __iter__(self) -> dict:
        for data in self.text_dataset:
            generated_scans, label_masks = self._generate_scans_fron_sample(data)
            for scan, mask in zip(generated_scans, label_masks):
                if self.transform:
                    scan, mask = self.transform(scan, mask)
                    
                inputs = {
                    "pixel_values": scan,
                    "label_mask": mask,
                    "num_patches": self.config.num_patches,
                    "attention_mask": self.attention_mask,
                }
                yield inputs
