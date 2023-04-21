from weasyprint import HTML, CSS
from weasyprint.fonts import FontConfiguration
import matplotlib.pyplot as plt
import cv2
import numpy as np
from cairocffi import FORMAT_ARGB32
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image, ImageDraw as D

from datasets import load_dataset, Dataset
import fuzzysearch

from typing import List, Tuple, Dict, Callable, Optional
import numpy as np
import copy
import wandb
from wandb.sdk.wandb_config import Config
import pandas as pd
from pprint import pprint
from utils.utils import crop_image, concatenate_images, embed_image
from torch.utils.data import IterableDataset, get_worker_info
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)


@dataclass
class CustomFont:
    """
    A class to represent a custom font
    """
    file_name: str
    font_name: str
    font_size: int
    
    def __str__(self) -> str:
        return f"Name: {self.font_name}\nSize: {self.file_name}\nPath: {self.font_size}"


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

    def _get_updated_style_config(self, custom_fonts: List[CustomFont]):
        """
        A function to get the updated style config from wandb
        """
        style = copy.deepcopy(self.DEFAULT_COMBINATIONS)
        style["custom_fonts"] = custom_fonts
        style["font_family_pretraining"] = custom_fonts[0].font_name
        style["font_size_pretraining"] = f"{custom_fonts[0].font_size}px"
        
        for key in style:
            if key in wandb.config:
                style[key] = [wandb.config[key]]
        return style
    
    def _get_random_custom_font(self) -> CustomFont:
        """
        A method that returns a random custom font from the font list
        """
        random_index = self.rng.integers(0, self.font_list.shape[0])
        random_font = self.font_list["path"][random_index]
        font_name = random_font.split(".")[0].split("/")[1]

        font_size = self.font_list["base_size"][random_index]
        font_size = int(font_size / 2) + self.rng.integers(-4, 5, 1)[0]
        
        custom_font = CustomFont(file_name=random_font,
                                 font_name=font_name.title(),
                                 font_size=font_size)
        
        return custom_font

    def _generate_html_text(self, template: str, style: Dict, pretraining_text: str) -> str:
        env = Environment(loader=FileSystemLoader("./templates/"), 
                    autoescape=select_autoescape(["html", "xml"]))
        template = env.get_template(template)
        html_text = template.render(pretraining_text=pretraining_text, **style)
        return html_text
    
    def _render_html_as_image(self, html_text: str, channel: str = 'GRAYSCALE'):
        """
        A function to render an HTML text as an image
        """
        font_config = FontConfiguration()  # TODO define once outside the function
        html=HTML(string=html_text, base_url=".")
        doc = html.render(font_config=font_config)
        surface, width, height = doc.write_image_surface(resolution=self.config.image_resolution)
        img_format = surface.get_format()

        # This is BGRA channel in little endian (reverse)
        if img_format != FORMAT_ARGB32:
            raise RuntimeError(
                f"Expect surface format to be 'cairocffi.FORMAT_ARGB32', but got {img_format}." +
                "Please check the underlining implementation of 'weasyprint.document.Document.write_image_surface()'"
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
        style = self._get_updated_style_config(custom_fonts=[font])
        template = self.config.html_template
        html_text = self._generate_html_text(template, style, text)
        img_array = self._render_html_as_image(html_text, channel=self.config.channel)
        img_array = img_array[:self.config.image_heigh, :]
        return img_array
        
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
        cropped_img = img[:max_indices[0]+2, :]
        return cropped_img
        
    def check_if_can_concatenate(self, img):
        non_white_pixels = np.argwhere(img < 255)
        # Find the minimum and maximum indices in each dimension.
        max_indices = non_white_pixels.max(axis=0)
        return self.config.image_heigh - max_indices[0] > self.config.maximal_white_space_in_image
    
    def concatenate_images(self, image1, image2):
        """
        A method that concatenates two images vertically
        """
        concatenated = np.concatenate((self._crop_image(image1), self._crop_image(image2)), axis=0)
        concatenated = concatenated[:self.config.image_heigh, :]
        return concatenated
    
    def generate_random_mask(self, image_size):
        """
        Generate a random mask for the image.
        """
        mask = np.zeros(image_size)
        pixels_masked = 0
        while (pixels_masked / (image_size[0] * image_size[1])) < self.args.mask_block_probability:
            patch_height = self.rng.randint(self.args.mask_min_merged_blocks_size[0],
                                             self.args.mask_max_merged_blocks_size[0] + 1) * self.args.mask_block_size[0]
            patch_width = self.rng.randint(self.args.mask_min_merged_blocks_size[1],
                                            self.args.mask_max_merged_blocks_size[1] + 1) * self.args.mask_block_size[1]

            for i in range(10):
                random_mask_location_x = self.rng.choice(np.arange(0, image_size[0] - patch_height, self.args.mask_block_size[0]))
                random_mask_location_y = self.rng.choice(np.arange(0, image_size[1] - patch_width, self.args.mask_block_size[1]))

                slice = mask[random_mask_location_x: random_mask_location_x + patch_height,
                        random_mask_location_y:  random_mask_location_y + patch_width]
                if np.sum(slice) > 0:
                    continue
                else:
                    mask[random_mask_location_x: random_mask_location_x + patch_height,
                    random_mask_location_y:  random_mask_location_y + patch_width] = 1

                    pixels_masked += patch_height * patch_width
                    break

        small_mask = mask[::self.args.patch_base_size[0], ::self.args.patch_base_size[1]].flatten()
        return mask, small_mask


class PretrainingDataset(IterableDataset):
    
    def __init__(self, config: Config,
                 text_dataset: Dataset,
                 transform: Optional[Callable],
                 rng: np.random.RandomState) -> None:
        """
        :param config: Config object, wandb.config
        :param text_dataset: textual dataset, made out of (long) strings
        :param transform: transform function for the generated images
        :param rng: random number generator
        """
        super().__init__()
        self.config = config
        self.transform = transform
        self.text_dataset = text_dataset
        self.rng = rng
        self.image_generator = ImageGenerator(config, rng)
        
    
    def set_epoch(self, epoch):
        info = get_worker_info()
        self.rng = np.random.RandomState(epoch + info.id if info else epoch)
        logging.info(f"randomizing dataset with worker id={info.id if info else 0} and epoch={epoch}")
    
    def _clean_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """
        Removes empty paragraphs and paragraphs that are too short
        :param paragraphs: list of paragraphs
        :return: list of cleaned paragraphs
        """
        paragraphs = [p.trim() for p in paragraphs ]
        paragraphs = [p for p in paragraphs if len(p) > self.config.min_paragraph_length]  # TODO add to config file
        return paragraphs
    
    def _get_random_snippet(self, random_loc_within_paragraph: bool = True) -> str:
        """
        A method that returns a random snippet from a random paragraph in a random document
        :param random_loc_within_paragraph: whether to return a random location within the paragraph or the beginning
        """
        doc_index = self.rng.randint(0, len(self.text_dataset))
        doc = self.text_dataset[doc_index]["text"]
        paragraphs = doc.split("\n")
        paragraphs = self._clean_paragraphs(paragraphs)
        
        paragraph_index = self.rng.randint(0, len(paragraphs))
        paragraph = paragraphs[paragraph_index]
        
        random_loc = self.rng.randint(0, len(paragraph)) if random_loc_within_paragraph else 0
        return paragraph[random_loc:self.config.max_snippet_length]  # TODO add to config file
    
    def __iter__(self):
        while True:
            snippet = self._get_random_snippet()
            image = self.image_generator.generate(snippet)
            if self.image_generator.check_if_can_concatenate(image):  # TODO maybe add a probability to concatenate
                snippet_2 = self._get_random_snippet(random_loc_within_paragraph=False)
                image_2 = self.image_generator.generate(snippet_2)
                image = self.image_generator.concatenate_images(image, image_2)
                
            if self.transform:
                image = self.transform(image)
            
            mask, patch_mask = self.generate_random_mask(image.shape[1:])
            attention_mask = get_attention_mask(self.num_patches)
            inputs = {"pixel_values": image,
                      "patch_mask": torch.tensor(patch_mask, dtype=torch.float32),
                      "num_patches": self.num_patches,
                      "attention_mask": attention_mask}

            yield inputs
        