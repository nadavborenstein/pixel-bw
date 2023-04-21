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
from torch.utils.data import IterableDataset


class ImageGenerator(object):
    
    def __init__(self) -> None:
        pass
    
    def generate(self, text):
        pass
    
    def check_if_can_concatenate(self, image):
        pass
    
    def concatenate_images(self, image1, image2):
        pass
    
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
        info = torch.utils.data.get_worker_info()
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
            should_concatenate = self.image_generator.check_if_can_concatenate(image)
            if should_concatenate:  # TODO maybe add a probability to concatenate
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
        