from jinja2 import Environment, FileSystemLoader, select_autoescape
from torch.utils.data import IterableDataset, get_worker_info
from typing import Callable, List, Tuple, Dict
from wandb.sdk.wandb_config import Config
from configs.utils import crop_image, plot_arrays
from utils.dataset_utils import CustomFont, render_html_as_image, get_random_custom_font
from dataset_transformations import SyntheticDatasetTransform, SimpleTorchTransform
from utils.squad_utils import (
    generate_pixel_mask_from_recangles,
    merge_rectangle_lines,
    convert_pixel_mask_to_patch_mask,
)
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import fuzzysearch
import pytesseract
import logging
import torch
import wandb
import copy


class SquadImageGenerator(object):
    DEFAULT_COMBINATIONS = {
        "language": "en_US",
        "custom_fonts": [],
        "font_family": "Ariel",
        "font_size": "20px",
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
        "line_height": None,
    }

    def __init__(self, config: Config, rng: np.random.RandomState) -> None:
        """
        :param config: The wandb config
        :param rng: The random number generator
        """
        self.config = config
        self.rng = rng
        self.font_list = pd.read_csv(config.font_list_path)

    def _get_updated_style_config(self, custom_font: CustomFont, margins: List):
        """
        A function to get the updated style config from wandb
        """
        style = copy.deepcopy(self.DEFAULT_COMBINATIONS)

        if custom_font is not None:
            style["custom_fonts"] = [custom_font]
            style["font_family"] = custom_font.font_name
            style["font_size"] = f"{custom_font.font_size}px"
        style["left_margin"] = f"{margins[0]}px"
        style["right_margin"] = f"{margins[1]}px"
        style["top_margin"] = f"{margins[2]}px"
        style["bottom_margin"] = f"{margins[3]}px"

        for key in style:
            if key in wandb.config:
                style[key] = [wandb.config[key]]
        return style

    def _get_random_margins(self) -> List[int]:
        """
        Get random margins
        """
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
        self, template: str, style: Dict, question: str = None, context: str = None
    ) -> str:
        """
        A function to generate an HTML text from the given template and style
        """
        env = Environment(
            loader=FileSystemLoader("./templates/"),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.get_template(template)
        if question is None and context is not None:
            html_text = template.render(context=context, **style)
        elif context is None and question is not None:
            html_text = template.render(question=question, **style)
        else:
            raise ValueError(
                "One of 'question' or 'context' should be provided to the function, and only one"
            )
        return html_text

    def _generate_question(self, text: str, font: CustomFont = None) -> np.ndarray:
        """
        Generate a question image from the given text and font
        """
        if font is None:
            font = CustomFont(file_name="Ariel", font_name="Ariel", font_size=28)
        margins = [1, 1, 1, 1]
        style = self._get_updated_style_config(font, margins)
        html_text = self._generate_html_text(
            template=self.config.html_question_template, question=text, style=style
        )
        img_array = render_html_as_image(
            html_text,
            image_resolution=self.config.image_resolution,
            channel=self.config.channel,
        )
        img_array = crop_image(img_array)
        return img_array

    def _generate_context(
        self, text: str, font: CustomFont, randomize_font: bool = True
    ) -> np.ndarray:
        """
        Generate a context image from the given text and font
        """
        margins = self._get_random_margins() if randomize_font else [1, 1, 1, 0]
        style = self._get_updated_style_config(font, margins)
        html_text = self._generate_html_text(
            template=self.config.html_context_template, context=text, style=style
        )
        img_array = render_html_as_image(
            html_text,
            image_resolution=self.config.image_resolution,
            channel=self.config.channel,
        )
        return img_array

    def generate(
        self,
        question: str,
        context: str,
        font: CustomFont = None,
        method: str = "random_crop",
        randomize_font: bool = True,
    ):
        """
        Generate an image from the given text and font
        :param text: The text to be rendered
        :param font: The font to be used
        :param method: The method to be used for generating the image, one of ["random_crop", "concatenate", "list", "first_crop"]
        """
        if font is None and randomize_font:
            font = get_random_custom_font(self.font_list, self.rng)

        question = self._generate_question(question)
        context = self._generate_context(context, font, randomize_font)
        context_crop_size = self.config.image_height - question.shape[0]

        if method == "concatenate":
            pass  # no need to do anything
        elif method == "first_crop":
            context = context[:context_crop_size, :]
        elif method == "random_crop":
            random_context_crop_loc = self.rng.randint(
                0, context.shape[0] - context_crop_size
            )
            context = context[
                random_context_crop_loc : random_context_crop_loc + context_crop_size, :
            ]
        elif method == "list":
            scans = []
            for i in range(0, context.shape[0], context_crop_size):
                scans.append(context[i : i + context_crop_size, :])
            context = scans
        else:
            raise ValueError(
                f"Invalid method {method}. Valid values are: ['random_crop', 'concatenate', 'list', 'first_crop']"
            )

        return question, context

    def _locate_answer(self, data_dict: Dict, answer: str):
        all_text_offest_map = dict()
        all_texts = ""
        for i, text in enumerate(data_dict["text"]):
            if text.strip() != "":
                for j in range(len(text) + 1):
                    all_text_offest_map[len(all_texts) + j] = i
                all_texts += text + " "

        match = fuzzysearch.find_near_matches(
            answer,
            all_texts,
            max_l_dist=int(len(answer) ** self.config.max_l_dist_factor),
        )
        return match, all_text_offest_map

    def _generate_rectangle_for_matched_answer(
        self, match, data_dict, all_text_offest_map
    ):
        start_id = all_text_offest_map[match[0].start]
        end_id = all_text_offest_map[match[0].end - 1]

        all_rectangles = []
        for i in range(start_id, end_id + 1):
            if data_dict["text"][i].strip() != "":
                all_rectangles.append(
                    (
                        data_dict["left"][i],
                        data_dict["top"][i],
                        data_dict["width"][i],
                        data_dict["height"][i],
                    )
                )
        return all_rectangles

    def generate_pixel_mask(self, img_array: np.ndarray, answer: str):
        """
        Locate the suqad answer inside the scan using tesseract
        """
        if answer == "":
            return generate_pixel_mask_from_recangles([], img_array.shape)

        data_dict = pytesseract.image_to_data(
            img_array, lang="eng", output_type=pytesseract.Output.DICT
        )
        match, all_text_offest_map = self._locate_answer(data_dict, answer)
        if len(match) == 0 or match[0].end == 0:
            return generate_pixel_mask_from_recangles([], img_array.shape)
        else:
            matched_rectangles = self._generate_rectangle_for_matched_answer(
                match, data_dict, all_text_offest_map
            )
            matched_rectangles = merge_rectangle_lines(matched_rectangles, tolerance=10)
            return generate_pixel_mask_from_recangles(
                matched_rectangles, img_array.shape
            )

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
    """
    A class to represent the squad dataset for pixel
    """

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
        self.attention_mask = self.image_generator.get_attention_mask(
            config.num_patches
        )
        self.randomize_font = split == "train"

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
        """
        A method that generates the scans from a squad sample
        """
        question = instance["question"]
        context = instance["context"]
        answer = (
            instance["answers"]["text"][0]
            if len(instance["answers"]["text"]) > 0
            else ""
        )

        question_scan, context_scan = self.image_generator.generate(
            question,
            context,
            method=self.config.long_context_generation_method,
            randomize_font=self.randomize_font,
        )

        mask = self.image_generator.generate_pixel_mask(context_scan, answer)

        return question_scan, context_scan, mask

    def __iter__(self) -> dict:
        for data in self.text_dataset:
            question_scan, context_scan, mask = self._generate_scans_fron_sample(data)

            if self.transform:
                question_scan = torch.from_numpy(np.stack([question_scan] * 3))
                context_scan, mask = self.transform(context_scan, mask)

            scan = torch.concat([question_scan, context_scan], axis=1)
            scan = scan[: self.config.image_height, :]

            mask = torch.concat([torch.zeros(question_scan.shape[1:]), mask], axis=0)
            mask = mask[: self.config.image_height, :]
            mask = convert_pixel_mask_to_patch_mask(
                mask.numpy(),
                self.config.patch_base_size[0],
                self.config.mask_patching_tolerance,
            )
            mask = torch.from_numpy(mask)

            inputs = {
                "pixel_values": scan,
                "label_mask": mask,
                "num_patches": self.config.num_patches,
                "attention_mask": self.attention_mask,
            }
            yield inputs


def main():
    wandb.init(config="configs/squad_config.yaml", mode="disabled")
    rng = np.random.RandomState(2)

    transform = SyntheticDatasetTransform(wandb.config, rng=rng)
    train_dataset = SquadDatasetForPixel(
        config=wandb.config, transform=transform, rng=rng, split="train"
    )
    figures = []
    for i in range(1):
        train_dataset.set_epoch(i)
        counter = 0
        for batch in tqdm(train_dataset, total=3000):
            if counter == 3000:
                break
            # im = batch["pixel_values"].numpy().astype("float32").transpose(1, 2, 0)
            # mask = batch["label_mask"].numpy()
            # mask = np.kron(mask, np.ones((16, 16)))
            # im[mask == 1] = im[mask == 1] - 60
            # im = np.clip(im, 0, 255).astype("uint8")
            # figures.append(im)
            counter += 1

    im = plot_arrays(figures)
    im.save("results/sample_squad.png")


if __name__ == "__main__":
    main()
