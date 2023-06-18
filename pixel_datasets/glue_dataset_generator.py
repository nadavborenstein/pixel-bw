from jinja2 import Environment, FileSystemLoader, select_autoescape
from torch.utils.data import Dataset
from typing import Callable, List, Tuple, Dict
from wandb.sdk.wandb_config import Config
from .utils.utils import crop_image, concatenate_images, embed_image, plot_arrays
from .utils.dataset_utils import (
    CustomFont,
    render_html_as_image,
    get_random_custom_font,
    calculate_num_patches,
)
from .dataset_transformations import SyntheticDatasetTransform, SimpleTorchTransform
from datasets import load_dataset, concatenate_datasets
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
import wandb
import copy


TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


class GlueImageGenerator(object):
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
        self.template = self._preload_template()

    def _preload_template(self):
        dummy_fonts = [
            CustomFont(
                file_name=f"DUMMY_FILE_{i}",
                font_name=f"DUMMY_NAME_{i}",
                font_size=f"DUMMY_SIZE_{i}",
            )
            for i in [1, 2]
        ]

        dummy_margins = [
            [
                f"DUMMY_LEFT_{i}",
                f"DUMMY_RIGHT_{i}",
                f"DUMMY_TOP_{i}",
                f"DUMMY_BOTTOM_{i}",
            ]
            for i in [1, 2]
        ]
        dummy_style = self._get_updated_style_config(dummy_fonts, dummy_margins)
        html_template = self._generate_html_text(
            self.config.html_template,
            dummy_style,
            "DUMMY_SENTENCE_1",
            "DUMMY_SENTENCE_2",
        )
        return html_template

    def update_template(
        self, fonts: List[CustomFont], sentences: List[str], margins: List[List[int]]
    ):
        """
        A function to update the template with new font and text
        """
        html_template = self.template
        for i in [1, 2]:
            html_template = html_template.replace(
                f"DUMMY_FILE_{i}", fonts[i - 1].file_name
            )
            html_template = html_template.replace(
                f"DUMMY_NAME_{i}", fonts[i - 1].font_name
            )
            html_template = html_template.replace(
                f"DUMMY_SIZE_{i}", str(fonts[i - 1].font_size)
            )
            html_template = html_template.replace(
                f"DUMMY_LEFT_{i}", str(margins[i - 1][0])
            )
            html_template = html_template.replace(
                f"DUMMY_RIGHT_{i}", str(margins[i - 1][1])
            )
            html_template = html_template.replace(
                f"DUMMY_TOP_{i}", str(margins[i - 1][2])
            )
            html_template = html_template.replace(
                f"DUMMY_BOTTOM_{i}", str(margins[i - 1][3])
            )
            html_template = html_template.replace(
                f"DUMMY_SENTENCE_{i}", sentences[i - 1]
            )
        return html_template

    def _get_updated_style_config(
        self, custom_fonts: List[CustomFont], margins: List[List]
    ):
        """
        A function to get the updated style config from wandb
        """
        style = copy.deepcopy(self.DEFAULT_COMBINATIONS)
        style["custom_fonts"] = custom_fonts
        for i in [1, 2]:
            style[f"font_family_{i}"] = custom_fonts[i - 1].font_name
            style[f"font_size_{i}"] = f"{custom_fonts[i - 1].font_size}px"
            style[f"left_margin_{i}"] = f"{margins[i - 1][0]}px"
            style[f"right_margin_{i}"] = f"{margins[i - 1][1]}px"
            style[f"top_margin_{i}"] = f"{margins[i - 1][2]}px"
            style[f"bottom_margin_{i}"] = f"{margins[i - 1][3]}px"

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
        self, template: str, style: Dict, sentence_1: str, sentence_2: str
    ) -> str:
        env = Environment(
            loader=FileSystemLoader("pixel_datasets/templates/"),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.get_template(template)
        html_text = template.render(
            sentence_1=sentence_1, sentence_2=sentence_2, **style
        )
        return html_text

    def generate(self, sentence_1, sentence_2, fonts: List[CustomFont] = None):
        """
        Generate an image from the given text and font
        :param text: The text to be rendered
        :param font: The font to be used
        """
        if fonts is None:
            if self.config.use_same_font_for_both_sentences:
                font = get_random_custom_font(
                    self.font_list, self.rng, max_size_factor=1.0
                )
                margin = self._get_random_margins()
                fonts = [font, font]
                margins = [margin, margin]
            else:
                fonts = [
                    get_random_custom_font(
                        self.font_list, self.rng, max_size_factor=1.0
                    ),
                    get_random_custom_font(
                        self.font_list, self.rng, max_size_factor=1.0
                    ),
                ]
                margins = [self._get_random_margins(), self._get_random_margins()]
        else:
            margins = [self._get_random_margins(), self._get_random_margins()]
        html_text = self.update_template(fonts, [sentence_1, sentence_2], margins)
        img_array = render_html_as_image(
            html_text, self.config.image_resolution, channel=self.config.channel
        )
        img_array = img_array[: self.config.image_height, :]
        return img_array, fonts

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


class GlueDatasetForPixel(Dataset):
    """
    A class to represent the squad dataset for pixel
    """

    def __init__(
        self,
        task: str,
        config: Config,
        split: str = "train",
        transform: Callable = None,
        rng: np.random.RandomState = None,
    ) -> None:
        super().__init__()
        self.task = task
        if task == "mnli" and split == "validation":
            base_dataset = load_dataset(
                "glue", task, cache_dir=config.dataset_cache_dir
            )
            self.base_dataset = concatenate_datasets(
                [
                    base_dataset["validation_matched"],
                    base_dataset["validation_mismatched"],
                ]
            )
        else:
            self.base_dataset = load_dataset(
                "glue", task, split=split, cache_dir=config.dataset_cache_dir
            ).shuffle(seed=config.seed)
        self.config = config
        self.transform = transform
        self.rng = rng
        self.image_generator = GlueImageGenerator(config, rng)
        self.attention_mask = self.image_generator.get_attention_mask(
            config.num_patches
        )
        self.randomize_font = config.randomize_font
        self.base_fonts = [
            CustomFont(
                file_name="fonts/PackardAntique-Bold.ttf",
                font_name="Packardantique-Bold",
                font_size=16,
            )
        ] * 2

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _generate_scan_fron_sample(self, instance: Dict):
        """
        A method that generates the scans from a squad sample
        """
        sentence_1 = "<b>Sentence 1:</b> " + instance[TASK_TO_KEYS[self.task][0]]
        sentence_2 = (
            "<b>Sentence 2:</b> " + instance[TASK_TO_KEYS[self.task][1]]
            if TASK_TO_KEYS[self.task][1]
            else ""
        )
        scan = self.image_generator.generate(
            sentence_1,
            sentence_2,
            fonts=None if self.randomize_font else self.base_fonts,
        )
        return scan

    def __getitem__(self, index: int) -> Dict:
        sample = self.base_dataset[index]

        scan, _ = self._generate_scan_fron_sample(sample)
        num_patches = (
            calculate_num_patches(scan, config=self.config, noisy=True)
            if self.config.adaptive_num_patches
            else self.config.num_patches
        )
        if self.transform:
            scan = self.transform(scan)

        scan = scan / 255.0

        inputs = {
            "pixel_values": scan,
            "num_patches": num_patches,
            "attention_mask": self.attention_mask,
            "label": sample["label"],
        }
        return inputs


class GlueDatasetFromHub(Dataset):
    def __init__(self, config: Config, base_dataest) -> None:
        super().__init__()
        self.base_dataset = base_dataest
        self.config = config
        self.num_patches = config.num_patches

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

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index) -> Dict:
        image = self.base_dataset[index]["image"]
        image = image.copy().convert("RGB")
        image = np.array(image).astype(np.float32)

        num_patches = (
            calculate_num_patches(image, config=self.config, noisy=True)
            if self.config.adaptive_num_patches
            else self.num_patches
        )
        attention_mask = self.get_attention_mask(num_patches)

        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image)

        label = self.base_dataset[index]["label"]

        inputs = {
            "pixel_values": image,
            "attention_mask": attention_mask,
            "label": label,
            "num_patches": num_patches,
        }
        return inputs


def main():
    wandb.init(
        config="/home/knf792/PycharmProjects/pixel-2/configs/glue_config.yaml",
        mode="disabled",
    )
    rng = np.random.RandomState(2)

    transform = SyntheticDatasetTransform(wandb.config, rng=rng)
    train_dataset = GlueDatasetForPixel(
        config=wandb.config,
        task=wandb.config.task,
        transform=transform,
        rng=rng,
        split="train",
    )
    figures = []
    for i in range(3):
        counter = 0
        for batch in train_dataset:
            if counter == 3:
                break
            im = batch["pixel_values"].numpy().astype("float32").transpose(1, 2, 0)
            mask = batch["label_mask"].numpy()
            mask = np.kron(mask, np.ones((16, 16)))
            im[mask == 1] = im[mask == 1] - 60
            im = np.clip(im, 0, 255).astype("uint8")
            figures.append(im)
            counter += 1

    im = plot_arrays(figures)
    im.save("../results/sample_glue.png")


if __name__ == "__main__":
    main()
