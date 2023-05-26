from typing import Tuple, Optional

import numpy as np
from dataclasses import dataclass, asdict, field

from transformers import TrainingArguments

from pixel import PoolingMode


@dataclass
class RenderingArguments:
    """
    Arguments for loading the dataset.
    """

    font_dir: str = field(
        default="/home/it4i-nadavb/pixel/dataset_synthesis/synthetic_text_gen/all_fonts/",
        metadata={"help": "Path to the fonts directory."},
    )
    font_list_path: str = field(
        default="clear_fonts_with_size.csv", metadata={"help": "Path to the fonts list."}
    )
    font_size_max: int = field(
        default=1.3,
        metadata={"help": "Maximum font ratio of the renderer (1 ~= 25 pixels)"},
    )
    font_size_min: int = field(
        default=0.7,
        metadata={"help": "minimum font ratio of the renderer (1 ~= 25 pixels)"},
    )
    spacing_max: float = field(
        default=0.3,
        metadata={"help": "maximum spacing ratio between lines (1 ~= 25 pixels)"},
    )
    spacing_min: float = field(
        default=0.05,
        metadata={"help": "minimum spacing ratio between lines (1 ~= 25 pixels)"},
    )
    font_size_eval_length: int = field(
        default=100,
        metadata={
            "help": "how many characters to evaluate the font size on (no reason to change)"
        },
    )
    figure_size: Tuple[int] = field(
        default=(368, 368), metadata={"help": "size of the input figure to the model"}
    )
    figure_margin_max: int = field(
        default=40,
        metadata={"help": "maximal margin in pixels at the border of the figure"},
    )
    figure_margin_min: int = field(
        default=10,
        metadata={"help": "minimal margin in pixels at the border of the figure"},
    )
    frame_probability: float = field(
        default=0.5, metadata={"help": "probability of the figure having a frame"}
    )
    frame_width_max: int = field(
        default=10, metadata={"help": "maximum width of the frame in pixels"}
    )
    frame_width_min: int = field(
        default=4, metadata={"help": "minimum width of the frame in pixels"}
    )
    overflow_probability: float = field(
        default=0.5,
        metadata={
            "help": "probability of the text to overflow a bit from the sides of the figure"
        },
    )
    rotation_max_degrees: float = field(
        default=5.0, metadata={"help": "Maximum rotation angle in degrees"}
    )
    rotation_probability: float = field(
        default=0.2, metadata={"help": "probability of applying rotation"}
    )
    blur_probability: float = field(
        default=0.2, metadata={"help": "probability of applying blur"}
    )
    blur_max_sigma: int = field(
        default=5, metadata={"help": "Maximum sigma of the gaussian blur"}
    )
    salt_and_pepper_pixel_max_probability: float = field(
        default=0.01, metadata={"help": "intensity of salt and pepper noise"}
    )
    salt_and_pepper_probability: float = field(
        default=0.3, metadata={"help": "probability of salt and pepper noise"}
    )
    noise_probability: float = field(
        default=0.3, metadata={"help": "probability of applying pixel level noise"}
    )
    text_length_max: int = field(
        default=2000, metadata={"help": "maximum length of text to render in chars"}
    )
    text_length_min: int = field(
        default=500, metadata={"help": "minimum length of text to render in chars"}
    )
    random_line_horizontal_probability: float = field(
        default=0.1, metadata={"help": "probability of adding random horizontal line"}
    )
    random_line_vertical_probability: float = field(
        default=0.02, metadata={"help": "probability of adding random vertical line"}
    )
    random_line_length_min: int = field(
        default=10, metadata={"help": "min length of random lines"}
    )
    random_line_max_width: int = field(
        default=10, metadata={"help": "min width of random lines"}
    )
    blobs_probability: float = field(
        default=0.1,
        metadata={"help": "probability of adding large black holes to the figure"},
    )
    pepper_spots_probability: float = field(
        default=0.3,
        metadata={"help": "probability of adding small black holes to the figure"},
    )
    blobs_mask_size_max: int = field(
        default=100, metadata={"help": "max size of the water damage marks"}
    )
    blobs_num_max: int = field(
        default=6,
        metadata={"help": "the greater the number, the less spherical the blob is"},
    )
    color_jitter_probability: float = field(
        default=0.2, metadata={"help": "probability of applying color jitter"}
    )
    channel_noise_std: float = field(
        default=0.02,
        metadata={
            "help": 'for converting b%w image to color image, determines the "colorfulness" of it'
        },
    )
    cloud_probability: float = field(
        default=0.3,
        metadata={"help": "probability of adding random water damage to the figure"},
    )

    warmup_font = "fonts/GoNotoCurrent.ttf"
    mask_block_size: Tuple[int] = field(
        default=(32, 16),
        metadata={
            "help": "size in pixels of each maks patch. Must be a multiplication of the patch size of the encoder"
        },
    )
    patch_base_size: Tuple[int] = field(
        default=(16, 16), metadata={"help": "size in pixels of each patch"}
    )
    mask_block_probability: float = field(
        default=0.25, metadata={"help": "target ratio of pixels to mask from the input"}
    )
    mask_max_merged_blocks_size: Tuple[int] = field(
        default=(3, 8),
        metadata={"help": "maximum number of consecutive blocks that can be merged"},
    )
    mask_min_merged_blocks_size: Tuple[int] = field(
        default=(2, 3),
        metadata={"help": "minimum number of consecutive blocks that can be merged"},
    )
    embed_real_image: bool = field(default=False, metadata={"help": "Whether to embed real images in white background"})

    def __post_init__(self):
        """
        saves the initial values of the arguments.
        """
        self.initial_values = dict()
        for key in self.__dataclass_fields__:
            self.initial_values[key] = getattr(self, key)

    def update_randomness_intensity(
        self, max_step, step, update_type="linear", warmup_steps=0, gamma=None
    ):
        """
        Update the randomness intensity.
        :param update_type: one of ['linear', 'exponential', 'cosine']
        """
        if step < warmup_steps:
            factor = 0.0
        else:
            if update_type == "linear":
                factor = (step - warmup_steps) / (max_step - warmup_steps)
            elif update_type == "exponential":
                if gamma is None:
                    gamma = float(f"0.{'9' * (len(str(max_step)) - 1)}2")
                factor = 1 - (gamma ** (step - warmup_steps))
            elif update_type == "cosine":
                factor = 1 - (
                    (
                        1
                        + np.cos(
                            np.pi * (step - warmup_steps) / (max_step - warmup_steps)
                        )
                    )
                    / 2
                )
            elif update_type == "none":
                factor = 1
            else:
                raise ValueError("Unknown update_type: {}".format(update_type))

        factor = min(1, factor)
        self.rotation_max_degrees = self.initial_values["rotation_max_degrees"] * factor
        self.rotation_probability = self.initial_values["rotation_probability"] * factor
        self.blur_probability = self.initial_values["blur_probability"] * factor
        self.salt_and_pepper_pixel_max_probability = (
            self.initial_values["salt_and_pepper_pixel_max_probability"] * factor
        )
        self.salt_and_pepper_probability = (
            self.initial_values["salt_and_pepper_probability"] * factor
        )
        self.warp_probability = self.initial_values["font_size_max"] * factor
        self.random_line_horizontal_probability = (
            self.initial_values["random_line_horizontal_probability"] * factor
        )
        self.random_line_vertical_probability = (
            self.initial_values["random_line_vertical_probability"] * factor
        )
        self.random_line_length_min = max(
            2, self.initial_values["random_line_length_min"] * factor
        )
        self.random_line_max_width = max(
            2, self.initial_values["random_line_max_width"] * factor
        )
        self.blobs_probability = self.initial_values["blobs_probability"] * factor
        self.pepper_spots_probability = (
            self.initial_values["pepper_spots_probability"] * factor
        )
        self.blobs_mask_size_max = max(
            20, self.initial_values["blobs_mask_size_max"] * factor
        )
        self.blobs_num_max = max(2, self.initial_values["blobs_num_max"] * factor)
        self.noise_probability = self.initial_values["noise_probability"] * factor
        self.blur_max_sigma = max(1, self.initial_values["blur_max_sigma"] * factor)
        self.color_jitter_probability = (
            self.initial_values["color_jitter_probability"] * factor
        )
        self.channel_noise_std = self.initial_values["channel_noise_std"] * factor
        self.cloud_probability = self.initial_values["cloud_probability"] * factor

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_dataset_names: str = field(
        metadata={"help": "Name of train dataset in HuggingFace dataset hub"}
    )
    train_splits: str = field(metadata={"help": "Name of the training dataset split."})
    validation_dataset_name: str = field(
        metadata={"help": "Name of validation dataset in HuggingFace dataset hub"}
    )
    validation_split: str = field(
        metadata={"help": "Name of the validation dataset split."}
    )
    dataset_caches: Optional[str] = field(
        default=None, metadata={"help": "Directory where the dataset is cached"}
    )
    train_dataset_configs: str = field(
        default="20220301.simple", metadata={"help": "Train dataset config/subset"}
    )
    
    ## Settings of real scans dataset
    real_train_dataset_names: str = field(
        default="Nadav/MiniScans", metadata={"help": "Name of train dataset in HuggingFace dataset hub"}
    )
    real_train_splits: str = field(default="train", metadata={"help": "Name of the training dataset split."})
    real_dataset_caches: Optional[str] = field(
        default="/scratch/project/dd-22-70/cache/data/miniscans", metadata={"help": "Directory where the dataset is cached"}
    )
    
    
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=100,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    streaming: Optional[bool] = field(
        default=False, metadata={"help": "Whether to stream the training dataset"}
    )
    do_normalize: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to normalize to model's feature extractor's mean and std."
        },
    )

    def __post_init__(self):
        self.train_dataset_names = self.train_dataset_names.split(",")
        
        if self.real_train_dataset_names:
            self.real_train_dataset_names = self.real_train_dataset_names.split(",")
        
        self.train_splits = self.train_splits.split(",")
        
        if self.real_train_splits:
            self.real_train_splits = self.real_train_splits.split(",")
        
        if self.train_dataset_configs:
            self.train_dataset_configs = self.train_dataset_configs.split(",")
        else:
            self.train_dataset_configs = [None] * len(self.train_dataset_names)
            
        if self.dataset_caches:
            self.dataset_caches = self.dataset_caches.split(",")
        else:
            self.dataset_caches = [None] * len(self.train_dataset_names)
            
        if self.real_dataset_caches:
            self.real_dataset_caches = self.real_dataset_caches.split(",")
            
        assert (
            len(self.train_dataset_names)
            == len(self.train_splits)
            == len(self.train_dataset_configs)
            == len(self.dataset_caches)
        )

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/feature extractor we are going to pre-train.
    """

    text_renderer_name_or_path: str = field(
        metadata={
            "help": "Path / Huggingface identifier of the text renderer that was used to prerender the "
            "training/validation data."
        }
    )
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name_or_path"
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    feature_extractor_name: str = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    use_auth_token: str = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    mask_ratio: float = field(
        default=0.25,
        metadata={
            "help": "The ratio of the number of masked tokens in the input sequence."
        },
    )
    norm_pix_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to train with normalized pixel values as target."
        },
    )
    span_masking: bool = field(
        default=False,
        metadata={"help": "Whether to use span masking instead of random masking."},
    )
    masking_max_span_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum span length that can be masked when using span masking."
        },
    )
    masking_spacing: Optional[int] = field(
        default=None,
        metadata={
            "help": "Spacing between masked spans. Defaults to the length of the span."
            "Use this argument to set it to a fixed number of patches."
            "Recommended setting: For masking ratio <= 0.4 leave the default"
            "For ratios between 0.4 and 0.7 set it to 1. For higher, set it to 0"
        },
    )
    masking_cumulative_span_weights: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated list of cumulative probabilities of sampling a span of length n"
            "when using span masking. Must be a list of size model_args.masking_max_span_length."
        },
    )
    dropout_prob: float = field(
        default=0.1, metadata={"help": "Dropout probability for attention blocks"}
    )
    patch_size: Tuple[int] = field(
        default=(16, 16), metadata={"help": "size in pixels of each patch"}
    )
    max_seq_length: int = field(
        default=529, metadata={"help": "Maximum sequence length for the model"}
    )

    def __post_init__(self):
        if self.masking_cumulative_span_weights is not None:
            self.masking_cumulative_span_weights = [
                float(w) for w in self.masking_cumulative_span_weights.split(",")
            ]

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__


@dataclass
class CustomTrainingArguments(TrainingArguments):
    base_learning_rate: float = field(
        default=1.5e-4,
        metadata={
            "help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."
        },
    )
    warmup_render_steps: int = field(
        default=100000, metadata={"help": "Number of steps for the warmup."}
    )
    randomness_intensity_update_interval: int = field(
        default=100, metadata={"help": "Update interval for the randomness intensity."}
    )
    overfit: bool = field(
        default=False, metadata={"help": "Whether to overfit the model to 10 examples."}
    )
    run_name: str = field(default="default", metadata={"help": "Name of the run."})


@dataclass
class VisualizationArguments:
    input_str: str = """
    Dear Isabelle and the NLP teaching team,
    Thank you for your reply. I have gone through the announcement and syllabus previously, and it is immensely helpful that materials have already been uploaded.
    I do wish to reiterate a request: if any of the teaching team has their own lecture notes, it would be very helpful if I could use that. Since I rely heavily on facial cues to make sense of what is being said, it is incredibly difficult for me to take notes and listen to what you are saying at the same time. Lecture notes allow me equal access to the information a student gains when they listen to the lecture. It would not be circulated, and only used to supplement what I miss hearing during the lecture."
    """
    vis_model_path: str = "experiments/outputs/checkpoint-3250000"
    auth_token: str = "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"
    label_column_name: str = "label"
    number_of_labels: int = 1
    max_seq_length_in_patches: int = 529
    figure_size: Tuple[int] = (368, 368)
    patch_base_size: Tuple[int] = (16, 16)
    mask_block_size: Tuple[int] = (32, 16)
    mask_block_probability: float = 0.25
    mask_max_merged_blocks_size: Tuple[int] = (3, 8)
    mask_min_merged_blocks_size: Tuple[int] = (2, 3)
    seed: int = 42
    run_name: str = "visualization"
    embed_real_image: bool = False


@dataclass
class FineTuningDatasetArguments:
    dataset_name: str = field(default="Nadav/runaway_scans", metadata={"help": "Name of the dataset to fine-tune on"})
    task_name: str = field(default="mnli", metadata={"help": "Name of the task to fine-tune on"})
    dataset_cache_dir: str = field(default=None, metadata={"help": "Path to the dataset cache"})
    number_of_labels: int = field(default=2, metadata={"help": "Number of labels in the dataset"})
    label_column_name: str = field(default="Country", metadata={"help": "Name of the label column in the dataset"})
    embed_real_image: bool = field(default=False, metadata={"help": "Whether to embed real images in white background"})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    finetuning_font: str = field(default="fonts/CaslonAntique.ttf", metadata={"help": "Font to use for fine-tuning"})
    mask_block_size: Tuple[int] = field(
        default=(32, 16),
        metadata={
            "help": "size in pixels of each maks patch. Must be a multiplication of the patch size of the encoder"
        },
    )
    patch_base_size: Tuple[int] = field(
        default=(16, 16), metadata={"help": "size in pixels of each patch"}
    )
    mask_block_probability: float = field(
        default=0.25, metadata={"help": "target ratio of pixels to mask from the input"}
    )
    mask_max_merged_blocks_size: Tuple[int] = field(
        default=(2, 6),
        metadata={"help": "maximum number of consecutive blocks that can be merged"},
    )
    mask_min_merged_blocks_size: Tuple[int] = field(
        default=(1, 2),
        metadata={"help": "minimum number of consecutive blocks that can be merged"},
    )
    figure_size: Tuple[int] = field(
        default=(368, 368), metadata={"help": "size of the input figure to the model"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__


@dataclass
class FineTuningModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    processor_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained processor name or path if not the same as model_name"}
    )
    rendering_backend: Optional[str] = field(
        default="pangocairo", metadata={
            "help": "Rendering backend to use. Options are 'pygame' or 'pangocairo'. For most applications it is "
                    "recommended to use the default 'pangocairo'."}
    )
    fallback_fonts_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing fallback font files used by the text renderer for PIXEL. "
                          "PyGame does not support fallback fonts so this argument is ignored when using the "
                          "PyGame backend."},
    )
    render_rgb: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to render images in RGB. RGB rendering can be useful when working with emoji "
                    "but it makes rendering a bit slower, so it is recommended to turn on RGB rendering only "
                    "when there is need for it. PyGame does not support fallback fonts so this argument is ignored "
                    "when using the PyGame backend."
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: str = field(
        default=None,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    pooling_mode: str = field(
        default="mean",
        metadata={
            "help": f"Pooling mode to use in classification head (options are {[e.value for e in PoolingMode]}."
        },
    )
    pooler_add_layer_norm: bool = field(
        default=True,
        metadata={
            "help": "Whether to add layer normalization to the classification head pooler. Note that this flag is"
                    "ignored and no layer norm is added when using CLS pooling mode."
        },
    )
    dropout_prob: float = field(
        default=0.1, metadata={"help": "Dropout probability for attention blocks and classification head"}
    )

    def __str__(self):
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__