import numpy as np
from dataclasses import dataclass


@dataclass
class DatasetArguments:
    """
    Arguments for loading the dataset.
    """
    rotation_max_degrees: float = 5.0
    rotation_probability: float = 0.2
    blur_probability: float = 0.2
    salt_and_pepper_pixel_max_probability: float = 0.01
    salt_and_pepper_probability: float = 0.3
    salt_and_pepper_max_size: int = 2
    font_size_max: int = 50
    font_size_min: int = 20
    warp_probability: float = 0.05
    warp_max_intensity: int = 10
    spacing_max: float = 1.0
    spacing_min: float = 0.05
    text_length_max: int = 1000
    text_length_min: int = 500
    random_line_horizontal_probability: float = 0.1
    random_line_vertical_probability: float = 0.02
    random_line_length_min: int = 10
    random_line_max_width: int = 10
    blobs_probability: float = 0.1
    pepper_spots_probability: float = 0.3
    blobs_mask_size_max: int = 100
    blobs_num_max: int = 6
    noise_probability: float = 0.8
    blur_max_sigma: int = 5
    color_jitter_probability: float = 0.2
    channel_noise_std: float = 0.02
    cloud_probability: float = 0.3

    mask_block_size = (32, 16)
    mask_block_probability = 0.25
    mask_max_merged_blocks_size = (2, 6)
    mask_min_merged_blocks_size = (1, 2)
    mask_min_blocks = 3

    warmup_font = "fonts/GoNotoCurrent.ttf"


    def __post_init__(self):
        """
        saves the initial values of the arguments.
        """
        self.initial_values = dict()
        for key in self.__dataclass_fields__:
            self.initial_values[key] = getattr(self, key)

    def update_randomness_intensity(self, max_step, step, update_type='linear', warmup_steps=0, gamma=None):
        """
        Update the randomness intensity.
        :param update_type: one of ['linear', 'exponential', 'cosine']
        """
        if step < warmup_steps:
            factor = 0.0
        else:
            if update_type == 'linear':
                factor = (step - warmup_steps) / (max_step - warmup_steps)
            elif update_type == 'exponential':
                if gamma is None:
                    gamma = float(f"0.{'9' * (len(str(max_step)) - 1)}2")
                factor = 1 - (gamma ** (step - warmup_steps))
            elif update_type == 'cosine':
                factor = 1 - ((1 + np.cos(np.pi * (step - warmup_steps) / (max_step - warmup_steps))) / 2)
            else:
                raise ValueError('Unknown update_type: {}'.format(update_type))

        self.rotation_max_degrees = self.initial_values["rotation_max_degrees"] * factor
        self.rotation_probability = self.initial_values["rotation_probability"] * factor
        self.blur_probability = self.initial_values["blur_probability"] * factor
        self.salt_and_pepper_pixel_max_probability = self.initial_values["salt_and_pepper_pixel_max_probability"] * factor
        self.salt_and_pepper_probability = self.initial_values["salt_and_pepper_probability"] * factor
        self.salt_and_pepper_max_size = self.initial_values["salt_and_pepper_max_size"] * factor
        self.warp_probability = self.initial_values["font_size_max"] * factor
        self.warp_max_intensity = self.initial_values["warp_max_intensity"] * factor
        self.random_line_horizontal_probability = self.initial_values["random_line_horizontal_probability"] * factor
        self.random_line_vertical_probability = self.initial_values["random_line_vertical_probability"] * factor
        self.random_line_length_min = max(2, self.initial_values["random_line_length_min"] * factor)
        self.random_line_max_width = max(2, self.initial_values["random_line_max_width"] * factor)
        self.blobs_probability = self.initial_values["blobs_probability"] * factor
        self.pepper_spots_probability = self.initial_values["pepper_spots_probability"] * factor
        self.blobs_mask_size_max = max(2, self.initial_values["blobs_mask_size_max"] * factor)
        self.blobs_num_max = max(2, self.initial_values["blobs_num_max"] * factor)
        self.noise_probability = self.initial_values["noise_probability"] * factor
        self.blur_max_sigma = max(1, self.initial_values["blur_max_sigma"] * factor)
        self.color_jitter_probability = self.initial_values["color_jitter_probability"] * factor
        self.channel_noise_std = self.initial_values["channel_noise_std"] * factor
        self.cloud_probability = self.initial_values["cloud_probability"] * factor