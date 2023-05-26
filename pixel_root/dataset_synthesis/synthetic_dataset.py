import logging

import datasets
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.datasets import make_blobs
from dataset_synthesis.document_syntesis import DocumentSynthesizer
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2
from pixel.utils.misc import get_attention_mask
import pickle
import os

class SyntheticDatasetTransform(object):

    def __init__(self, args, rng):
        self.args = args
        self.rng = rng

    def add_random_horizontal_line(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Add a random horizontal line to the image.
        """
        line_length = self.rng.randint(self.args.random_line_length_min, image.shape[0])
        line_width  = self.rng.randint(1, self.args.random_line_max_width)
        line_start = self.rng.randint(0, image.shape[0] - line_length)

        image_center = image[50:image.shape[0] - 50, 50:image.shape[1] - 50]
        empty_lines = np.arange(image_center.shape[0])[np.all(image_center == 1, axis=1)]

        if empty_lines.shape[0] == 0:
            return image

        line_loc = self.rng.choice(empty_lines) + 50
        image[line_loc:line_loc + line_width, line_start:line_start + line_length] = 0
        return image

    def add_random_vertical_line(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Add a random vertical line to the image.
        """
        line_length = self.rng.randint(self.args.random_line_length_min, image.shape[1])
        line_width  = self.rng.randint(1, self.args.random_line_max_width)
        line_start = self.rng.randint(0, image.shape[1] - line_length)

        image_center = image[50:image.shape[0] - 50, 50:image.shape[1] - 50]
        empty_lines = np.arange(image_center.shape[1])[np.all(image_center == 1, axis=0)]

        if empty_lines.shape[0] == 0:
            return image

        line_loc = self.rng.choice(empty_lines) + 50
        image[line_start:line_start + line_length, line_loc:line_loc + line_width] = 0
        return image

    def add_random_salt_and_pepper_noise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Add salt and pepper noise to the image.
        """
        salt_pepper_probability = self.rng.uniform(0, self.args.salt_and_pepper_pixel_max_probability)
        salt = self.rng.rand(image.shape[0], image.shape[1]) < salt_pepper_probability
        pepper = self.rng.rand(image.shape[0], image.shape[1]) < salt_pepper_probability

        image[pepper] = 0
        image[salt] = 1
        return image

    def add_random_blobs(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Add random holes in the image.
        """
        num_blobs = self.rng.randint(1, self.args.blobs_num_max)
        mask_size = self.rng.randint(10, self.args.blobs_mask_size_max)
        blob_density = self.rng.randint(1, mask_size // 2)
        blob_std = self.rng.uniform(1, mask_size ** 0.3)
        num_samples = self.rng.randint(500, 5000)
        blobs, _ = make_blobs(centers=num_blobs, n_samples=num_samples, n_features=2,
                              cluster_std=blob_std,
                              center_box=(-blob_density, blob_density))
        blobs = np.round(blobs + (mask_size / 2))
        blobs = np.clip(blobs, 0, mask_size - 1).astype("int32")

        mask_loc_x = self.rng.randint(0, image.shape[0] - mask_size)
        mask_loc_y = self.rng.randint(0, image.shape[1] - mask_size)
        blobs[:, 0] += mask_loc_x
        blobs[:, 1] += mask_loc_y
        image[blobs[:, 0], blobs[:, 1]] = 0
        return image

    def add_random_pepper_spots(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Add random holes in the image.
        """
        num_blobs = self.rng.randint(1, self.args.blobs_num_max)
        mask_size = self.rng.randint(20, self.args.blobs_mask_size_max * 2)
        blob_density = self.rng.randint(1, mask_size // 2)
        blob_std = self.rng.uniform(1, mask_size ** 0.5)
        num_samples = self.rng.randint(10, 60)
        blobs, _ = make_blobs(centers=num_blobs, n_samples=num_samples, n_features=2,
                              cluster_std=blob_std,
                              center_box=(-blob_density, blob_density))
        blobs = np.round(blobs + (mask_size / 2))
        blobs = np.clip(blobs, 0, mask_size - 1).astype("int32")

        mask_loc_x = self.rng.randint(0, image.shape[0] - mask_size)
        mask_loc_y = self.rng.randint(0, image.shape[1] - mask_size)
        blobs[:, 0] += mask_loc_x
        blobs[:, 1] += mask_loc_y
        image[blobs[:, 0], blobs[:, 1]] = 0
        return image

    def add_random_clouds(self, image: np.ndarray, **kwargs) -> np.ndarray:
        num_blobs = self.rng.randint(1, self.args.blobs_num_max)
        mask_size = 100
        mask = np.ones(shape=(mask_size, mask_size))
        blob_density = self.rng.randint(10, mask_size // 4)
        blob_std = self.rng.uniform(2, mask_size ** 0.5)
        num_samples = self.rng.randint(10000, 100000)
        blobs, _ = make_blobs(centers=num_blobs, n_samples=num_samples, n_features=2,
                              cluster_std=blob_std,
                              center_box=(-blob_density, blob_density))
        blobs = np.round(blobs + (mask_size / 2))
        blobs = np.clip(blobs, 0, mask_size - 1).astype("int32")
        mask[blobs[:, 0], blobs[:, 1]] = 0

        cloud_size = self.rng.randint(180, 300)
        mask = Image.fromarray((mask * 255).astype("uint8"), mode="L")
        upsampled_mask = mask.resize((cloud_size, cloud_size))
        cloud = np.array(upsampled_mask) / 255

        cloud_loc_x = self.rng.randint(0, image.shape[0] - cloud_size)
        cloud_loc_y = self.rng.randint(0, image.shape[1] - cloud_size)

        mask_strength = self.rng.uniform(0.03, 0.1)
        hole = image[cloud_loc_x:cloud_loc_x + cloud_size, cloud_loc_y:cloud_loc_y + cloud_size]
        hole[cloud < 1] = hole[cloud < 1] * (1 - mask_strength) + cloud[cloud < 1] * mask_strength
        image[cloud_loc_x:cloud_loc_x + cloud_size, cloud_loc_y:cloud_loc_y + cloud_size] = hole
        return image

    def to_rgb(self, image, **kwargs):
        """
        Convert the image to RGB.
        """
        return np.stack([image, image, image], 2).astype("float32")


    def add_noise_to_channels(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Add noise to the channels of the image, to make sure that it's not completely grayscale.
        """
        noise = self.rng.normal(0, self.args.channel_noise_std, size=(1, 1, 3))
        image = image + noise
        image = np.clip(image, 0, 1).astype("float32")
        return image

    def __call__(self, image: np.ndarray) -> np.ndarray:
        transformation = A.Compose([
            A.Lambda(image=self.add_random_horizontal_line, p=self.args.random_line_horizontal_probability),
            A.Lambda(image=self.add_random_vertical_line, p=self.args.random_line_vertical_probability),
            A.OneOf([
                A.Lambda(image=self.add_random_salt_and_pepper_noise, p=1.0),
                A.GaussNoise(p=0.5, var_limit=(0, 0.03))
            ], p=self.args.noise_probability),
            A.OneOf([
                A.Lambda(image=self.add_random_clouds),
                A.Compose([
                    A.Lambda(image=self.add_random_clouds),
                    A.Lambda(image=self.add_random_clouds),
                ]),
                A.Compose([
                    A.Lambda(image=self.add_random_clouds),
                    A.Lambda(image=self.add_random_clouds),
                    A.Lambda(image=self.add_random_clouds),
                ]),
            ], self.args.cloud_probability),
            A.OneOf([
                A.Lambda(image=self.add_random_salt_and_pepper_noise, p=1.0),
                A.GaussNoise(p=0.5, var_limit=(0, 0.05))
            ], p=self.args.noise_probability),
            A.OneOf([
                A.Lambda(image=self.add_random_blobs),
                A.Compose([
                    A.Lambda(image=self.add_random_blobs),
                    A.Lambda(image=self.add_random_blobs),
                ]),
                A.Compose([
                    A.Lambda(image=self.add_random_blobs),
                    A.Lambda(image=self.add_random_blobs),
                    A.Lambda(image=self.add_random_blobs),
                ]),
            ], self.args.blobs_probability),
            A.OneOf([
                A.Lambda(image=self.add_random_salt_and_pepper_noise, p=1.0),
                A.GaussNoise(p=0.5, var_limit=(0, 0.05))
                ], p=self.args.noise_probability),
            A.OneOf([
                    A.Lambda(image=self.add_random_pepper_spots),
                A.Compose([
                    A.Lambda(image=self.add_random_pepper_spots),
                    A.Lambda(image=self.add_random_pepper_spots),
                ]),
                A.Compose([
                    A.Lambda(image=self.add_random_pepper_spots),
                    A.Lambda(image=self.add_random_pepper_spots),
                    A.Lambda(image=self.add_random_pepper_spots),
                    ]),
            ], self.args.pepper_spots_probability),
            A.Blur(blur_limit=self.args.blur_max_sigma, p=self.args.blur_probability),
            A.Rotate(limit=self.args.rotation_max_degrees, p=self.args.rotation_max_degrees),
            A.Lambda(image=self.to_rgb),
            A.Compose([
                A.Lambda(image=self.add_noise_to_channels),
                A.OneOf([
                    A.ColorJitter(),
                    A.RandomBrightnessContrast(),
                    A.RandomGamma(),
                ])
                ]
                , p=self.args.color_jitter_probability),
            ToTensorV2(),
            ])

        image = transformation(image=image)["image"]
        return image


class SyntheticDatasetTorch(Dataset):
    """
    Synthetic Dataset for torchvision.
    """

    def __init__(self, text_dataset: datasets.Dataset,
                 transform=None,
                 args=None,
                 document_synthesizer=DocumentSynthesizer,
                 overfit=False,
                 rng: np.random.RandomState = None):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.text_dataset = text_dataset.shuffle(rng.randint(0, 2 ** 32 - 1))
        self.transform = transform
        self.args = args
        self.ds = document_synthesizer
        self.overfit_examples = [] if overfit else None
        self.eval_dataset = []

        self.warmup = False
        self.num_patches = 529 # TODO add this as an argument
        self.rng = rng

    def set_epoch(self, epoch):
        info = torch.utils.data.get_worker_info()
        self.rng = np.random.RandomState(epoch + info.id if info else epoch)
        logging.info(f"randomizing dataset with worker id={info.id if info else 0} and epoch={epoch}")

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

    def render_string(self, text):
        """
        Render a string to an image.
        """
        font, font_size, spacing = self.ds.get_random_font()
        image = self.ds.generate_base_image(text, font, spacing)
        if self.transform:
            image = self.transform(image)
        mask, patch_mask = self.generate_random_mask(image.shape[1:])
        attention_mask = get_attention_mask(self.num_patches)
        return {"pixel_values": image,
               "patch_mask": torch.tensor(patch_mask, dtype=torch.float32),
               "num_patches": self.num_patches,
               "attention_mask": attention_mask}

    def get_evaluation_set(self):
        """
        Generate a set of examples for evaluation.
        """
        if self.overfit_examples is not None:
            for i in range(10):
                example = self[i]
                self.eval_dataset.append(example)
        else:
            print(os.getcwd())
            self.eval_dataset = pickle.load(open("/home/it4i-nadavb/pixel/pixel/test_data/develop_set.p", "rb"))
        return self.eval_dataset

    def __len__(self):
        return len(self.text_dataset)

    def __getitem__(self, item):
        """
        Iterate over the dataset.
        """
        if self.overfit_examples is not None and len(self.overfit_examples) >= 10:  # then we will return one of 10 examples
            return self.overfit_examples[item % 10]

        tries = 0
        while True:  # we try several times, in case something fails
            text = self.text_dataset[item + tries]["text"]
            if len(text) < self.args.text_length_min:
                tries += 1
                continue

            # get a text span from the dataset
            text_span_length = self.rng.randint(self.args.text_length_min, self.args.text_length_max)
            text_span_start = self.rng.randint(0, len(text) - text_span_length) if len(text) > text_span_length else 0
            text_span = text[text_span_start:text_span_start + text_span_length]

            # if we do warmup then we use an easier font
            if self.warmup:
                font, font_size, spacing = self.ds.get_font_by_name(self.args.warmup_font, 20), 20, 1.0
            else:
                font, font_size, spacing = self.ds.get_random_font()

            try:
                image = self.ds.generate_base_image(text_span, font, spacing)
            except (ValueError, OSError):  # if it fails we try again with a different text
                tries += 1
                continue

            if self.transform:
                image = self.transform(image)

            mask, patch_mask = self.generate_random_mask(image.shape[1:])
            attention_mask = get_attention_mask(self.num_patches)
            
            inputs = {"pixel_values": image,
                      "patch_mask": torch.tensor(patch_mask, dtype=torch.float32),
                      "num_patches": self.num_patches,
                      "attention_mask": attention_mask}

            if self.overfit_examples is not None:
                self.overfit_examples.append(inputs)

            return inputs

            # self.args.update_randomness_intensity(self.max_step, self.step, update_type='linear', warmup_steps=self.warmup_steps)
            # self.transform = SyntheticDatasetTransform(self.args, rng=self.rng)


