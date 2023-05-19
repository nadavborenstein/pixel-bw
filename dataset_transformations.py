import numpy as np
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2
from sklearn.datasets import make_blobs
from wandb.sdk.wandb_config import Config
import cv2


def overlay_weighted(src, background, alpha, beta, gamma=0):
    """overlay two images together, pixels from each image is weighted as follow

        dst[i] = alpha*src[i] + beta*background[i] + gamma

    Arguments:
        src (numpy.ndarray) : source image of shape (rows, cols)
        background (numpy.ndarray) : background image. Must be in same shape are `src`
        alpha (float) : transparent factor for the foreground
        beta (float) : transparent factor for the background
        gamma (int, optional) : luminance constant. Defaults to 0.

    Returns:
        numpy.ndarray: a copy of the source image after apply the effect
    """
    return cv2.addWeighted(src, alpha, background, beta, gamma).astype(np.uint8)


def overlay(src, background):
    """Overlay two images together via bitwise-and:

        dst[i] = src[i] & background[i]

    Arguments:
        src (numpy.ndarray) : source image of shape (rows, cols)
        background (numpy.ndarray) : background image. Must be in same shape are `src`

    Returns:
        numpy.ndarray: a copy of the source image after apply the effect
    """
    return cv2.bitwise_and(src, background).astype(np.uint8)


def translation(src, offset_x, offset_y):
    """Shift the image in x, y direction

    Arguments:
        src (numpy.ndarray) : source image of shape (rows, cols)
        offset_x (int) : pixels in the x direction.
                          Positive value shifts right and negative shifts right.
        offset_y (int) : pixels in the y direction.
                          Positive value shifts down and negative shifts up.

    Returns:
        numpy.ndarray: a copy of the source image after apply the effect
    """
    rows, cols = src.shape
    trans_matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    # size of the output image should be in the form of (width, height)
    dst = cv2.warpAffine(src, trans_matrix, (cols, rows), borderValue=255)
    return dst.astype(np.uint8)


def to_rgb(image, **kwargs):
    """
    Convert the image to RGB.
    """
    return np.stack([image, image, image], 2).astype("float32")


class SyntheticDatasetTransform(object):
    def __init__(self, args: Config, rng: np.random.RandomState):
        self.args = args
        self.rng = rng

    def add_bleed_through(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Adds bleed through effect to the image. Background is flipped horizontally
        """
        background = image.copy()
        background = cv2.flip(background, 1)  # flipped horizontally
        background = translation(
            background,
            self.rng.randint(self.args.max_bleed_offset),
            self.rng.randint(self.args.max_bleed_offset),
        )
        alpha = self.rng.normal(self.args.bleed_alpha_mean, self.args.bleed_alpha_std)
        alpha = max(0, min(1, alpha))
        beta = 1 - alpha

        return overlay_weighted(image, background, alpha, beta, self.args.bleed_gamma)

    def add_random_horizontal_line(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Add a random horizontal line to the image.
        """
        num_lines = self.rng.randint(1, 3)
        empty_lines = np.arange(image.shape[0])[np.all(image == 255, axis=1)]

        for line in range(num_lines):
            if empty_lines.shape[0] == 0:
                return image

            line_width = self.rng.randint(2, self.args.random_line_max_width)
            if self.rng.rand() < 0.5:
                line_length = self.rng.randint(
                    min(
                        self.args.random_horisontal_line_length_min, image.shape[0] - 1
                    ),
                    image.shape[0],
                )
                line_start = (image.shape[1] - line_length) // 2
            else:
                line_length = image.shape[0]
                line_start = 0

            line_loc = self.rng.choice(empty_lines)
            image[
                line_loc : line_loc + line_width, line_start : line_start + line_length
            ] = 0.0

            empty_lines = empty_lines[
                ~np.isin(empty_lines, np.arange(line_loc, line_loc + line_width))
            ]
        return image

    def add_random_vertical_line(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Add a random vertical line to the image.
        """
        line_width = self.rng.randint(2, self.args.random_line_max_width)

        if self.rng.rand() < 0.5:
            line_length = self.rng.randint(
                self.args.random_line_length_min, image.shape[1]
            )
            line_start = self.rng.randint(0, image.shape[1] - line_length)
        else:
            line_length = image.shape[1]
            line_start = 0

        empty_lines = np.arange(image.shape[1])[np.all(image == 255, axis=0)]
        empty_lines = empty_lines[
            (empty_lines < self.args.max_margins[0] - line_width)
            | (empty_lines > image.shape[1] - self.args.max_margins[1] + line_width)
        ]

        num_lines = self.rng.randint(1, 3)
        for line in range(num_lines):
            if empty_lines.shape[0] == 0:
                break
            line_loc = self.rng.choice(empty_lines)
            image[
                line_start : line_start + line_length, line_loc : line_loc + line_width
            ] = 0.0
            empty_lines = empty_lines[
                ~np.isin(empty_lines, np.arange(line_loc, line_loc + line_width))
            ]
        return image

    def add_random_salt_and_pepper_noise(
        self, image: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Add salt and pepper noise to the image.
        """
        salt_pepper_probability = self.rng.uniform(
            0, self.args.salt_and_pepper_pixel_max_probability
        )
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
        mask_size = min(
            self.rng.randint(10, self.args.blobs_mask_size_max), image.shape[0] - 1
        )
        blob_density = self.rng.randint(1, mask_size // 2)
        blob_std = self.rng.uniform(1, mask_size**0.3)
        num_samples = self.rng.randint(500, 5000)
        blobs, _ = make_blobs(
            centers=num_blobs,
            n_samples=num_samples,
            n_features=2,
            cluster_std=blob_std,
            center_box=(-blob_density, blob_density),
        )
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
        mask_size = min(
            self.rng.randint(20, self.args.blobs_mask_size_max * 2), image.shape[0] - 1
        )
        blob_density = self.rng.randint(1, mask_size // 2)
        blob_std = self.rng.uniform(1, mask_size**0.5)
        num_samples = self.rng.randint(10, 60)
        blobs, _ = make_blobs(
            centers=num_blobs,
            n_samples=num_samples,
            n_features=2,
            cluster_std=blob_std,
            center_box=(-blob_density, blob_density),
        )
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
        blob_std = self.rng.uniform(2, mask_size**0.5)
        num_samples = self.rng.randint(10000, 100000)
        blobs, _ = make_blobs(
            centers=num_blobs,
            n_samples=num_samples,
            n_features=2,
            cluster_std=blob_std,
            center_box=(-blob_density, blob_density),
        )
        blobs = np.round(blobs + (mask_size / 2))
        blobs = np.clip(blobs, 0, mask_size - 1).astype("int32")
        mask[blobs[:, 0], blobs[:, 1]] = 0

        max_size = min(300, min(image.shape[0], image.shape[1]))
        min_size = min(180, max_size - 1)
        cloud_size = self.rng.randint(min_size, max_size)
        mask = Image.fromarray((mask * 255).astype("uint8"), mode="L")
        upsampled_mask = mask.resize((cloud_size, cloud_size))
        cloud = np.array(upsampled_mask) / 255

        cloud_loc_x = self.rng.randint(0, image.shape[0] - cloud_size)
        cloud_loc_y = self.rng.randint(0, image.shape[1] - cloud_size)

        mask_strength = self.rng.uniform(0.03, 0.1)
        hole = image[
            cloud_loc_x : cloud_loc_x + cloud_size,
            cloud_loc_y : cloud_loc_y + cloud_size,
        ]
        hole[cloud < 1] = (
            hole[cloud < 1] * (1 - mask_strength) + cloud[cloud < 1] * mask_strength
        )
        image[
            cloud_loc_x : cloud_loc_x + cloud_size,
            cloud_loc_y : cloud_loc_y + cloud_size,
        ] = hole
        return image

    def add_noise_to_channels(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Add noise to the channels of the image, to make sure that it's not completely grayscale.
        """
        noise = self.rng.normal(0, self.args.channel_noise_std * 255, size=(1, 1, 3))
        image = image + noise
        image = np.clip(image, 0, 255).astype("uint8")
        return image

    def __call__(self, image: np.ndarray, mask=None) -> np.ndarray:
        transformation = A.Compose(
            [
                A.Lambda(
                    image=self.add_random_horizontal_line,
                    p=self.args.random_line_horizontal_probability,
                ),
                A.Lambda(
                    image=self.add_random_vertical_line,
                    p=self.args.random_line_vertical_probability,
                ),
                A.Lambda(
                    image=self.add_bleed_through,
                    p=self.args.bleed_through_probability,
                ),
                A.OneOf(
                    [
                        A.Lambda(image=self.add_random_salt_and_pepper_noise, p=1.0),
                        A.GaussNoise(p=0.5, var_limit=(0, 0.03)),
                    ],
                    p=self.args.noise_probability,
                ),
                A.OneOf(
                    [
                        A.Lambda(image=self.add_random_clouds),
                        A.Compose(
                            [
                                A.Lambda(image=self.add_random_clouds),
                                A.Lambda(image=self.add_random_clouds),
                            ]
                        ),
                        A.Compose(
                            [
                                A.Lambda(image=self.add_random_clouds),
                                A.Lambda(image=self.add_random_clouds),
                                A.Lambda(image=self.add_random_clouds),
                            ]
                        ),
                    ],
                    self.args.cloud_probability,
                ),
                A.OneOf(
                    [
                        A.Lambda(image=self.add_random_salt_and_pepper_noise, p=1.0),
                        A.GaussNoise(p=0.5, var_limit=(0, 0.05)),
                    ],
                    p=self.args.noise_probability,
                ),
                A.OneOf(
                    [
                        A.Lambda(image=self.add_random_blobs),
                        A.Compose(
                            [
                                A.Lambda(image=self.add_random_blobs),
                                A.Lambda(image=self.add_random_blobs),
                            ]
                        ),
                        A.Compose(
                            [
                                A.Lambda(image=self.add_random_blobs),
                                A.Lambda(image=self.add_random_blobs),
                                A.Lambda(image=self.add_random_blobs),
                            ]
                        ),
                    ],
                    self.args.blobs_probability,
                ),
                A.OneOf(
                    [
                        A.Lambda(image=self.add_random_salt_and_pepper_noise, p=1.0),
                        A.GaussNoise(p=0.5, var_limit=(0, 0.05)),
                    ],
                    p=self.args.noise_probability,
                ),
                A.OneOf(
                    [
                        A.Lambda(image=self.add_random_pepper_spots),
                        A.Compose(
                            [
                                A.Lambda(image=self.add_random_pepper_spots),
                                A.Lambda(image=self.add_random_pepper_spots),
                            ]
                        ),
                        A.Compose(
                            [
                                A.Lambda(image=self.add_random_pepper_spots),
                                A.Lambda(image=self.add_random_pepper_spots),
                                A.Lambda(image=self.add_random_pepper_spots),
                            ]
                        ),
                    ],
                    self.args.pepper_spots_probability,
                ),
                A.Blur(
                    blur_limit=self.args.blur_max_sigma, p=self.args.blur_probability
                ),
                A.Rotate(
                    limit=self.args.rotation_max_degrees,
                    p=self.args.rotation_probability,
                ),
                A.Lambda(image=to_rgb),
                A.Compose(
                    [
                        A.Lambda(image=self.add_noise_to_channels),
                        A.OneOf(
                            [
                                A.ColorJitter(),
                                A.RandomBrightnessContrast(),
                                A.RandomGamma(),
                            ]
                        ),
                    ],
                    p=self.args.color_jitter_probability,
                ),
                ToTensorV2(),
            ]
        )
        if mask is None:
            image = transformation(image=image)["image"]
            return image
        else:
            transformed = transformation(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            return image, mask


class SimpleTorchTransform(object):
    def __init__(self, args: Config, rng: np.random.RandomState):
        self.args = args
        self.rng = rng

    def __call__(self, image: np.ndarray, mask=None) -> np.ndarray:
        transformation = A.Compose(
            [
                A.Lambda(image=to_rgb),
                ToTensorV2(),
            ]
        )

        if mask is None:
            image = transformation(image=image)["image"]
            return image
        else:
            transformed = transformation(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            return image, mask
