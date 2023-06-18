from weasyprint.fonts import FontConfiguration
import matplotlib.pyplot as plt
import cv2
import numpy as np
from cairocffi import FORMAT_ARGB32
from jinja2 import Environment, FileSystemLoader, select_autoescape
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw as D

from datasets import load_dataset

from typing import List, Tuple, Dict, Union, Optional
import numpy as np
import io
import pandas as pd
import torch
import math
import pytesseract
from .squad_utils import (
    generate_pixel_mask_from_recangles,
    merge_rectangle_lines,
    convert_pixel_mask_to_patch_mask,
)
from scipy.ndimage import binary_erosion, binary_dilation


def merge_rectangles(rect1: Tuple[float], rect2: Tuple[float], tolerance=5):
    """
    A function to merge two rectangles if they are adjacent
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    if abs(x1 + w1 - x2) <= tolerance or abs(x2 + w2 - x1) <= tolerance:
        # The rectangles are adjacent horizontally
        new_x = min(x1, x2)
        new_y = min(y1, y2)
        new_w = max(x1 + w1, x2 + w2) - new_x
        new_h = max(y1 + h1, y2 + h2) - new_y
        return (new_x, new_y, new_w, new_h)

    elif abs(y1 + h1 - y2) <= tolerance or abs(y2 + h2 - y1) <= tolerance:
        return None  # The rectangles are adjacent vertically
    else:
        # The rectangles are not adjacent
        return None


def merge_close_rectangles(rectangles: List[Tuple[float]], tolerance=10):
    """
    A function to merge a list of rectangles that are close to each other
    """
    while True:
        merged = False
        for i in range(len(rectangles)):
            for j in range(i + 1, len(rectangles)):
                rect1 = rectangles[i]
                rect2 = rectangles[j]
                merged_rect = merge_rectangles(rect1, rect2, tolerance)
                if merged_rect is not None:
                    # Remove the original rectangles and add the merged rectangle
                    rectangles.remove(rect1)
                    rectangles.remove(rect2)
                    rectangles.append(merged_rect)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    return rectangles


def merge_rectangle_line(rectangles: List[Tuple[float]]):
    """
    A line of rectangles is a set of rectangles that are adjacent horizontally
    """
    min_x = min([x for x, y, w, h in rectangles])
    min_y = min([y for x, y, w, h in rectangles])
    max_x = max([x + w for x, y, w, h in rectangles])
    max_y = max([y + h for x, y, w, h in rectangles])
    merged = (min_x, min_y, max_x - min_x, max_y - min_y)
    return merged


def find_rectangle_centers(rectangles: List[Tuple[float]]):
    """
    A function to find the center of a list of rectangles
    """
    centers = []
    for x, y, w, h in rectangles:
        centers.append((x + w / 2, y + h / 2))
    return centers


def find_rectangle_lines(rectangles: List[Tuple[float]], tolerance=10):
    """
    A function to find lines of rectangles
    """
    centers = find_rectangle_centers(rectangles)
    center_to_index = {center: i for i, center in enumerate(centers)}
    lines = []
    while len(centers) > 0:
        line = []
        center = centers.pop(0)
        line.append(center)
        for i in range(len(centers)):
            center2 = centers[i]
            if abs(center[1] - center2[1]) <= tolerance:
                line.append(center2)
        for center in line[1:]:
            centers.remove(center)
        lines.append([rectangles[center_to_index[center]] for center in line])
    return lines


def merge_rectangle_lines(rectangles: List[Tuple[float]], tolerance=10):
    """
    A function to merge lines of rectangles
    """
    lines = find_rectangle_lines(rectangles, tolerance)
    merged_lines = []
    for line in lines:
        merged_lines.append(merge_rectangle_line(line))
    return merged_lines


def crop_image(img, vertical: bool = True, horizontal: bool = False):
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
    min_indices_vertical = non_white_pixels.min(axis=0)
    max_indices_vertical = non_white_pixels.max(axis=0)

    min_indices_horizontal = non_white_pixels.min(axis=1)
    max_indices_horizontal = non_white_pixels.max(axis=1)

    # Crop the image to the smallest possible size containing all non-white pixels.
    if vertical:
        img = img[min_indices_vertical[0] : max_indices_vertical[0] + 1, :]
    if horizontal:
        img = img[:, min_indices_horizontal[0] : max_indices_horizontal[0] + 1]
    return img


def embed_image(img: np.ndarray, width=368, height=368):
    """
    Embeds an image in a larger image of a given size.

    Parameters:
        img (np.ndarray): A numpy array representing the image to embed.
        width (int, optional): The width of the larger image. Defaults to 368.
        height (int, optional): The height of the larger image. Defaults to 368.

    Returns:
        np.ndarray: The larger image containing the embedded image.
    """

    # Create a larger image of the given size.
    if img.shape[0] == height and img.shape[1] == width:
        return img

    if len(img.shape) == 2:
        embedded_img = np.ones((height, width), dtype=img.dtype) * img.max()
    else:
        embedded_img = (
            np.ones((height, width, img.shape[2]), dtype=img.dtype) * img.max()
        )

    # Calculate the offset of the embedded image.
    offset = 0, (width - img.shape[1]) // 2

    # Embed the image in the larger image. Starting from the top and centering the image horizontally.
    embedded_img[: img.shape[0], offset[1] : offset[1] + img.shape[1]] = img

    return embedded_img


def concatenate_images(images: List[np.ndarray], axis=0, max_heigh=368):
    """
    Concatenates a list of scans along a given axis.

    Parameters:
        scans (List[np.ndarray]): A list of numpy arrays representing the scans to concatenate.
        axis (int, optional): The axis along which to concatenate the scans. Defaults to 0.

    Returns:
        np.ndarray: The concatenated scans.
    """

    concatenated_scans = np.concatenate(images, axis=axis)
    concatenated_scans = concatenated_scans[:max_heigh]
    assert concatenated_scans.shape[0] <= max_heigh
    return concatenated_scans


def plot_arrays(arrays: List[np.ndarray], titles=None) -> Image:
    # get the number of arrays
    n: int = len(arrays)
    # get the shape of each array
    shape: tuple = arrays[0].shape
    # compute the number of rows and columns for the grid
    rows: int = math.ceil(math.sqrt(n))
    cols: int = math.ceil(n / rows)
    # create a figure with rows x cols subplots
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    # adjust the spacing and margins of the subplots
    fig.subplots_adjust(wspace=-0.1, hspace=-0.1)
    fig.tight_layout(pad=0.01)
    # loop through the arrays and plot them on each subplot
    for i in range(n):
        # get the i-th array and subplot
        array: np.ndarray = arrays[i]
        # compute the row and column index for the subplot
        r: int = i // cols
        c: int = i % cols
        ax = axes[r][c]
        # plot the array as an image
        if len(shape) == 2:
            ax.imshow(array, cmap="gray")
        else:
            ax.imshow(array)
        # ax.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)
        # set the title as the index of the array
        if titles:
            ax.set_title(titles[i])
    # save the figure to a buffer
    buf = io.BytesIO()
    fig.subplots_adjust(top=0.95)
    fig.savefig(buf, format="png", dpi=300)
    # close the figure
    plt.close(fig)
    # create a PIL Image object from the buffer
    img = Image.open(buf)
    # return the image object
    return img


def plot_patch_mask(mask, patch_size=16):
    if type(mask) == torch.Tensor:
        mask = mask.detach().cpu().numpy()
    if len(mask.shape) == 2:
        mask = mask.squeeze(0)

    mask_square_size = int(np.sqrt(mask.shape[0]))
    mask = mask.reshape(mask_square_size, mask_square_size)

    if mask.max() == 1 and mask.min() == 0:
        mask = mask * 255
    elif mask.max() == 1 and mask.min() == -1:
        mask = (mask + 1) * 255 / 2

    mask = mask.astype(np.uint8)
    mask = np.kron(mask, np.ones((patch_size, patch_size)))
    plt.imshow(mask, cmap="gray")
    plt.savefig("/home/knf792/PycharmProjects/pixel-2/pixel_datasets/results/mask.png")


def convert_torch_tensor_to_image(tensor):
    """
    a function to convert a torch tensor to an image
    """
    im = tensor.detach().cpu().numpy()
    im = np.transpose(im, (1, 2, 0))
    if im.max() == 1 and im.min() == 0:
        im = im * 255
    elif im.max() == 1 and im.min() == -1:
        im = (im + 1) * 255 / 2
    im = im.astype(np.uint8)
    return Image.fromarray(im)


def generate_rectangle_for_matched_answer(match, word, data_dict, all_text_offest_map):
    start_id = all_text_offest_map[match]
    end_id = all_text_offest_map[match + len(word) - 1]

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


def locate_word_within_scan(data_dict, word):
    all_text_offest_map = dict()
    all_texts = ""
    for i, text in enumerate(data_dict["text"]):
        if text.strip() != "":
            for j in range(len(text) + 1):
                all_text_offest_map[len(all_texts) + j] = i
            all_texts += text + " "

    match = all_texts.find(word)
    return match, all_text_offest_map


def select_random_word_from_scan(data_dict):
    valid_words = [
        i for i, word in enumerate(data_dict["text"]) if len(word.strip()) > 3
    ]
    random_word_loc = np.random.choice(valid_words)
    return data_dict["text"][random_word_loc]


def mask_single_word_from_scan(im, word: str = None):
    pixel_mask = np.zeros(im.size[:2])
    while pixel_mask.sum() == 0:
        if type(im) == np.ndarray:
            if im.max() == 1 and im.min() == 0:
                im = im * 255
            im = Image.fromarray(im.astype(np.uint8))

        data_dict = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)
        if word is None:
            word = select_random_word_from_scan(data_dict)
        match, all_text_offest_map = locate_word_within_scan(data_dict, word)
        all_rectangles = generate_rectangle_for_matched_answer(
            match, word, data_dict, all_text_offest_map
        )
        matched_rectangles = merge_rectangle_lines(all_rectangles, tolerance=10)
        pixel_mask = generate_pixel_mask_from_recangles(matched_rectangles, im.size)
    for i in range(10):
        pixel_mask = binary_dilation(pixel_mask)
    return pixel_mask, data_dict, word


def convert_patch_mask_to_pixel_mask(mask):
    if len(mask.shape) == 1:
        mask = mask.reshape((23, 23))
    return np.kron(mask, np.ones((16, 16)))


def merge_mask_with_image(mask, image, alpha=0.5, colour=(255, 0, 0)):
    mask = np.stack([mask, mask, mask], axis=-1)
    if colour is not None:
        colour_mask = mask * np.array(colour)
        image[mask == 1] = image[mask == 1] * alpha + colour_mask[mask == 1] * (
            1 - alpha
        )
    else:
        image[mask == 1] = image[mask == 1] * alpha

    return image


def get_mask_edges(mask, times=1):
    # mask is a 2D numpy array of 0s and 1s
    # return an array of the same size where only the edges of each connected component are 1
    # use binary erosion to shrink each component by one pixel
    eroded_mask = binary_erosion(mask)
    for _ in range(times - 1):
        eroded_mask = binary_erosion(eroded_mask)
    # subtract the eroded mask from the original mask to get the edges
    edges = mask - eroded_mask
    # return the edges as an array of 0s and 1s
    return edges
