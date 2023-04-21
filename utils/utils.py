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

from datasets import load_dataset
import fuzzysearch

from typing import List, Tuple, Dict, Union, Optional
import numpy as np
import copy
import wandb
import pandas as pd


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
            for j in range(i+1, len(rectangles)):
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
        centers.append((x + w/2, y + h/2))
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


def crop_image(img):
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
    min_indices = non_white_pixels.min(axis=0)
    max_indices = non_white_pixels.max(axis=0)

    # Crop the image to the smallest possible size containing all non-white pixels.
    cropped_img = img[min_indices[0]:max_indices[0]+1, :]

    return cropped_img


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
        embedded_img = np.ones((height, width, img.shape[2]), dtype=img.dtype) * img.max()

    # Calculate the offset of the embedded image.
    offset = 0, (width - img.shape[1]) // 2

    # Embed the image in the larger image. Starting from the top and centering the image horizontally.
    embedded_img[:img.shape[0], offset[1]:offset[1]+img.shape[1]] = img

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