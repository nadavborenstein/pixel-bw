from typing import List, Tuple
import numpy as np
from math import ceil, floor


def merge_rectangles(rect1: Tuple[float], rect2: Tuple[float], tolerance=5):
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


def generate_pixel_mask_from_recangles(
    rectangles: List[Tuple[int]], img_shape: Tuple[int]
):
    pixel_image = np.zeros(img_shape)
    for rectangle in rectangles:
        x = rectangle[0]
        y = rectangle[1]
        w = rectangle[2]
        h = rectangle[3]
        pixel_image[y : y + h + 1, x : x + w + 1] = 1
    return pixel_image


def convert_pixel_mask_to_patch_mask(
    pixel_mask: np.ndarray, patch_size: int = 16, tolerance: float = 0.5
):
    """
    A function to convert a pixel mask to a patch mask using its mean value
    """
    patch_mask = np.zeros(
        (pixel_mask.shape[0] // patch_size, pixel_mask.shape[1] // patch_size)
    )
    for i in range(patch_mask.shape[0]):
        for j in range(patch_mask.shape[1]):
            patch_mask[i, j] = np.mean(
                pixel_mask[
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
            )
    patch_mask = np.where(patch_mask > tolerance, 1, 0)
    return patch_mask


def generate_patch_mask_from_recangles(
    rectangles: List[Tuple[int]], img_shape: Tuple[int], patch_size: int = 16
):
    patch_image = np.zeros((img_shape[0] // patch_size, img_shape[1] // patch_size))
    for rectangle in rectangles:
        x_patch = floor(rectangle[0] / patch_size)
        y_patch = floor(rectangle[1] / patch_size)

        w_patch = round((rectangle[0] + rectangle[2]) / patch_size)
        h_patch = round((rectangle[1] + rectangle[3]) / patch_size)

        patch_image[y_patch : h_patch + 1, x_patch : w_patch + 1] = 1
    return patch_image, np.kron(patch_image, np.ones((patch_size, patch_size)))


def merge_close_rectangles(rectangles: List[Tuple[float]], tolerance=10):
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
