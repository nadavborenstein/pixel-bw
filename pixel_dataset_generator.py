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
from pprint import pprint
from utils.utils import crop_image, concatenate_images, embed_image
from torch.utils.data import Dataset

