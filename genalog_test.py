from genalog.pipeline import AnalogDocumentGeneration
from genalog.degradation.degrader import ImageState
from genalog.degradation import effect
import requests

from genalog.degradation.degrader import Degrader, ImageState
from genalog.generation.content import CompositeContent, ContentType
from genalog.generation.document import DEFAULT_STYLE_COMBINATION
from genalog.generation.document import DocumentGenerator
from weasyprint import HTML
from PIL import Image

STYLE_COMBINATIONS = {
    "language": ["en_US"],
     "font_family": ["Segeo UI"],
     "font_size": ["12px"],
     "text_align": ["justify"],
     "hyphenate": [True],
}

HTML_TEMPLATE = "text_block.html.jinja"

DEGRADATIONS = [
    ("blur", {"radius": 5}),
    ("bleed_through", {
        "src": ImageState.CURRENT_STATE,
        "background": ImageState.ORIGINAL_STATE,
        "alpha": 0.8,
        "offset_x": -6,
        "offset_y": -12,
    }),
    ("morphology", {"operation": "open", "kernel_shape":(9,9), "kernel_type":"plus"}),
    ("pepper", {"amount": 0.005}),
    ("salt", {"amount": 0.15}),
]


sample_text_url = "https://raw.githubusercontent.com/microsoft/genalog/main/example/sample/generation/example.txt"
sample_text = "example.txt"

r = requests.get(sample_text_url, allow_redirects=True)
open(sample_text, "wb").write(r.content)

IMG_RESOLUTION = 300 # dots per inch (dpi) of the generated pdf/image
doc_generation = AnalogDocumentGeneration(styles=STYLE_COMBINATIONS, degradations=DEGRADATIONS, resolution=IMG_RESOLUTION, template_path=None)
img_array = doc_generation.generate_img(sample_text, HTML_TEMPLATE, target_folder=None) # returns the raw image bytes if target_folder is not specified
im = Image.fromarray(img_array)
im.save("test.png")