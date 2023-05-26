import datasets
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import numpy as np
from dataset_synthesis.document_syntesis import DocumentSynthesizer
from pixel import get_attention_mask, glue_strip_spaces

task_to_keys = {
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

class GlueDataset(Dataset):
    def __init__(self, args, task: str, data: datasets.Dataset, document_synthesizer=DocumentSynthesizer):
        self.args = args
        self.task = task
        self.data = data
        self.num_patches = 529
        self.ds = document_synthesizer
        self.torch_transform = ToTensorV2()

    def __len__(self):
        return len(self.data)

    def _generate_text_(self, sample: dict):
        sentence1_key, sentence2_key = task_to_keys[self.task]
        sentence_1 = glue_strip_spaces(sample[sentence1_key])
        sentence_2 = glue_strip_spaces(sample[sentence2_key]) if sentence2_key is not None else None
        full_text = f"SENTENCE 1: {sentence_1}; SENTENCE 2: {sentence_2}" if sentence_2 is not None else sentence_1

        return full_text
    
    def _torch_transform(self, image):
        image = np.asarray(image).astype("float32")
        image = np.stack([image, image, image], axis=2)
        image = self.torch_transform(image=image)["image"]
        return image

    def __getitem__(self, item):
        font, font_size, spacing = self.ds.get_font_by_name(self.args.finetuning_font, 25), 25, 0.5

        text = self._generate_text_(self.data[item])
        image = self.ds.generate_base_image(text, font, spacing, deterministic=True)
        pixel_values = self._torch_transform(image)

        attention_mask = get_attention_mask(self.num_patches)
        label = self.data[item]["label"]
        inputs = {
            "pixel_values": pixel_values,  
            "num_patches": self.num_patches,
            "attention_mask": attention_mask,
            "label": label,
        }
        return inputs