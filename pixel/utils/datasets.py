from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset
from transformers.file_utils import ModelOutput

UPOS_LABELS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]
UD_HEAD_LABELS = [
    "_",
    "acl",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "aux",
    "case",
    "cc",
    "ccomp",
    "clf",
    "compound",
    "conj",
    "cop",
    "csubj",
    "dep",
    "det",
    "discourse",
    "dislocated",
    "expl",
    "fixed",
    "flat",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod",
    "nsubj",
    "nummod",
    "obj",
    "obl",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "root",
    "vocative",
    "xcomp",
]


class Split(Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


@dataclass
class DependencyParsingModelOutput(ModelOutput):
    """
    Class for outputs of dependency parsing models.
    """

    loss: Optional[torch.FloatTensor] = None
    arc_logits: torch.FloatTensor = None
    rel_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None



class InterleaveTorchDataset(Dataset):
    
    def __init__(self, datasets: List[Dataset], rng):
        self.datasets = datasets
        self.rng = rng
        
        self.smallest_dataset_length = min([len(d) for d in datasets])
        self.num_of_datasets = len(datasets)
        
    def __len__(self):
        return self.smallest_dataset_length * self.num_of_datasets
    
    def __getitem__(self, index: int):
        
        dataset_id = index % self.num_of_datasets
        inside_dataset_id = index // self.num_of_datasets
        
        return self.datasets[dataset_id][inside_dataset_id]