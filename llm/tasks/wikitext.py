import os 
from .task_utils import AbstractTask 
from typing import List, Dict, Any
import re 
from datasets import load_dataset, load_from_disk

class WikiText(AbstractTask): 
    name = "wikitext"
    def __init__(self, 
                 splits=["train", "validation", "test"], 
                 load_local = False, 
                 local_dir = None): 
        super().__init__(splits=splits, load_local=load_local, local_dir=local_dir)

    def get_datasets(self, **dataset_kwargs):

        assert "splits" in dataset_kwargs, f"dataset_kwargs must contain 'splits'" 
        splits = dataset_kwargs["splits"]
        dsets = {}
        if isinstance(splits, list): 
            for spl in splits: 
                dset = self.load_dataset(split=spl, load_local=dataset_kwargs["load_local"], local_dir=dataset_kwargs["local_dir"])
                dsets[spl] = dset
        else: 
            assert isinstance(splits, str), f"splits is neither list nor str. It MUST be one of the two."
            dset = self.load_dataset(split=splits, load_local=dataset_kwargs["load_local"], local_dir=dataset_kwargs["local_dir"])
            dsets[splits] = dset
        return dsets

    def load_dataset(self, split, load_local=False, local_dir=None): 
        if load_local: 
            assert local_dir is not None, "load_local specified, but local_dir is None"
            dset = load_from_disk(os.path.join(local_dir, split))
        else: 
            dset = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split=split)
        return dset

    def pre_process_fn(self, examples: List[Dict]) -> Any:
        if isinstance(examples, list):
            outputs = []
            for example in examples:
                processed = wikitext_detokenizer(example)
                # options = "\n".join([f"{opt}: " + e for opt, e in zip(["A", "B", "C", "D"], example["endings"])]) + "\n\n"
                # answer = f'Answer: {["A", "B", "C", "D"][int(example["label"])]}'
                outputs.append(processed)
            return outputs
        else: 
            processed = wikitext_detokenizer(examples)
            return processed


def wikitext_detokenizer(doc):
    string = doc["page"]
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string
