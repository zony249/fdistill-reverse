import os 
from copy import deepcopy 
from typing import Tuple, List, Union, Dict, Any, Optional 
import re

import datasets
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk

from .task_utils import AbstractTask

class Hellaswag(AbstractTask): 
    name: str = "hellaswag" 
    def __init__(self, split="train", 
                 load_local=False, 
                 local_dir=None): 

        super().__init__(split=split, 
                         load_local=load_local, 
                         local_dir=local_dir)
        self.metrics = None

    def get_datasets(self, **dataset_kwargs) -> Dict[str, Dataset]: 
        """
        For Hellaswag, we look for the "split", "local_dir", and "load_from_disk" kwarg from dataset_kwargs 
        split: Union[List[str], str]
        """
        assert "split" in dataset_kwargs, f"dataset_kwargs missing argument 'split'"
        datasets = {}
        if isinstance(dataset_kwargs["split"], list):
            for spl in dataset_kwargs["split"]:
                assert spl in ["train", "validation", "test"]
                dataset = self.load_data_split("Rowan/hellaswag", 
                                               split=spl, 
                                               load_local=dataset_kwargs["load_local"], 
                                               local_dir=dataset_kwargs["local_dir"])
                datasets[spl] = dataset
        else: 
            assert dataset_kwargs["split"] in ["train", "validation", "test"]
            dataset = self.load_data_split("Rowan/hellaswag", 
                                            split=dataset_kwargs["split"], 
                                            load_local=dataset_kwargs["load_local"], 
                                            local_dir=dataset_kwargs["local_dir"])
            datasets[dataset_kwargs["split"]] = dataset
        return datasets

    def load_data_split(self, 
                        name: str, 
                        split:str, 
                        load_local:Optional[bool]=False, 
                        local_dir:Optional[str]=None) -> Dataset: 
        if load_local: 
            assert local_dir is not None, f"load_from_disk is set to {load_local}, however local_dir is None." 
            dataset = load_from_disk(os.path.join(local_dir, split))
        else: 
            if split == "validation": 
                split = "validation[:10%]"
            dataset = load_dataset(name, split=split)
        dataset = process_docs(dataset)
        return dataset


    def pre_process_fn(self, examples: List[Dict]) -> Any:
        """
        DEPRECATED
        """
        if isinstance(examples, list):
            outputs = []
            for example in examples:
                question = f"TEST TEXT: {example['ctx_a']} {example['ctx_b'].capitalize()}" + example["endings"][int(example["label"])]
                # options = "\n".join([f"{opt}: " + e for opt, e in zip(["A", "B", "C", "D"], example["endings"])]) + "\n\n"
                # answer = f'Answer: {["A", "B", "C", "D"][int(example["label"])]}'
                outputs.append(question)
            return outputs
        else: 
            question = f"TEST TEXT: {examples['ctx_a']} {examples['ctx_b'].capitalize()}" + examples["endings"][int(examples["label"])]
            return question
        
    def compute_metrics(self, eval_prediction, compute_result): 
        pass
        preds = eval_prediction[0] 
        label_ids = eval_prediction[1]

        preds = preds[..., :-1, :].contiguous()
        label_ids = label_ids[..., 1:].contiguous()


        assert len(preds) == len(label_ids) 

        matches = 0 
        total = 1

        matches += torch.sum(torch.argmax(preds, dim=-1) == label_ids)
        total += torch.sum(label_ids != -100)

        if self.metrics is None: 
            self.metrics = {
                "matches": matches, 
                "total": total
            }
        else: 
            self.metrics["matches"] += matches 
            self.metrics["total"] += total 

        if compute_result: 
            output =  {"mean_token_accuracy": self.metrics["matches"] / self.metrics["total"]}
            self.metrics = None 
            return output
        return {"mean_token_accuracy": matches / total}


# This is from LM_Eval: 

def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        count=0
        try: 
            gold = int(doc["label"]) 
        except ValueError: 
            count += 1
            print(f"count: {count}. Label does not exist... defaulting to 0.")
            gold = 0
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": gold,
        }
        return out_doc

    dataset = dataset.map(_process_doc) 

    def _convert_to_template(doc): 
        out_doc = { 
            "prompt": doc["query"], 
            "completion": doc["choices"][doc["gold"]]
        }
        return out_doc
    
    return dataset.map(_convert_to_template)
    




if __name__ == "__main__": 
    hellaswag = Hellaswag("train") 
    print(hellaswag.dataset[0]) 
    print(hellaswag.pre_process_fn([hellaswag.dataset[0]]))

    dataset = load_dataset("Rowan/hellaswag", split="train")
    print(process_docs(dataset)[0])
