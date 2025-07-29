import os 
import sys 
from copy import deepcopy 
from typing import Tuple, List, Union, Dict, Any, Optional
from abc import ABC, abstractmethod

import torch 
from torch.utils.data import Dataset
from transformers import EvalPrediction
from datasets import load_dataset 

class AbstractTask: 
    name: str 
    def __init__(self, **dataset_kwargs): 
        self.datasets = self.get_datasets(**dataset_kwargs) # {"split": dataset}
    @abstractmethod
    def get_datasets(self, **dataset_kwargs): 
        raise NotImplementedError 
    @abstractmethod
    def compute_metrics(self, eval_prediction: EvalPrediction, compute_result: bool) -> Any:
        raise NotImplementedError