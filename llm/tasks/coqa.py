from itertools import zip_longest
import os
import transformers.data.metrics.squad_metrics as squad_metrics
from datasets import load_dataset, load_from_disk

import torch
import numpy as np
from .task_utils import AbstractTask 

class CoQA(AbstractTask):
    def __init__(self, 
                 splits=["train", "validation"], 
                 tok=None, 
                 load_local=False, 
                 local_dir=None):
        super().__init__(splits=splits, 
                         load_local=load_local, 
                         local_dir=local_dir)
        self.tok = tok 
        assert tok is not None, f"CoQA requires tokenizer to be defined, however it is not."
        self.metrics = None

    def get_datasets(self, **dataset_kwargs):

        assert "splits" in dataset_kwargs, f"dataset_kwargs missing argument 'split'"
        datasets = {}
        if isinstance(dataset_kwargs["splits"], list):
            for spl in dataset_kwargs["splits"]:
                assert spl in ["train", "validation"]
                dataset = self.load_data_split("EleutherAI/coqa", 
                                               split=spl, 
                                               load_local=dataset_kwargs["load_local"], 
                                               local_dir=dataset_kwargs["local_dir"])
                datasets[spl] = dataset
        else: 
            assert dataset_kwargs["splits"] in ["train", "validation", "test"]
            dataset = self.load_data_split("EleutherAI/coqa", 
                                            split=dataset_kwargs["split"], 
                                            load_local=dataset_kwargs["load_local"], 
                                            local_dir=dataset_kwargs["local_dir"])
            datasets[dataset_kwargs["splits"]] = dataset
        return datasets
    def load_data_split(self, 
                        name:str, 
                        split:str, 
                        load_local:bool=False, 
                        local_dir:str=None):
        if load_local: 
            assert local_dir is not None, f"load_local is True, but local_dir is not specified"
            dataset = load_from_disk(os.path.join(local_dir, split))
        else: 
            if split == "validation": 
                split = "validation[:10%]"
            dataset = load_dataset(name, split=split)
        pass

        dataset = dataset.map(self.expand_questions, batched=True, remove_columns=["questions", "answers", "story", "source", "id", "additional_answers"])

        return dataset
    
    def expand_questions(self, example) -> list:
        list_texts = []
        for story, questions, answers in zip(example["story"], example["questions"], example["answers"]):
            texts = doc_to_text({"story": story, "questions": questions, "answers": answers})
            list_texts += texts
        # answers = doc_to_target(example)
        batched = {k: [t[k] for t in list_texts] for k in list_texts[0]}
        return batched


    def compute_metrics(self, eval_prediction, compute_result=False):
        
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
        

def doc_to_text(doc):
    # Given a passage p, the conversation history {q1, a1, . . . qi−1, ai−1}
    # and a question qi, the task is to predict the answer ai
    outputs = []
    doc_text = doc["story"] + "\n\n"
    for q, a in zip_longest(
        doc["questions"]["input_text"], doc["answers"]["input_text"][:-1]
    ):  # omit target answer ai
        question = f"Q: {q}\n\n"
        answer = f"A: {a}\n\n" if a is not None else "A:"
        outputs.append({"prompt":doc_text + question, "completion" :answer})
    return outputs


def doc_to_target(doc):
    turn_id = len(doc["questions"]["input_text"])
    # Returns unique answers and valid alternatives (Some questions in CoQA have multiple valid answers).
    answers = []
    answer_forturn = doc["answers"]["input_text"][turn_id - 1]
    answers.append(answer_forturn)

    additional_answers = doc.get("additional_answers")
    if additional_answers:
        for key in additional_answers:
            additional_answer_for_turn = additional_answers[key]["input_text"][
                turn_id - 1
            ]
            if additional_answer_for_turn.lower() not in map(str.lower, answers):
                answers.append(additional_answer_for_turn)
    return answers