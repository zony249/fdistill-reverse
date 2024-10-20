# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import inspect

import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from glue_metrics import Glue
from dist_utils import (
    MeanLayerDistance, 
    MeanPairwiseLayerTransformDist, 
    MeanPairwiseLayerTransformCosine, 
    MeanNormedPairwiseCosine, 
    MeanNormedCosine, 
    MeanPairwiseCosine, 
    StructuredCosine, 
    StructuredCosineHistogram, 
    CosineFromStudent
    )


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

from trainer import Trainer


logger = logging.getLogger(__name__)

class Writer(object):
    def __init__(self, dir): 
        self.filename = os.path.join(dir, "logs.txt")
        self.file = open(self.filename, "w")
        self.out = sys.stdout 

    def write(self, data):
        self.out.write(data)
        self.file.write(data)
        self.flush()

    def flush(self): 
        self.out.flush() 
        self.file.flush()

    def close(self): 
        self.file.flush() 
        self.file.close()


class MetricSuite: 
    def __init__(self, savedir, config1, config2=None, center_hidden_states=False, from_student_layer=1):
        self.savedir = savedir 
        self.config1 = config1
        self.teacher_pairwise_dist = MeanPairwiseLayerTransformDist() 
        # self.teacher_pairwise_cosines = MeanNormedPairwiseCosine() if center_hidden_states else MeanPairwiseCosine()
        self.teacher_pairwise_cosines = StructuredCosine()
        self.teacher_mean_dist = MeanLayerDistance() 
        self.teacher_mean_cosines = MeanNormedCosine()

        self.config2 = config2 
        if self.config2 is not None: 
            self.teacher_student_pairwise_dist = MeanPairwiseLayerTransformDist() 
            # self.teacher_student_pairwise_cosines = MeanNormedPairwiseCosine() if center_hidden_states else MeanPairwiseCosine()
            self.teacher_student_pairwise_cosines = StructuredCosine()
            self.teacher_student_mean_dist = MeanLayerDistance()
            self.teacher_student_mean_cosines = MeanNormedCosine(between_model=False)
            self.teacher_student_cosine_histogram = StructuredCosineHistogram()

            self.cosine_from_student = CosineFromStudent(from_student_layer=from_student_layer)

            self.student_pairwise_dist = MeanPairwiseLayerTransformDist() 
            # self.student_pairwise_cosines = MeanNormedPairwiseCosine() if center_hidden_states else MeanPairwiseCosine()
            self.student_pairwise_cosines = StructuredCosine()
            self.student_mean_dist = MeanLayerDistance()
            self.student_mean_cosines = MeanNormedCosine()

    def compute_accum(self, hidden_t, hidden_s=None): 
        """
        hidden_t = Tuple[torch.Tensor[batch, seq_len, hidden]]
        """
        if hidden_s is None and not self.config2 is None: 
            raise ValueError("student model exists, yet hidden_s is none!")
        hidden_t_stacked = torch.stack(hidden_t, dim=1) #[batch, layers, seq_len, hidden] 
        hidden_s_stacked = torch.stack(hidden_s, dim=1) if hidden_s is not None else None
        self.teacher_pairwise_dist(hidden_t_stacked, hidden_t_stacked).accum()
        # self.teacher_pairwise_cosines(hidden_t_stacked, hidden_t_stacked).accum()
        self.teacher_mean_dist(self.teacher_pairwise_dist.get_val()).accum() 
        # self.teacher_mean_cosines(self.teacher_pairwise_cosines.get_val()).accum() 
        if self.config2 is not None: 
            self.teacher_student_pairwise_dist(hidden_t_stacked, hidden_s_stacked).accum()
            self.teacher_student_pairwise_cosines(hidden_t_stacked, hidden_s_stacked).accum()
            self.teacher_student_mean_dist(self.teacher_student_pairwise_dist.get_val()).accum()
            self.teacher_student_mean_cosines(self.teacher_student_pairwise_cosines.get_val()).accum()
            self.teacher_student_cosine_histogram(self.teacher_student_pairwise_cosines.get_val()).accum()

            self.cosine_from_student(hidden_t_stacked, hidden_s_stacked).accum()

            self.student_pairwise_dist(hidden_s_stacked, hidden_s_stacked).accum()
            # self.student_pairwise_cosines(hidden_s_stacked, hidden_s_stacked).accum()
            self.student_mean_dist(self.student_pairwise_dist.get_val()).accum()
            # self.student_mean_cosines(self.student_pairwise_cosines.get_val()).accum()

    def savefigs(self): 
        teacher_teacher_dist = self.teacher_pairwise_dist.get_mean().detach().cpu()
        fig, ax = plt.subplots(ncols=1, figsize=(5, 4), gridspec_kw={'wspace': 0.5})
        im1 = ax.imshow(teacher_teacher_dist)
        ax.set_title("Encoder")
        ax.set_xlabel("Teacher Layer j")
        ax.set_ylabel("Teacher Layer i")

        fig.colorbar(im1, ax=ax)

        plt.savefig(os.path.join(self.savedir, "teacher-pairwise-dist"))
        plt.close()

        # teacher_teacher_cosines = self.teacher_pairwise_cosines.get_mean().detach().cpu()
        # fig, ax = plt.subplots(ncols=1, figsize=(5, 4), gridspec_kw={'wspace': 0.5})
        # im1 = ax.imshow(teacher_teacher_cosines)
        # ax.set_title("Encoder")
        # ax.set_xlabel("Teacher Layer j")
        # ax.set_ylabel("Teacher Layer i")

        # fig.colorbar(im1, ax=ax)

        # plt.savefig(os.path.join(self.savedir, "teacher-pairwise-cosines"))
        # plt.close()

        if self.config2 is not None: 
            teacher_student_dist = self.teacher_student_pairwise_dist.get_mean().detach().cpu()
            fig, ax = plt.subplots(ncols=1, figsize=(5, 4), gridspec_kw={'wspace': 0.5})
            im1 = ax.imshow(teacher_student_dist)
            ax.set_title("Encoder")
            ax.set_xlabel("Student Layer j")
            ax.set_ylabel("Teacher Layer i")

            fig.colorbar(im1, ax=ax)

            plt.savefig(os.path.join(self.savedir, "teacher-student-pairwise-dist"))
            plt.close()

            teacher_student_cosines = self.teacher_student_pairwise_cosines.get_mean().detach().cpu()
            fig, ax = plt.subplots(ncols=1, figsize=(5, 4), gridspec_kw={'wspace': 0.5})
            im1 = ax.imshow(teacher_student_cosines)
            ax.set_title("Encoder")
            ax.set_xlabel("Student Layer j")
            ax.set_ylabel("Teacher Layer i")

            fig.colorbar(im1, ax=ax)

            plt.savefig(os.path.join(self.savedir, "teacher-student-pairwise-cosines"))
            plt.close()


            teacher_student_cosine_hist = self.teacher_student_cosine_histogram.get_mean().detach().cpu() 
            plt.hist(teacher_student_cosine_hist, bins=40, range=(-1., 1.))
            plt.savefig(os.path.join(self.savedir, "transform_sim_hist"))
            plt.close() 


            cosines_from_student = self.cosine_from_student.get_mean()
            plt.hist(cosines_from_student, bins=1000, range=(-1., 1.))
            plt.savefig(os.path.join(self.savedir, "student_directions_hist"))
            plt.close() 




            student_student_dist = self.student_pairwise_dist.get_mean().detach().cpu()
            fig, ax = plt.subplots(ncols=1, figsize=(5, 4), gridspec_kw={'wspace': 0.5})
            im1 = ax.imshow(student_student_dist)
            ax.set_title("Encoder")
            ax.set_xlabel("Student Layer j")
            ax.set_ylabel("Student Layer i")

            fig.colorbar(im1, ax=ax)

            plt.savefig(os.path.join(self.savedir, "student-pairwise-dist"))
            plt.close()

            # student_student_cosines = self.student_pairwise_cosines.get_mean().detach().cpu()
            # fig, ax = plt.subplots(ncols=1, figsize=(5, 4), gridspec_kw={'wspace': 0.5})
            # im1 = ax.imshow(student_student_cosines)
            # ax.set_title("Encoder")
            # ax.set_xlabel("Student Layer j")
            # ax.set_ylabel("Student Layer i")

            # fig.colorbar(im1, ax=ax)

            # plt.savefig(os.path.join(self.savedir, "student-pairwise-cosines"))
            # plt.close()


            # Saving histogram to text file 
            hist, bins = np.histogram(cosines_from_student, bins=1000, range=(-1., 1.))
            print(hist.shape, bins.shape)
            data = np.stack([bins[:hist.shape[0]], hist], axis=-1)
            np.savetxt(os.path.join(self.savedir, "student_directions_hist.txt"), data, delimiter=',')

            with open(os.path.join(self.savedir, "between-model-pairwise-dist.csv"), "w") as f:
                # indices
                f.write(f"layer_index,")
                for i in range(teacher_student_dist.shape[1]):
                    f.write(f"{i}{',' if i != teacher_student_dist.shape[1]-1 else ''}")
                f.write("\n")
                for i in range(teacher_student_dist.shape[0]): 
                    f.write(f"{i},")
                    for j in range(teacher_student_dist.shape[1]): 
                        f.write(f"{teacher_student_dist[i, j].item()}{',' if j != teacher_student_dist.shape[1]-1 else ''}")
                    f.write("\n")

            with open(os.path.join(self.savedir, "between-model-pairwise-cosines.csv"), "w") as f:
                # indices
                f.write(f"layer_index,")
                for i in range(teacher_student_cosines.shape[1]):
                    f.write(f"{i}{',' if i != teacher_student_cosines.shape[1]-1 else ''}")
                f.write("\n")
                for i in range(teacher_student_cosines.shape[0]): 
                    f.write(f"{i},")
                    for j in range(teacher_student_cosines.shape[1]): 
                        f.write(f"{teacher_student_cosines[i, j].item()}{',' if j != teacher_student_cosines.shape[1]-1 else ''}")
                    f.write("\n")

        
        with open(os.path.join(self.savedir, "stats.csv"), "w") as f: 
            f.write(f"teacher_mean_dist,{self.teacher_mean_dist.get_val()}\n")
            # f.write(f"teacher_mean_cosines,{self.teacher_mean_cosines.get_val()}\n")
            if self.config2 is not None: 
                f.write(f"student_mean_dist,{self.student_mean_dist.get_val()}\n")
                # f.write(f"student_mean_cosines,{self.student_mean_cosines.get_val()}\n")
                f.write(f"between_model_mean_dist,{self.teacher_student_mean_dist.get_val()}\n")
                f.write(f"between_model_mean_cosines,{self.teacher_student_mean_cosines.get_val()}\n")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    student_model: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to pretrained student model to compare with teacher"}
    )
    
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    center_hidden_states: Optional[bool] = field(
        default=False,
        metadata={
            "help": "centers hidden states for cosine similarity measurement"
        },
    )
    from_student_layer: Optional[int] = field(
        default=1, 
        metadata={"help": "from which student layer to anchor cosine similarity observations"}
    )




def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir)
    redir = Writer(training_args.output_dir)
    sys.stdout = redir
    sys.stderr = redir

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("nyu-mll/glue", data_args.task_name, cache_dir="glue_dataset")
    else:
        raise ValueError("task_name cannot be None")
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
    )
    student_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.student_model,
    ) if model_args.student_model is not None else None

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        raise ValueError("task_name must be defined")

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=data_args.max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    if data_args.task_name is not None or data_args.test_file is not None:
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    # Get the metric function
    if data_args.task_name is not None:
        # metric = load_metric("glue", data_args.task_name)
        metric = Glue(data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics


    if data_args.task_name is None:
        training_args.metric_for_best_model = "combined_score"
        training_args.greater_is_better = True
    elif is_regression:
        training_args.metric_for_best_model = "mse"
        training_args.greater_is_better = False
    else:
        training_args.metric_for_best_model = "accuracy"
        training_args.greater_is_better = True

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    model.to(training_args.device)
    if student_model is not None:
        student_model.to(training_args.device)

    model.eval()
     


    def _remove_unused_columns(dataset: Dataset, description: Optional[str] = None):
        if not training_args.remove_unused_columns:
            return
        # Inspect model forward signature to keep only the arguments it accepts.
        signature = inspect.signature(model.forward)
        signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        signature_columns += ["label", "label_ids"]
        columns = [k for k in signature_columns if k in dataset.column_names]
        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        dset_description = "" if description is None else f"in the {description} set "
        logger.info(
            f"The following columns {dset_description}don't have a corresponding argument in `{model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
        )
        dataset.set_format(type=dataset.format["type"], columns=columns)

    _remove_unused_columns(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, training_args.eval_batch_size, sampler=SequentialSampler(eval_dataset), collate_fn=default_data_collator)

    mets = []
    output_cat = None
    labels_cat = None
    with torch.no_grad():
        for step, batch in enumerate(tqdm(eval_dataloader, desc=f"Evaluating {data_args.task_name}")): 
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(training_args.device)
            if data_args.task_name in []: 
                outputs = model(batch["input_ids"], attention_mask = batch["attention_mask"], labels = batch["labels"])
            else:
                outputs = model(**batch)
            output_cat = outputs.logits if output_cat is None else torch.cat([output_cat, outputs.logits], dim=0)
            labels_cat = batch["labels"] if labels_cat is None else torch.cat([labels_cat, batch["labels"]], dim=0)

        @dataclass 
        class AnClass:
            predictions: torch.Tensor
            label_ids: torch.Tensor
            # mets.append(compute_metrics(AnClass(predictions=outputs.logits.detach().cpu().numpy(), label_ids=batch["labels"].detach().cpu().numpy())))
        mets = compute_metrics(AnClass(predictions=output_cat.detach().cpu().numpy(), label_ids=labels_cat.detach().cpu().numpy()))
    print("METRICS: ", mets)

    
    _remove_unused_columns(test_dataset)
    tasks = [data_args.task_name]
    test_loaders = [DataLoader(test_dataset, training_args.eval_batch_size, sampler=SequentialSampler(test_dataset), collate_fn=default_data_collator)]
    if data_args.task_name == "mnli": 
        tasks.append("mnli-mm")
        mismatched = datasets["test_mismatched"]
        _remove_unused_columns(mismatched)
        test_loaders.append(DataLoader(mismatched, training_args.eval_batch_size, sampler=SequentialSampler(mismatched), collate_fn=default_data_collator))

    if student_model is None: 
        metric_suite = MetricSuite(training_args.output_dir, model.config)
    else: 
        metric_suite = MetricSuite(training_args.output_dir, model.config, student_model.config)

    with torch.no_grad():
        for task, test_loader in zip(tasks, test_loaders):
            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            with open(output_test_file, "a") as writer:
                writer.write(f"index\tprediction\n")
            counter = 0
            for step, batch in enumerate(tqdm(test_loader, desc=f"Testing {task}:")): 
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(training_args.device)
                batch["labels"] = None
                outputs = model(**batch, output_hidden_states=True)
                predictions = outputs.logits[:, 0] if is_regression else torch.argmax(outputs.logits, dim=-1)

                if student_model is not None: 
                    student_outputs = student_model(**batch, output_hidden_states=True)
                    predictions = outputs.logits[:, 0] if is_regression else torch.argmax(outputs.logits, dim=-1)

                with open(output_test_file, "a") as writer:
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{counter}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{counter}\t{item}\n")
                        counter += 1
                if student_model is None: 
                    metric_suite.compute_accum(outputs.hidden_states)
                else: 
                    metric_suite.compute_accum(outputs.hidden_states, student_outputs.hidden_states)
                metric_suite.savefigs() 
            


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
