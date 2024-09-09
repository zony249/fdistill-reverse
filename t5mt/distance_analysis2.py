#!/usr/bin/env python

import argparse
import os
import datetime
import json
import math
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List
import plotly.express as px
import matplotlib.pyplot as plt


import torch
import pandas as pd
from torch import nn 
from torch.nn import functional as F
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from modeling_t5 import T5ForConditionalGeneration
from utils import calculate_test_bleu, calculate_rouge, chunks, parse_numeric_n_bool_cl_kwargs, use_task_specific_params


from dist_utils import (
    compute_pca, 
    pca_transform, 
    create_display_df, 
    DistBetweenModels, 
    DistBetweenLayers, 
    DistPairwise, 
    MeanPairwiseLayerTransformDist, 
    MeanLayerDistance, 
    tensorize_encoder_outputs, 
    tensorize_decoder_generate_outputs)


logger = getLogger(__name__)


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LAYER_MAP = {
    6: {
        1: [5], 
        2: [2, 5], 
        3: [1, 3, 5], 
        6: list(range(6))
    }, 
    12: {
        1: [11], 
        2: [5, 11],
        3: [3, 7, 11], 
        4: [2, 5, 8, 11], 
        6: [1, 3, 5, 7, 9, 11], 
        12: list(range(12)) 
    }
}


class MetricsSuite: 
    def __init__(self, config1, config2 = None, reverse_encoder=False, reverse_decoder=False): 
        """
        """
        self.config1 = config1 
        self.config2 = config2 
        self.reverse_encoder = reverse_encoder 
        self.reverse_decoder = reverse_decoder

        self.teacher_encoder_pairwise_dist = MeanPairwiseLayerTransformDist() 
        self.teacher_decoder_pairwise_dist = MeanPairwiseLayerTransformDist() 
        self.teacher_encoder_mean_dist = MeanLayerDistance() 
        self.teacher_decoder_mean_dist = MeanLayerDistance() 

        if self.config2 is not None: 
            self.student_encoder_pairwise_dist = MeanPairwiseLayerTransformDist() 
            self.student_decoder_pairwise_dist = MeanPairwiseLayerTransformDist() 
            self.student_encoder_mean_dist = MeanLayerDistance() 
            self.student_decoder_mean_dist = MeanLayerDistance() 

            self.teacher_student_encoder_pairwise_dist = MeanPairwiseLayerTransformDist()
            self.teacher_student_decoder_pairwise_dist = MeanPairwiseLayerTransformDist()
            self.between_model_encoder_mean_dist = MeanLayerDistance() 
            self.between_model_decoder_mean_dist = MeanLayerDistance() 
            

    def compute_accum(self, enc_hidden_1, dec_hidden_1, enc_hidden_2=None, dec_hidden_2=None): 
        """
        hidden_1: torch.Tensor[batch, num_layers_1, seq_len, hidden]
        hidden_2: torch.Tensor[batch, num_layers_2, seq_len, hidden]
        
        """
        self.teacher_encoder_pairwise_dist(enc_hidden_1, enc_hidden_1).accum()
        self.teacher_decoder_pairwise_dist(dec_hidden_1, dec_hidden_1).accum()
        self.teacher_encoder_mean_dist(self.teacher_encoder_pairwise_dist.get_mean())
        self.teacher_decoder_mean_dist(self.teacher_decoder_pairwise_dist.get_mean())


        if self.config2 is not None: 
            self.student_encoder_pairwise_dist(enc_hidden_2, enc_hidden_2).accum()
            self.student_decoder_pairwise_dist(dec_hidden_2, dec_hidden_2).accum()
            self.student_encoder_mean_dist(self.student_encoder_pairwise_dist.get_mean())
            self.student_decoder_mean_dist(self.student_decoder_pairwise_dist.get_mean())

            self.teacher_student_encoder_pairwise_dist(enc_hidden_1, enc_hidden_2).accum()
            self.teacher_student_decoder_pairwise_dist(dec_hidden_1, dec_hidden_2).accum()
            self.between_model_encoder_mean_dist(self.teacher_student_encoder_pairwise_dist.get_mean())
            self.between_model_decoder_mean_dist(self.teacher_student_decoder_pairwise_dist.get_mean())

    def save_figs(self, dirname): 
        """
        """
        teacher_teacher_enc = self.teacher_encoder_pairwise_dist.get_mean().detach().cpu()
        teacher_teacher_dec = self.teacher_decoder_pairwise_dist.get_mean().detach().cpu()
        fig, ax = plt.subplots(ncols=2, figsize=(10, 4), gridspec_kw={'wspace': 0.5})
        im1 = ax[0].imshow(teacher_teacher_enc)
        im2 = ax[1].imshow(teacher_teacher_dec)
        ax[0].set_title("Encoder")
        ax[0].set_xlabel("Teacher Layer j")
        ax[0].set_ylabel("Teacher Layer i")
        ax[1].set_title("Decoder")
        ax[1].set_xlabel("Teacher Layer j")
        ax[1].set_ylabel("Teacher Layer i")

        fig.colorbar(im1, ax=ax[0])
        fig.colorbar(im2, ax=ax[1])

        plt.savefig(os.path.join(dirname, "teacher-teacher-pairwise-dist"))
        plt.close()


        if self.config2 is not None: 
            student_student_enc = self.student_encoder_pairwise_dist.get_mean().detach().cpu()
            student_student_dec = self.student_decoder_pairwise_dist.get_mean().detach().cpu()
            fig, ax = plt.subplots(ncols=2, figsize=(10, 4), gridspec_kw={'wspace': 0.5})
            im1 = ax[0].imshow(student_student_enc)
            im2 = ax[1].imshow(student_student_dec)
            ax[0].set_title("Encoder")
            ax[0].set_xlabel("Student Layer j")
            ax[0].set_ylabel("Student Layer i")
            ax[1].set_title("Decoder")
            ax[1].set_xlabel("Student Layer j")
            ax[1].set_ylabel("Student Layer i")

            fig.colorbar(im1, ax=ax[0])
            fig.colorbar(im2, ax=ax[1])

            plt.savefig(os.path.join(dirname, "student-student-pairwise-dist"))
            plt.close()

            between_enc_pairwise = self.teacher_student_encoder_pairwise_dist.get_mean().detach().cpu()
            between_dec_pairwise = self.teacher_student_decoder_pairwise_dist.get_mean().detach().cpu()
            fig, ax = plt.subplots(ncols=2, figsize=(10, 4), gridspec_kw={'wspace': 0.5})
            im1 = ax[0].imshow(between_enc_pairwise)
            im2 = ax[1].imshow(between_dec_pairwise)
            ax[0].set_title("Encoder")
            ax[0].set_xlabel("Student Layer j")
            ax[0].set_ylabel("Teacher Layer i")
            ax[1].set_title("Decoder")
            ax[1].set_xlabel("Student Layer j")
            ax[1].set_ylabel("Teacher Layer i")

            fig.colorbar(im1, ax=ax[0])
            fig.colorbar(im2, ax=ax[1])

            plt.savefig(os.path.join(dirname, "between-model-pairwise-dist"))
            plt.close()

        with open(os.path.join(dirname, "between-model-pairwise-encoder.csv"), "w") as f:
            # indices
            f.write(f"layer_index,")
            for i in range(between_enc_pairwise.shape[1]):
                f.write(f"{i}{',' if i != between_enc_pairwise.shape[1]-1 else ''}")
            f.write("\n")
            for i in range(between_enc_pairwise.shape[0]): 
                f.write(f"{i},")
                for j in range(between_enc_pairwise.shape[1]): 
                    f.write(f"{between_enc_pairwise[i, j].item()}{',' if j != between_enc_pairwise.shape[1]-1 else ''}")
                f.write("\n")
                
        with open(os.path.join(dirname, "between-model-pairwise-decoder.csv"), "w") as f:
            # indices
            f.write(f"layer_index,")
            for i in range(between_dec_pairwise.shape[1]):
                f.write(f"{i}{',' if i != between_dec_pairwise.shape[1]-1 else ''}")
            f.write("\n")
            for i in range(between_dec_pairwise.shape[0]): 
                f.write(f"{i},")
                for j in range(between_dec_pairwise.shape[1]): 
                    f.write(f"{between_dec_pairwise[i, j].item()}{',' if j != between_dec_pairwise.shape[1]-1 else ''}")
                f.write("\n")
                

        with open(os.path.join(dirname, "stats.csv"), "w") as f: 
            f.write(f"teacher_encoder_mean_dist,{self.teacher_encoder_mean_dist.get_val()}\n")
            f.write(f"teacher_decoder_mean_dist,{self.teacher_decoder_mean_dist.get_val()}\n")

            if self.config2 is not None: 
                f.write(f"student_encoder_mean_dist,{self.student_encoder_mean_dist.get_val()}\n")
                f.write(f"student_decoder_mean_dist,{self.student_decoder_mean_dist.get_val()}\n")
                f.write(f"between_model_encoder_mean_dist,{self.between_model_encoder_mean_dist.get_val()}\n")
                f.write(f"between_model_decoder_mean_dist,{self.between_model_decoder_mean_dist.get_val()}\n")


def generate_summaries_or_translations(
    examples: List[str],
    references: List[str], 
    out_file: str,
    model_name: str,
    compare_with_second_model: str = None, 
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    num_beams=5,
    task="translation",
    prefix=None,
    max_length=1024,
    top_k=None,
    do_sample=False,
    pca_mode="both", 
    unnormalized=False, 
    display_centroids=False, 
    reverse_encoder=False, 
    reverse_decoder=False, 
    **generate_kwargs,
):#-> Dict:
    """Save model.generate results to <out_file>, and return how long it took."""
    fout = Path(out_file).open("w", encoding="utf-8")
    model_name = str(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    compare_model = T5ForConditionalGeneration.from_pretrained(compare_with_second_model).to(device) if compare_with_second_model is not None else None
    if fp16:
        model = model.half()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Inferred tokenizer type: {tokenizer.__class__}")  # if this is wrong, check config.model_type.
    #print(model.model.encoder.embed_tokens.weight.size())
    start_time = time.time()
    # update config with task specific params
    use_task_specific_params(model, task)
    if prefix is None:
        prefix = prefix or getattr(model.config, "prefix", "") or ""


    if compare_model is None: 
        metrics = MetricsSuite(model.config, reverse_encoder=reverse_encoder, reverse_decoder=reverse_decoder)
    else: 
        metrics = MetricsSuite(model.config, compare_model.config, reverse_encoder=reverse_encoder, reverse_decoder=reverse_decoder)


    for examples_chunk in tqdm(list(chunks([list(a) for a in zip(examples, references)], batch_size))):
        input_chunk = ["t" + text[0][1:] for text in examples_chunk]
        decoder_chunk = [text[1] for text in examples_chunk]
        batch = tokenizer(input_chunk, max_length=max_length, return_tensors="pt", truncation=True, padding="longest").to(device)
        decoder_inputs = tokenizer(decoder_chunk, max_length=max_length, return_tensors="pt", truncation=True, padding="longest").to(device)


        # gen_config = GenerationConfig(output_hidden_states=True, **generate_kwargs)

        # print(input_chunk)
        # print(decoder_chunk)
        outputs = model.generate(
            input_ids=batch.input_ids,
            num_beams=num_beams,
            attention_mask=batch.attention_mask,
            # generation_config=gen_config, 
            output_hidden_states=True,
            return_dict_in_generate=True,  
            output_scores=True, 
            **generate_kwargs,
        )

        encoder_hidden_states = tensorize_encoder_outputs(outputs.encoder_hidden_states)
        decoder_hidden_states = tensorize_decoder_generate_outputs(outputs.decoder_hidden_states, num_beams=num_beams)

        if compare_model is not None: 
            compare_outputs =  compare_model.generate(
                input_ids=batch.input_ids,
                num_beams=num_beams,
                attention_mask=batch.attention_mask,
                # generation_config=gen_config, 
                output_hidden_states=True,
                return_dict_in_generate=True,  
                output_scores=True, 
                **generate_kwargs,
            ) 
            compare_encoder_hidden_states = tensorize_encoder_outputs(compare_outputs.encoder_hidden_states)
            compare_decoder_hidden_states = tensorize_decoder_generate_outputs(compare_outputs.decoder_hidden_states, num_beams=num_beams)
        else:
            compare_outputs = None
        

        if compare_model is not None:
            metrics.compute_accum(encoder_hidden_states, decoder_hidden_states, compare_encoder_hidden_states, compare_decoder_hidden_states)
            metrics.save_figs("/".join(out_file.split("/")[:-1]))
        else: 
            metrics.compute_accum(encoder_hidden_states, decoder_hidden_states)
            metrics.save_figs("/".join(out_file.split("/")[:-1]))





        dec = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for hypothesis in dec:
            fout.write(hypothesis + "\n")
            fout.flush()

    # encoder_hidden = [e.mean(dim=0, keepdims=True) for e in encoder_hidden]
    # decoder_hidden = [d.mean(dim=0, keepdims=True) for d in decoder_hidden]


    fout.close()
    runtime = int(time.time() - start_time)  # seconds
    n_obs = len(examples)
    return dict(n_obs=n_obs, runtime=runtime, seconds_per_sample=round(runtime / n_obs, 4))


def datetime_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_generate(verbose=True):
    """

    Takes input text, generates output, and then using reference calculates the BLEU scores.

    The results are saved to a file and returned to the caller, and printed out unless ``verbose=False`` is passed.

    Args:
        verbose (:obj:`bool`, `optional`, defaults to :obj:`True`): print results to stdout

    Returns:
        a tuple: ``(scores, params}``
        - ``scores``: a dict of scores data ``{'bleu': 39.6501, 'n_obs': 2000, 'runtime': 186, 'seconds_per_sample': 0.093}``
        - ``params``: a dict of custom params, e.g. ``{'num_beams': 5, 'length_penalty': 0.8}``
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--student_model", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--input_path", type=str, help="like cnn_dm/test.source")
    parser.add_argument("--save_path", type=str, help="where to save summaries")
    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test.target")
    parser.add_argument("--score_path", type=str, required=False, default="metrics.json", help="where to save metrics")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument(
        "--prefix", type=str, required=False, default=None, help="will be added to the begininng of src examples"
    )
    parser.add_argument("--max_input_length", type=int, required=False, default=1024)
    parser.add_argument("--top_k", type=int, required=False, default=None)
    parser.add_argument("--task", type=str, default="translation", help="used for task_specific_params + metrics")
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument("--num_beams", type=int, default=5, required=False, help="batch size")
    parser.add_argument(
        "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--dump-args", action="store_true", help="print the custom hparams with the results")
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument(
        "--info",
        nargs="?",
        type=str,
        const=datetime_now(),
        help="use in conjunction w/ --dump-args to print with the results whatever other info you'd like, e.g. lang=en-ru. If no value is passed, the current datetime string will be used.",
    )
    parser.add_argument("--pca_mode", type=str, choices=["both", "encoder_only", "decoder_only"], default="both")
    parser.add_argument("--unnormalized", action="store_true", default=False)
    parser.add_argument("--display_centroids", action="store_true", default=False)
    parser.add_argument("--reverse_encoder", action="store_true", default=False)
    parser.add_argument("--reverse_decoder", action="store_true", default=False)

    # Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate
    args, rest = parser.parse_known_args()
    parsed_args = parse_numeric_n_bool_cl_kwargs(rest)
    if parsed_args and verbose:
        print(f"parsed the following generate kwargs: {parsed_args}")
    examples = [x.rstrip() for x in open(args.input_path).readlines()]
    reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()][:len(examples)]
    if args.n_obs > 0:
        examples = examples[: args.n_obs]
    Path(args.save_path).parent.mkdir(exist_ok=True)
    if args.reference_path is None and Path(args.score_path).exists():
        warnings.warn(f"score_path {args.score_path} will be overwritten unless you type ctrl-c.")
    runtime_metrics = generate_summaries_or_translations(
        examples,
        reference_lns, 
        args.save_path,
        args.model_name,
        args.student_model, 
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        num_beams=args.num_beams,
        prefix=args.prefix,
        max_length=args.max_input_length,
        top_k=args.top_k,
        do_sample=args.do_sample,
        pca_mode=args.pca_mode, 
        unnormalized=args.unnormalized, 
        display_centroids=args.display_centroids, 
        reverse_encoder=args.reverse_encoder, 
        reverse_decoder=args.reverse_decoder, 
        **parsed_args,
    )

    if args.reference_path is None or args.unnormalized:
        return {}

    # Compute scores
    score_fn = calculate_test_bleu# if "translation" in args.task else calculate_rouge
    output_lns = [x.rstrip() for x in open(args.save_path).readlines()]
    reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()][: len(output_lns)]
    scores = score_fn(output_lns, reference_lns)
    #scores.update(runtime_metrics)
    print(scores)
    #if args.dump_args:
    #    scores.update(parsed_args)
    #if args.info:
    #    scores["info"] = args.info

    if verbose:
        print(scores)
    scores["bleu_score"] = scores["bleu_score"].score
    scores["chrf_score"] = scores["chrf_score"].score
    scores["ter_score"] = scores["ter_score"].score

    if args.score_path is not None:
        json.dump(dict(scores), open(args.score_path, "w"))

    return scores


if __name__ == "__main__":
    # Usage for MT:
    # python run_eval.py MODEL_NAME $DATA_DIR/test.source $save_dir/test_translations.txt --reference_path $DATA_DIR/test.target --score_path $save_dir/test_bleu.json  --task translation $@
    run_generate(verbose=True)
