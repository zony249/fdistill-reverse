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


from dist_utils import compute_pca, pca_transform, create_display_df, DistBetweenModels, DistBetweenLayers, DistPairwise


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
        self.dist_between_decoders = DistBetweenModels()
        self.dist_between_encoders = DistBetweenModels()
        self.dist_teacher_teacher_enc = DistPairwise()
        self.dist_teacher_teacher_dec = DistPairwise()
        self.dist_teacher_student_enc = DistPairwise()
        self.dist_teacher_student_dec = DistPairwise()
        self.dist_student_student_enc = DistPairwise()
        self.dist_student_student_dec = DistPairwise()
        self.dist_between_teacher_enc = DistBetweenLayers()
        self.dist_between_teacher_dec = DistBetweenLayers()
        self.dist_between_student_enc = DistBetweenLayers()
        self.dist_between_student_dec = DistBetweenLayers()
        self.config1 = config1 
        self.config2 = config2
        self.reverse_enc = reverse_encoder 
        self.reverse_dec = reverse_decoder
    def compute_accum(self, enc_hidden_1, dec_hidden_1, enc_hidden_2=None, dec_hidden_2=None): 
        """
        hidden_1: torch.Tensor[batch, num_layers_1, hidden]
        hidden_2: torch.Tensor[batch, num_layers_2, hidden]
        
        """
        dec_indices = [0] + [i+1 for i in LAYER_MAP[self.config1.num_decoder_layers][self.config2.num_decoder_layers]] if self.config2 is not None else None
        enc_indices = [0] + [i+1 for i in LAYER_MAP[self.config1.num_layers][self.config2.num_layers]] if self.config2 is not None else None

        if self.reverse_enc: 
            enc_indices = enc_indices[::-1]
        if self.reverse_dec: 
            dec_indices = dec_indices[::-1]
        
        self.enc_indices = enc_indices 
        self.dec_indices = dec_indices

        self.dist_teacher_teacher_enc(enc_hidden_1, enc_hidden_1).accum()
        self.dist_teacher_teacher_dec(dec_hidden_1, dec_hidden_1).accum()
        self.dist_between_teacher_enc(enc_hidden_1).accum()
        self.dist_between_teacher_dec(dec_hidden_1).accum()

        if self.config2 is not None:
            self.dist_between_encoders(enc_hidden_1[:, enc_indices], enc_hidden_2).accum()
            self.dist_between_decoders(dec_hidden_1[:, dec_indices], dec_hidden_2).accum()
            self.dist_student_student_enc(enc_hidden_2, enc_hidden_2).accum()
            self.dist_student_student_dec(dec_hidden_2, dec_hidden_2).accum()
            self.dist_teacher_student_enc(enc_hidden_1, enc_hidden_2).accum()
            self.dist_teacher_student_dec(dec_hidden_1, dec_hidden_2).accum()
            self.dist_between_student_enc(enc_hidden_2).accum()
            self.dist_between_student_dec(dec_hidden_2).accum()

    def save_figs(self, dirname): 
        teacher_teacher_enc = self.dist_teacher_teacher_enc.get_mean()
        teacher_teacher_dec = self.dist_teacher_teacher_dec.get_mean()
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

        teacher_between_enc_dist = self.dist_between_teacher_enc.get_mean() 
        teacher_between_dec_dist = self.dist_between_teacher_dec.get_mean() 
        fig, ax = plt.subplots(ncols=2, figsize=(10, 4), gridspec_kw={'wspace': 0.5})
        x_enc = torch.arange(self.config1.num_layers)
        x_dec = torch.arange(self.config1.num_decoder_layers)
        width = 0.35 
        shift = width/ 2 if self.config2 is not None else 0
        im1 = ax[0].bar(x_enc + 1 - shift, teacher_between_enc_dist, width, label="Teacher")
        im2 = ax[1].bar(x_dec + 1 - shift, teacher_between_dec_dist, width, label="Teacher")
        ax[0].set_title("Encoder")
        ax[0].set_xlabel("Layers")
        ax[0].set_ylabel("L2 Norm")
        ax[1].set_title("Decoder")
        ax[1].set_xlabel("Layers")
        ax[1].set_ylabel("L2 Norm")

        if self.config2 is not None: 
            student_between_enc_dist = self.dist_between_student_enc.get_mean()
            student_between_dec_dist = self.dist_between_student_dec.get_mean()
            ax[0].bar(torch.tensor([i+1 for i in LAYER_MAP[self.config1.num_layers][self.config2.num_layers]]) + shift, student_between_enc_dist, width, label= "Student") 
            ax[1].bar(torch.tensor([i+1 for i in LAYER_MAP[self.config1.num_decoder_layers][self.config2.num_decoder_layers]]) + shift, student_between_dec_dist, width, label= "Student") 
            plt.savefig(os.path.join(dirname, "between-layer-distances"))
            plt.close()
            

            teacher_student_encoder_dist = self.dist_between_encoders.get_mean() 
            teacher_student_decoder_dist = self.dist_between_decoders.get_mean()

            x = torch.arange(max(len(teacher_student_encoder_dist), len(teacher_student_decoder_dist)))
            width = 0.35
            fig, ax = plt.subplots(ncols=2, figsize=(10, 4)) 
            ax[0].set_title("Encoder matching distances")
            ax[0].set_ylabel("L2 norm")
            ax[0].set_xlabel("Teacher layer matched to student")
            ax[0].bar(x , teacher_student_encoder_dist, width, label="Encoder") 
            ax[0].set_xticks(x)
            ax[0].set_xticklabels(self.enc_indices)

            ax[1].set_title("Decoder matching distances")
            ax[1].set_ylabel("L2 norm")
            ax[1].set_xlabel("Teacher layer matched to student")
            ax[1].bar(x , teacher_student_decoder_dist, width, label="Decoder") 
            ax[1].set_xticks(x)
            ax[1].set_xticklabels(self.dec_indices)

            
            plt.savefig(os.path.join(dirname, "teacher-student-matching-dist"))
            plt.close() 


            teacher_student_enc = self.dist_teacher_student_enc.get_mean()
            teacher_student_dec = self.dist_teacher_student_dec.get_mean()
            fig, ax = plt.subplots(ncols=2, figsize=(10, 4), gridspec_kw={'wspace': 0.5})
            im1 = ax[0].imshow(teacher_student_enc)
            im2 = ax[1].imshow(teacher_student_dec)
            ax[0].set_title("Encoder")
            ax[0].set_xlabel("Student Layer j")
            ax[0].set_ylabel("Teacher Layer i")
            ax[1].set_title("Decoder")
            ax[1].set_xlabel("Student Layer j")
            ax[1].set_ylabel("Teacher Layer i")

            fig.colorbar(im1, ax=ax[0])
            fig.colorbar(im2, ax=ax[1])

            plt.savefig(os.path.join(dirname, "teacher-student-pairwise-dist"))
            plt.close()

            student_student_enc = self.dist_student_student_enc.get_mean()
            student_student_dec = self.dist_student_student_dec.get_mean()
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

            # compute the average distance between each student layer and all teacher layers
            student_teacher_avg_dist_per_layer_enc = teacher_student_enc.mean(dim=0) # torch.Tensor[num_student_layers]
            student_teacher_avg_dist_enc = student_teacher_avg_dist_per_layer_enc.mean()
            student_teacher_avg_dist_per_layer_dec = teacher_student_dec.mean(dim=0) # torch.Tensor[num_student_layers]
            student_teacher_avg_dist_dec = student_teacher_avg_dist_per_layer_dec.mean()
            teacher_avg_enc_dist = teacher_between_enc_dist.mean()
            teacher_avg_dec_dist = teacher_between_dec_dist.mean()
            with open(os.path.join(dirname, "teacher-student-stats.tsv"), "w") as f:
                f.write(f"student_teacher_dist_per_layer_enc\t{student_teacher_avg_dist_per_layer_enc.tolist()}\n")
                f.write(f"student_teacher_avg_dist_enc\t{student_teacher_avg_dist_enc.item()}\n")
                f.write(f"student_teacher_dist_per_layer_dec\t{student_teacher_avg_dist_per_layer_dec.tolist()}\n")
                f.write(f"student_teacher_avg_dist_dec\t{student_teacher_avg_dist_dec.item()}\n")
                f.write(f"teacher_between_enc_dist\t{teacher_between_enc_dist.tolist()}\n")
                f.write(f"teacher_avg_between_enc_dist\t{teacher_avg_enc_dist.item()}\n")
                f.write(f"teacher_between_dec_dist\t{teacher_between_dec_dist.tolist()}\n")
                f.write(f"teacher_avg_between_dec_dist\t{teacher_avg_dec_dist.item()}\n")
                f.write(f"||teacher - student|| / ||teacher_encoder_between_layer||\t{(student_teacher_avg_dist_enc / teacher_avg_enc_dist).item()}\n")



        else: 
            plt.savefig(os.path.join(dirname, "between-layer-distances"))
            plt.close()

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

    encoder_hidden = []
    decoder_hidden = []
    compare_encoder_hidden = []
    compare_decoder_hidden = []

    dist_between_decoders = DistBetweenModels()
    dist_between_encoders = DistBetweenModels()
    dist_teacher_teacher_enc = DistPairwise()
    dist_teacher_teacher_dec = DistPairwise()
    dist_teacher_student_enc = DistPairwise()
    dist_teacher_student_dec = DistPairwise()
    dist_student_student_enc = DistPairwise()
    dist_student_student_dec = DistPairwise()

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
        else:
            compare_outputs = None

        if unnormalized: 
            break
        
        #computing the encoder hidden states
        encoder_hidden.append(torch.stack([(e.sum(dim=1)/batch.attention_mask.sum(dim=1, keepdims=True)).detach().cpu() for e in outputs.encoder_hidden_states], dim=1))
        #computing the decoder hidden states is a bit weird, since its tokenwise via autoregressive generation
        timesteps = []
        for tokens in outputs.decoder_hidden_states: 
            layers = []
            for layer in tokens: 
                # This is [batch_size, 1, hidden]
                hidden_state_normed = torch.cat([layer[i*num_beams:(i+1)*num_beams].mean(dim=0, keepdims=True).detach().cpu() for i in range(math.floor(layer.shape[0]/num_beams))], dim=0)
                layers.append(hidden_state_normed)
            # [batch_size, layers, hidden]
            layers = torch.cat(layers, dim=1)
            timesteps.append(layers)
        #
        decoder_hidden_T = torch.stack(timesteps, dim=2)
        decoder_hidden_T = decoder_hidden_T.mean(dim=2)
        decoder_hidden.append(decoder_hidden_T)

        if compare_model is not None: 
            # copy-paste of above code for student
            compare_encoder_hidden.append(torch.stack([(e.sum(dim=1)/batch.attention_mask.sum(dim=1, keepdims=True)).detach().cpu() for e in compare_outputs.encoder_hidden_states], dim=1))

            timesteps = []
            for tokens in compare_outputs.decoder_hidden_states: 
                layers = []
                for layer in tokens: 
                    # This is [batch_size, 1, hidden]
                    hidden_state_normed = torch.cat([layer[i*num_beams:(i+1)*num_beams].mean(dim=0, keepdims=True).detach().cpu() for i in range(math.floor(layer.shape[0]/num_beams))], dim=0)
                    layers.append(hidden_state_normed)
                # [batch_size, layers, hidden]
                layers = torch.cat(layers, dim=1)
                timesteps.append(layers)
            #
            decoder_hidden_S = torch.stack(timesteps, dim=2)
            decoder_hidden_S = decoder_hidden_S.mean(dim=2)
            compare_decoder_hidden.append(decoder_hidden_S)

            metrics.compute_accum(encoder_hidden[-1], decoder_hidden[-1], compare_encoder_hidden[-1], compare_decoder_hidden[-1])
            metrics.save_figs("/".join(out_file.split("/")[:-1]))
        else: 
            metrics.compute_accum(encoder_hidden[-1], decoder_hidden[-1]) 
            metrics.save_figs("/".join(out_file.split("/")[:-1]))





        dec = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for hypothesis in dec:
            fout.write(hypothesis + "\n")
            fout.flush()

    # encoder_hidden = [e.mean(dim=0, keepdims=True) for e in encoder_hidden]
    # decoder_hidden = [d.mean(dim=0, keepdims=True) for d in decoder_hidden]


    if unnormalized: 
        encoder_hidden = [torch.stack([e.reshape(-1, e.shape[-1]).detach().cpu() for e in outputs.encoder_hidden_states], dim=1)]
        timesteps = []
        for tokens in outputs.decoder_hidden_states: 
            layers = []
            for layer in tokens: 
                # This is [batch_size, 1, hidden]
                hidden_state_normed = torch.cat([layer[i*num_beams:(i+1)*num_beams].mean(dim=0, keepdims=True).detach().cpu() for i in range(math.floor(layer.shape[0]/num_beams))], dim=0)
                layers.append(hidden_state_normed)

            # [batch_size, layers, hidden]
            layers = torch.cat(layers, dim=1)
            timesteps.append(layers)

        decoder_hidden_T = torch.stack(timesteps, dim=1)
        decoder_hidden = [decoder_hidden_T.reshape(-1, layers.shape[1], layers.shape[-1])]
        if compare_model is not None:
            compare_encoder_hidden.append(torch.stack([(e.sum(dim=1)/batch.attention_mask.sum(dim=1, keepdims=True)).detach().cpu() for e in compare_outputs.encoder_hidden_states], dim=1))
            # outputs.decoder_hidden_states are Tuple[Tuple[Tensor[batch * num_beams, 1, hidden]]]
            # outer dimension is timesteps 
            # inner dimension is layers 
            timesteps = []
            for tokens in compare_outputs.decoder_hidden_states: 
                layers = []
                for layer in tokens: 
                    # This is [batch_size, 1, hidden]
                    hidden_state_normed = torch.cat([layer[i*num_beams:(i+1)*num_beams].mean(dim=0, keepdims=True).detach().cpu() for i in range(math.floor(layer.shape[0]/num_beams))], dim=0)
                    layers.append(hidden_state_normed)
                # [batch_size, layers, hidden]
                layers = torch.cat(layers, dim=1)
                timesteps.append(layers)
            #
            decoder_hidden_T = torch.stack(timesteps, dim=2)
            decoder_hidden_T = decoder_hidden_T.mean(dim=2)
            compare_decoder_hidden.append(decoder_hidden_T)



    if pca_mode == "encoder_only":
        decoder_hidden = []
        compare_decoder_hidden = []
    elif pca_mode == "decoder_only": 
        encoder_hidden = []
        compare_encoder_hidden = []

    if len(encoder_hidden) > 0:
        print("Num encoder samples for PCA computation: ",  torch.cat([e.view(-1, e.shape[-1]) for e in encoder_hidden], dim=0).shape)
    if len(decoder_hidden) > 0:
        print("Num decoder samples for PCA computation: ", torch.cat([d.view(-1, d.shape[-1]) for d in decoder_hidden], dim=0).shape)

    if len(compare_encoder_hidden) > 0:
        print("Num encoder samples for PCA computation (second model): ",  torch.cat([e.view(-1, e.shape[-1]) for e in encoder_hidden], dim=0).shape)
    if len(compare_decoder_hidden) > 0:
        print("Num decoder samples for PCA computation (second model): ", torch.cat([d.view(-1, d.shape[-1]) for d in decoder_hidden], dim=0).shape)


    U_red = compute_pca([e.view(-1, e.shape[-1]) for e in encoder_hidden] + 
                        [d.view(-1, d.shape[-1]) for d in decoder_hidden] + 
                        [e.view(-1, e.shape[-1]) for e in compare_encoder_hidden] + 
                        [d.view(-1, d.shape[-1]) for d in compare_decoder_hidden]) 
    print(U_red.shape)


    enc_batch_concat = torch.cat(encoder_hidden, dim=0) if len(encoder_hidden) > 0 else None
    dec_batch_concat = torch.cat(decoder_hidden, dim=0) if len(decoder_hidden) > 0 else None
    compare_enc_batch_concat = torch.cat(compare_encoder_hidden, dim=0) if len(compare_encoder_hidden) > 0 else None
    compare_dec_batch_concat = torch.cat(compare_decoder_hidden, dim=0) if len(compare_decoder_hidden) > 0 else None
    
    if display_centroids: 
        enc_batch_concat = enc_batch_concat.mean(dim=0, keepdims=True) if enc_batch_concat is not None else None 
        dec_batch_concat = dec_batch_concat.mean(dim=0, keepdims=True) if dec_batch_concat is not None else None 
        compare_enc_batch_concat = compare_enc_batch_concat.mean(dim=0, keepdims=True) if compare_enc_batch_concat is not None else None 
        compare_dec_batch_concat = compare_dec_batch_concat.mean(dim=0, keepdims=True) if compare_dec_batch_concat is not None else None 

    enc_pca = pca_transform(enc_batch_concat, U_red) if enc_batch_concat is not None else None
    dec_pca = pca_transform(dec_batch_concat, U_red) if dec_batch_concat is not None else None
    compare_enc_pca = pca_transform(compare_enc_batch_concat, U_red) if compare_enc_batch_concat is not None else None
    compare_dec_pca = pca_transform(compare_dec_batch_concat, U_red) if compare_dec_batch_concat is not None else None

    print(enc_pca.shape if enc_pca is not None else "")
    print(dec_pca.shape if dec_pca is not None else "")
    print(compare_enc_pca.shape if compare_enc_pca is not None else "")
    print(compare_dec_pca.shape if compare_dec_pca is not None else "")

    df1 = create_display_df(enc_pca, dec_pca, "/".join(out_file.split("/")[:-1]))
    df1["student model"] = 0.0

    if compare_model is not None:
        df2 = create_display_df(compare_enc_pca, compare_dec_pca, "") 
        df2["student model"] = 1.0
        print(df2)
        df1.reset_index(drop=True, inplace=True)
        df2.reset_index(drop=True, inplace=True)
        df1 = pd.concat([df1, df2])

    print(df1)

    fig = px.scatter_3d(df1, x="x", y="y", z="z", color="layer", symbol="student model", hover_data="is_decoder")
    fig.show()

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
