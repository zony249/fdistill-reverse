import os
from os import path 
from argparse import ArgumentParser

import pandas as pd 
import numpy as np


def cola(df):
    df["prediction"] = (df["prediction"] == "acceptable").astype(int)
    return df[["index", "prediction"]]

def mrpc(df):
    df["prediction"] = (df["prediction"] == "equivalent").astype(int)
    return df[["index", "prediction"]]

def qnli(df): 
    return df[["index", "prediction"]]

def mnli(df):
    return df[["index", "prediction"]]

def qqp(df): 
    df["prediction"] = (df["prediction"] == "duplicate").astype(int)
    return df[["index", "prediction"]]

def sst2(df):
    df["prediction"] = (df["prediction"] == "positive").astype(int)
    return df[["index", "prediction"]]

def stsb(df):
    df[["prediction"]] = df[["prediction"]].clip(lower=0, upper=5)
    return df

def rte(df): 
    return df

taskmap = {
    "CoLA": cola,
    "MRPC": mrpc, 
    "QNLI": qnli, 
    "MNLI-m": mnli,
    "MNLI-mm": mnli, 
    "QQP": qqp, 
    "RTE" : rte, 
    "SST-2": sst2, 
    "STS-B": stsb,
}


if __name__ == "__main__":
    parser = ArgumentParser("HF to GLUE converter")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    
    args = parser.parse_args() 

    df = pd.read_csv(args.input_file, sep="\t")
    df = taskmap[args.task](df)

    output_file = os.path.join("/".join(args.input_file.split("/")[:-1]), args.task + ".tsv")

    df.to_csv(output_file, sep="\t", index=False)

    print(df)