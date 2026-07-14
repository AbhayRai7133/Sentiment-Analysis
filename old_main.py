import os
import time
import warnings

import numpy as np
import pandas as pd

import mlflow
import mlflow.transformers

from fastapi import FastAPI
from pydantic import BaseModel

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

import evaluate

warnings.filterwarnings("ignore")

MODEL_NAME = "distilbert-base-uncased"

REGISTERED_MODEL_NAME = "review-intelligence"

TRAIN_MODEL = True

NUM_LABELS = 2

EPOCHS = 2

LEARNING_RATE = 2e-5

BATCH_SIZE = 16

# ==========================
# Load IMDb Dataset
# ==========================

print("Loading IMDb dataset...")

dataset = load_dataset("imdb")

print(dataset)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example):

    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)

tokenized_dataset = tokenized_dataset.remove_columns(
    ["text"]
)

tokenized_dataset = tokenized_dataset.rename_column(
    "label",
    "labels"
)

tokenized_dataset.set_format(
    "torch"
)

train_dataset = tokenized_dataset["train"]

test_dataset = tokenized_dataset["test"]