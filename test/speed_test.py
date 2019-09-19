from embedlib.utils import load_model, print_batch, mem_report
from embedlib.datasets import CorpusData, collate_wrapper
from torch.utils.data import Dataset, DataLoader, Subset
from embedlib.metrics import calc_mrr, calc_placeholder_metric
from embedlib import metrics
import sys
import torch
import time
import gc

max_dataset_size = int(1e3)
batch_size = 1

def get_model_perfomance(embedder, dataset_name=None):
    lang = embedder.model.lang

    if dataset_name is None:
        dataset_name = 'ru-toloka' if lang == 'ru' else 'en-twitt-corpus'

    full_dataset = CorpusData([dataset_name], embedder.model.tokenizer, max_dataset_size)
    gc.collect()
    start = time.clock()
    for i in range(len(full_dataset)):
        question = full_dataset[i][0].replace('[CLS] ', '')
        embedder(question)
    return len(full_dataset) / (time.clock() - start)
