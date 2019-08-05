import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pickle
import tqdm

import losses
import metrics
from metrics import get_mean_on_data

from utils import load_model

import datasets
from datasets import collate_wrapper

import models
import optimizers

from sacred import Experiment
from sacred.observers import MongoObserver, TelegramObserver

import logging
import os

ex = Experiment()
# ex.observers.append(TelegramObserver.from_config("./aishutin-telegramobserver-config.json"))

@ex.config
def config():
    test_split = 0.2
    checkpoint_dir = 'checkpoints/'
    learning_rate = 5e-5  # 5e-5, 3e-5, 2e-5 are recommended in the paper
    epochs = 3
    warmup = 0.1

    metric_name = 'mrr'
    metric_func = f'calc_{metric_name}'
    metric_baseline_func = f'calc_random_{metric_name}'
    criterion_func = 'hinge_loss'
    batch_size = 32  # 16, 32 are recommended in the paper

    model_name = 'BERTLike'
    model_config = {'bert_type': 'bert-base-uncased', 'lang': 'ru', 'float_mode': 'fp16'}

    dataset_names = ['en-twitt-corpus' if model_config['lang'] == 'en' else 'ru-opendialog-corpus']
    max_dataset_size = int(1e3)

@ex.capture
def get_data(_log, data_path, tokenizer, test_split, max_seq_len, batch_size, max_dataset_size, \
            dataset_names):
    corpus = datasets.CorpusData(dataset_names, tokenizer, max_dataset_size)
    _log.info(f"Corpus size: {len(corpus)}")
    test_size = int(len(corpus) * test_split)
    train_size = len(corpus) - test_size
    train_data, test_data = torch.utils.data.random_split(corpus, [train_size, test_size])
    return DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_wrapper), \
           DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_wrapper)

@ex.capture
def get_metric(metric_func):
    metric = getattr(metrics, metric_func)
    return metric

@ex.capture
def get_criterion(criterion_func):
    criterion = getattr(losses, criterion_func)
    return criterion

@ex.capture
def get_model(model_name, model_config):
    model = getattr(models, model_name)(**model_config)

    return model

@ex.capture
def get_model_optimizer(model):
    return getattr(optimizers, f'{model.__name__}Optimizer')


@ex.automain
def train(_log, epochs, batch_size, learning_rate, w
        mean_train_score = total_train_score / batch_numarmup, checkpoint_dir, metric_func, \
        metric_baseline_func, criterion_func, metric_name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()
    model = get_model()
    model.to(device)

    train, test = get_data(_log, '.', model.tokenizer, max_seq_len=model.max_seq_len)
    metric = get_metric()
    metric_baseline = get_metric(metric_baseline_func)
    criterion = get_criterion()

    num_train_optimization_steps = len(train) * epochs

    num_warmup_steps = int(warmup * num_train_optimization_steps)

    optimizer = get_model_optimizer(model)(model, num_train_optimization_steps=num_train_optimization_steps,\
                                            num_warmup_steps=num_warmup_steps,\
                                            warmup=warmup,\
                                            learning_rate=learning_rate)

    val_score_before, val_loss_before = metrics.get_mean_on_data([metric, criterion], \
                                                                test, model)
    _log.info("***** Running training *****")
    _log.info(f"  Num steps = {num_train_optimization_steps}")
    _log.info(f"Score before fine-tuningfloat_mode: {val_score_before:9.4f}")
    _log.info(f"Loss before fine-tuning: {val_loss_before:9.4f}")
    _log.info(f"Random choice score: {metric_baseline(batch_size):9.4f}")
    writer.add_scalar("val/score", val_score_before, 0)
    writer.add_scalar("val/loss", val_loss_before, 0)

    step = 0
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        total_train_score = 0
        total_mrr = 0
        batch_num = 0

        for bidx, batch in enumerate(tqdm.tqdm(iter(train), desc=f"epoch {epoch}")):
            optimizer.step()

            embeddings = model(batch)
            loss = criterion(*embeddings)
            score = metric(*embeddings)
            total_train_score += score

            writer.add_scalar("train/loss_dynamic", loss.item(), step)
            writer.add_scalar("train/score_dynamic", score, step)
            step += 1
            batch_num += 1

            total_loss += loss.item()
            loss.backward()

            optimizer.step()

        mean_train_score = total_train_score / batch_num

        # ToDo do something with score
        val_score, val_loss = metrics.get_mean_on_data([metric, criterion], test, \
                                                        model)
        writer.add_scalar("val/score", val_score, epoch + 1)
        writer.add_scalar("val/loss", val_loss, epoch + 1)
        writer.add_scalar("train/total_loss", total_loss, epoch)
        writer.add_scalar("train/total_score", mean_train_score, epoch)

        _log.info(f"score:{val_score:9.4f} | loss:{total_loss:9.4f}")
        checkpoint_name = checkpoint_dir + f"epoch:{epoch:2d} {metric_func}:{val_score:9.4f} {criterion_func}:{total_loss:9.4f}/"
        model.save_to(checkpoint_name)

    val_score_before, val_loss_before = metrics.get_mean_on_data([metric, criterion], \
                                                        test, load_model(checkpoint_name).to(device))
