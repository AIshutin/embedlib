import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

import numpy as np
import pickle
import tqdm

import embedlib
from embedlib.metrics import get_mean_on_data

from embedlib.utils import mem_report, load_model
from embedlib import losses, metrics

from embedlib.datasets import collate_wrapper
from embedlib import datasets

from embedlib import models, optimizers

from sacred import Experiment
from sacred.observers import MongoObserver, TelegramObserver

import logging
import os
import time
import gc

ex = Experiment()

password = '8jxIlp0znlJm8qhL' # +srv
# omniboard --mu "mongodb+srv://cerebra-autofaq:8jxIlp0znlJm8qhL@testing-pjmjc.gcp.mongodb.net/experiments?retryWrites=true&w=majority&authMechanism=SCRAM-SHA-1"
observer = MongoObserver.create(url=f'mongodb+srv://cerebra-autofaq:{password}@testing-pjmjc.gcp.mongodb.net/test?retryWrites=true&w=majority&authMechanism=SCRAM-SHA-1',
                                db_name='experiments',
                                port=27017)
ex.observers.append(observer)

# ex.observers.append(TelegramObserver.from_config("./aishutin-telegramobserver-config.json"))

@ex.config
def config():
    test_split = 0.2
    checkpoint_dir = 'checkpoints/'
    learning_rate = 5e-5  # 5e-5, 3e-5, 2e-5 are recommended in the paper
    epochs = 20
    warmup = 0.1

    seed = 0
    metric_name = 'mrr'
    metric_func = f'calc_{metric_name}'
    metric_baseline_func = f'calc_random_{metric_name}'
    criterion_func = 'hinge_loss'
    batch_size = 128  # 16, 32 are recommended in the paper
    test_batch_size = 16 # batch_size
    statistic_accumalation = 100

    continue_training_from_checkpoint = None
    #f'{checkpoint_dir}epoch: 9 calc_mrr:   0.7970 hinge_loss:   0.1879'
    # None if not

    model_name = 'LASERembedder'
    model_config = None
    if model_name is 'BERTLike':
        model_config = {'bert_type': '-6-attentions',
                    'lang': 'ru', 'float_mode': 'fp32'}
    elif model_name is 'USEncoder':
        model_config = {'float_mode': 'fp32', 'lang': 'ru'}
    elif model_name is 'LASERembedder' or 'LASERtransformer_embedder':
        model_config = {'lang': 'en', 'lay_num': 22}
    else:
        raise Exception('model is not defined')

    dataset_names = ['en-twitt-corpus' if model_config['lang'] == 'en' else 'ru-opendialog-corpus']
    max_dataset_size = int(1e2)

@ex.capture
def get_data(_log, data_path, tokenizer, test_split, max_seq_len, batch_size, max_dataset_size, \
            dataset_names, test_batch_size):
    corpus = datasets.CorpusData(dataset_names, tokenizer, max_dataset_size)
    _log.info(f"Corpus size: {len(corpus)}")
    test_size = int(len(corpus) * test_split)
    train_size = len(corpus) - test_size
    train_data, test_data = torch.utils.data.random_split(corpus, [train_size, test_size])
    return DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_wrapper), \
           DataLoader(test_data, batch_size=test_batch_size, shuffle=True, collate_fn=collate_wrapper)

@ex.capture
def get_metric(metric_func):
    metric = getattr(embedlib.metrics, metric_func)
    return metric

@ex.capture
def get_criterion(criterion_func):
    criterion = getattr(embedlib.losses, criterion_func)
    return criterion

@ex.capture
def get_model(model_name, model_config, continue_training_from_checkpoint):
    if continue_training_from_checkpoint is not None:
        return embedlib.utils.load_model(continue_training_from_checkpoint)
    return getattr(embedlib.models, model_name)(**model_config)

@ex.capture
def get_model_optimizer(model):
    return getattr(embedlib.optimizers, f'{model.__name__}Optimizer')

@ex.automain
def train(_log, epochs, batch_size, learning_rate, warmup, checkpoint_dir, metric_func, \
        metric_baseline_func, criterion_func, metric_name, statistic_accumalation, \
        test_batch_size, seed):

    torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 'cpu'
    writer = SummaryWriter()
    model = get_model()
    model.to(device)
    gc.collect()

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
    val_loss_before = val_loss_before.item()
    _log.info("***** Running training *****")
    _log.info(f"  Num steps = {num_train_optimization_steps}")
    _log.info(f"Score before fine-tuning: {val_score_before:9.4f}")
    _log.info(f"Loss before fine-tuning: {val_loss_before:9.4f}")
    _log.info(f"Random choice score: {metric_baseline(test_batch_size):9.4f}")
    writer.add_scalar("val/score", val_score_before, 0)
    writer.add_scalar("val/loss", val_loss_before, 0)
    ex.log_scalar("val.score", val_score_before)
    ex.log_scalar("val.loss", val_loss_before)

    step = 0
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        total_train_score = 0
        total_mrr = 0
        batch_num = 0
        curr_loss = 0
        curr_score = 0
        for bidx, batch in enumerate(tqdm.tqdm(iter(train), desc=f"epoch {epoch}")):
            optimizer.zero_grad()

            embeddings = model(batch)
            loss = criterion(*embeddings)
            score = metric(*embeddings)
            total_train_score += score
            curr_loss += loss.item()
            curr_score += score

            if (bidx + 1) % statistic_accumalation == 0:
                mean_loss = curr_loss / statistic_accumalation
                mean_score = curr_score / statistic_accumalation
                writer.add_scalar("train/loss_dynamic", mean_loss, step)
                writer.add_scalar("train/score_dynamic", mean_score, step)
                ex.log_scalar("train.loss_dynamic", mean_loss)
                ex.log_scalar("train.score_dynamic", mean_score)
                #print(mean_loss, mean_score, step)
                step += 1
                curr_loss = curr_score = 0

            batch_num += 1

            total_loss += loss.item()
            loss.backward()

            optimizer.step()

        mean_train_score = total_train_score / batch_num

        # ToDo do something with score
        val_score, val_loss = metrics.get_mean_on_data([metric, criterion], test, \
                                                        model)
        val_loss = val_loss.item()
        writer.add_scalar("val/score", val_score, epoch + 1)
        writer.add_scalar("val/loss", val_loss, epoch + 1)
        writer.add_scalar("train/total_loss", total_loss, epoch)
        writer.add_scalar("train/total_score", mean_train_score, epoch)

        ex.log_scalar("val.score", val_score)
        ex.log_scalar("val.loss", val_loss)
        ex.log_scalar("train.total_loss", total_loss)
        ex.log_scalar("train.total_score", mean_train_score)

        _log.info(f"score:{val_score:9.4f} | loss:{total_loss:9.4f}")
        checkpoint_name = checkpoint_dir + f"epoch:{epoch:2d} {metric_func}:{val_score:9.4f} {criterion_func}:{total_loss:9.4f}/"
        model.save_to(checkpoint_name)

    val_score_before, val_loss_before = metrics.get_mean_on_data([metric, criterion], \
                                            test, load_model(checkpoint_name).to(device))
