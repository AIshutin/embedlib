import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

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

from embedlib import models

from sacred import Experiment
from sacred.observers import MongoObserver, TelegramObserver

import logging
import os
import time
import gc

ex = Experiment()

# omniboard --mu "mongodb+srv://cerebra-autofaq:8jxIlp0znlJm8qhL@testing-pjmjc.gcp.mongodb.net/experiments?retryWrites=true&w=majority&authMechanism=SCRAM-SHA-1"
observer = MongoObserver.create(url='mongodb+srv://cerebra-autofaq:8jxIlp0znlJm8qhL@testing-pjmjc.gcp.mongodb.net/test?retryWrites=true&w=majority&authMechanism=SCRAM-SHA-1', \
                                db_name='experiments', \
                                port=27017)
ex.observers.append(observer)

@ex.config
def config():
    test_split = 0.2
    checkpoint_dir = 'checkpoints/'
    learning_rate = 5e-5  # 5e-5, 3e-5, 2e-5 are recommended in the paper
    epochs = 3
    warmup = 0.1

    seed = 0
    metric_name = 'mrr'
    metric_func = 'calc_' + metric_name
    metric_baseline_func = 'calc_random_' + metric_name
    criterion_func = 'hinge_loss'
    batch_size = 16  # 16, 32 are recommended in the paper
    test_batch_size = 16 # batch_size
    statistic_accumalation = 100

    continue_training_from_checkpoint = None
    model_name = 'RoBERTaLike'
    model_config = None
    has_scheduler = True

    if model_name == 'BERTLike':
        model_config = {'bert_type': 'bert-base-uncased',
                    'lang': 'en', 'float_mode': 'fp32'}
    elif model_name == 'GPT2Like':
        model_config = {'lang': 'en'}
    elif model_name == 'RoBERTaLike':
        model_config = {'lang': 'ru'}
    else:
        raise Exception('model is not defined')
    dataset_names = ['en-twitt-corpus' if model_config['lang'] == 'en' else 'ru-opendialog-corpus']
    max_dataset_size = int(1e5)
    multigpu = True
    max_grad_norm = 1.0

@ex.capture
def get_data(_log, data_path, tokenizer, test_split, max_seq_len, batch_size, max_dataset_size, \
            dataset_names, test_batch_size, multigpu):
    if multigpu:
        import horovod.torch as hvd
    corpus = datasets.CorpusData(dataset_names, tokenizer, max_dataset_size)
    _log.info("Corpus size: " + str(len(corpus)))
    test_size = int(len(corpus) * test_split)
    train_size = len(corpus) - test_size
    train_data, test_data = torch.utils.data.random_split(corpus, [train_size, test_size])

    if multigpu:
        _log.info('MultiDataset')
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_data, num_replicas=hvd.size(), rank=hvd.rank())
        test_sampler = torch.utils.data.distributed.DistributedSampler(
                    test_data, num_replicas=hvd.size(), rank=hvd.rank())
        shuffle = False
    else:
        train_sampler = None
        test_sampler = None
        shuffle = True

    return DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_wrapper, sampler=train_sampler), \
           DataLoader(test_data, batch_size=test_batch_size, shuffle=shuffle, collate_fn=collate_wrapper, sampler=test_sampler)

@ex.capture
def get_metric(metric_func):
    metric = getattr(embedlib.metrics, metric_func)
    return metric

@ex.capture
def get_criterion(criterion_func):
    criterion = getattr(embedlib.losses, criterion_func)
    return criterion

@ex.capture
def get_model(model_name, model_config, continue_training_from_checkpoint, multigpu):
    if continue_training_from_checkpoint is not None:
        return embedlib.utils.load_model(continue_training_from_checkpoint)
    return getattr(embedlib.models, model_name)(**model_config)

@ex.capture
def get_model_optimizer(model):
    return AdamW

@ex.automain
def train(_log, epochs, batch_size, learning_rate, warmup, checkpoint_dir, metric_func, \
        metric_baseline_func, criterion_func, metric_name, statistic_accumalation, \
        test_batch_size, seed, has_scheduler, multigpu, max_grad_norm):
    if checkpoint_dir[-1] != '/':
        checkpoint_dir = checkpoint_dir + '/'

    torch.manual_seed(seed)

    if multigpu:
        # Initialize Horovod
        import horovod.torch as hvd
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
        device = torch.cuda.current_device()
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

    if multigpu:
        optimizer = get_model_optimizer(model)(model.parameters(), lr=learning_rate / hvd.size())
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    else:
        optimizer = get_model_optimizer(model)(model.parameters(), lr=learning_rate)
    if has_scheduler:
        scheduler =  get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, \
                                    num_training_steps=num_train_optimization_steps)

    val_score_before, val_loss_before = metrics.get_mean_on_data([metric, criterion], \
                                                                test, model)
    val_loss_before = val_loss_before.item()
    _log.info("***** Running training *****")
    _log.info(f"  Num steps = {num_train_optimization_steps}")
    _log.info(f"Score before fruine-tuning: {val_score_before:9.4f}")
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
                step += 1
                curr_loss = curr_score = 0

            batch_num += 1

            total_loss += loss.item()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if has_scheduler:
                scheduler.step()

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
