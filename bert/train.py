import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from pytorch_transformers import BertTokenizer, BertForQuestionAnswering
from pytorch_transformers import BertModel, BertConfig
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

import numpy as np
import pickle
import tqdm

import metrics
from metrics import get_mean_on_data
import losses
from utils import embed_batch, prepare_batch
from utils import load_model, save_model
from datasets import UbuntuCorpus, TwittCorpus, collate_wrapper

from sacred import Experiment
from sacred.observers import MongoObserver, TelegramObserver

import logging
import os

ex = Experiment()
#ex.observers.append(TelegramObserver.from_config("./aishutin-telegramobserver-config.json"))

was = False

@ex.config
def config():
    test_split = 0.2
    checkpoint_dir = "checkpoints/"
    learning_rate = 5e-5 # 5e-5, 3e-5, 2e-5 are recommended in the paper
    epochs = 3
    warmup = 0.1

    metric_name = 'mrr'
    metric_func = f'calc_{metric_name}'
    metric_baseline_func = f'calc_random_{metric_name}'
    criterion_func = 'hinge_loss'
    batch_size = 16 # 16, 32 are recommended in the paper

    float_mode = 'fp32'
    max_dataset_size = int(16)

    # BERT config
    max_seq_len = 512
    bert_type = 'bert-base-uncased'
    cache_dir = '../pretrained-' + bert_type + '/'

@ex.capture
def get_data(_log, data_path, tokenizer, test_split, max_seq_len, batch_size, max_dataset_size):
    corpus = TwittCorpus(tokenizer, '../corp.txt', max_seq_len, max_dataset_size)
    _log.info(f'Corpus size: {len(corpus)}')
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
def get_model(bert_type, cache_dir, float_mode):
    tokenizer = BertTokenizer.from_pretrained(bert_type, cache_dir=cache_dir)
    qembedder = BertModel.from_pretrained(bert_type, cache_dir=cache_dir)
    print(qembedder)
    aembedder = BertModel.from_pretrained(bert_type, cache_dir=cache_dir)
    qembedder.config.output_hidden_states = True
    aembedder.config.output_hidden_states = True
    if float_mode == 'fp16':
        qembedder.half()
        aembedder.half()

    model = (qembedder, aembedder)

    return tokenizer, model

@ex.automain
def train(_log, epochs, batch_size, learning_rate, warmup, checkpoint_dir, metric_func, \
        metric_baseline_func, criterion_func, float_mode, metric_name):
    global was

    if was:
        return
    else:
        was = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()
    tokenizer, (qembedder, aembedder) = get_model()
    qembedder.to(device)
    aembedder.to(device)

    train, test = get_data(_log, '.', tokenizer)
    writer.add_text('config', f'train dataset size: {len(train)*batch_size}', 0)
    writer.add_text('config', f'batch_size: {batch_size}', 1)
    writer.add_text('config', f'learning_rate: {learning_rate}', 2)
    writer.add_text('config', f'metic: {metric_name}', 3)
    writer.add_text('config', f'loss: {criterion_func}', 4)
    writer.add_text('config', f'float_mode: {float_mode}', 5)
    writer.add_text('config', f'epochs: {epochs}', 6)
    writer.add_text('config', f'warmup: {warmup}', 7)
    metric = get_metric()
    metric_baseline = get_metric(metric_baseline_func)
    criterion = get_criterion()

    num_train_optimization_steps = len(train) * epochs

    num_warmup_steps = int(warmup * num_train_optimization_steps)

    qoptim = AdamW(qembedder.parameters(), lr=learning_rate, correct_bias=False)
    qscheduler = WarmupLinearSchedule(qoptim, warmup_steps=num_warmup_steps, \
                                    t_total=num_train_optimization_steps)

    aoptim = AdamW(aembedder.parameters(), lr=learning_rate, correct_bias=False)
    ascheduler = WarmupLinearSchedule(aoptim, warmup_steps=num_warmup_steps, \
                                    t_total=num_train_optimization_steps)

    val_score_before, val_loss_before = metrics.get_mean_on_data([metric, criterion], test, \
                                                            (qembedder, aembedder), \
                                                            float_mode)

    _log.info("***** Running training *****")
    _log.info("  Num steps = %d", num_train_optimization_steps)
    _log.info(f"Score before fine-tuning: {val_score_before:9.4f}")
    _log.info(f"Loss before fine-tuning: {val_loss_before:9.4f}")
    _log.info(f'Random choice score: {metric_baseline(batch_size):9.4f}')
    writer.add_scalar('val/score', val_score_before, 0)
    writer.add_scalar('val/loss', val_loss_before, 0)

    step = 0
    for epoch in range(epochs):
        total_loss = 0
        qembedder.train()
        aembedder.train()
        total_train_score = 0
        total_mrr = 0
        batch_num = 0

        for bidx, batch in enumerate(tqdm.tqdm(iter(train), desc=f"epoch {epoch}")):
            qoptim.zero_grad()
            aoptim.zero_grad()

            embeddings = embed_batch(prepare_batch(batch, device), qembedder, aembedder, float_mode)
            loss = criterion(*embeddings)
            score = metric(*embeddings)
            total_train_score += score

            writer.add_scalar('train/loss_dynamic', loss.item(), step)
            writer.add_scalar('train/score_dynamic', score, step)
            step += 1
            batch_num += 1

            total_loss += loss.item()
            loss.backward()

            qscheduler.step()
            ascheduler.step()
            qoptim.step()
            aoptim.step()

def train(_log, epochs, batch_size, learning_rate, warmup, checkpoint_dir, metric_func, \
        metric_baseline_func, criterion_func, float_mode, metric_name):
    global was

    if was:
        return

        mean_train_score = total_train_score / batch_num

        # ToDo do something with score
        val_score, val_loss = metrics.get_mean_on_data([metric, criterion], test, \
                                            (qembedder, aembedder), \
                                            float_mode)
        writer.add_scalar('val/score', val_score, epoch + 1)
        writer.add_scalar('val/loss', val_loss, epoch + 1)
        writer.add_scalar('train/total_loss', total_loss, epoch)
        writer.add_scalar('train/total_score', mean_train_score, epoch)

        _log.info(f'score:{val_score:9.4f} | loss:{total_loss:9.4f} ')
        checkpoint_name = checkpoint_dir + f"epoch:{epoch:2d} {metric_func}:{val_score:9.4f} {criterion_func}:{total_loss:9.4f}/"
        save_model((qembedder, aembedder), tokenizer, checkpoint_name)

    _log.info('Fine-tuning is compleated')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    writer.add_text('status', 'compleated', 0)
    writer.close()
    # ToDo save model

ex.run()
