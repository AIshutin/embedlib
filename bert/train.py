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
import losses
from utils import embed_batch, prepare_batch
from utils import load_model, save_model
from datasets import UbuntuCorpus, TwittCorpus

from sacred import Experiment
from sacred.observers import MongoObserver, TelegramObserver

import logging
import os

ex = Experiment()
#ex.observers.append(TelegramObserver.from_config("./aishutin-telegramobserver-config.json"))

writer = SummaryWriter()

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
	max_dataset_size = int(1e4)

	# BERT config
	max_seq_len = 512
	bert_type = 'bert-base-uncased'
	cache_dir = '../pretrained-' + bert_type + '/'

@ex.capture
def get_data(_log, data_path, tokenizer, test_split, max_seq_len, batch_size, max_dataset_size):
	corpus = TwittCorpus(tokenizer, '../corp.txt', max_seq_len, max_dataset_size)
	_log.info(f'Corpus length: {len(corpus)}')
	test_size = int(len(corpus) * test_split)
	train_size = len(corpus) - test_size
	train_data, test_data = torch.utils.data.random_split(corpus, [train_size, test_size])
	return DataLoader(train_data, batch_size=batch_size, shuffle=True), \
		   DataLoader(test_data, batch_size=batch_size, shuffle=True)

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
		metric_baseline_func, criterion_func, float_mode,):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	tokenizer, (qembedder, aembedder) = get_model()
	qembedder.to(device)
	aembedder.to(device)

	train, test = get_data(_log, '.', tokenizer)

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

	score_before_finetuning = metrics.get_mean_score_on_data(metric, test, \
															(qembedder, aembedder), \
															tokenizer, float_mode)

	_log.info("***** Running training *****")
	_log.info("  Num steps = %d", num_train_optimization_steps)
	_log.info(f"Score before fine-tuning: {score_before_finetuning:9.4f}")
	_log.info(f'Random choice score: {metric_baseline(batch_size):9.4f}')

	step = 0
	for epoch in range(epochs):
		total_loss = 0
		qembedder.train()
		aembedder.train()

		for bidx, batch in enumerate(tqdm.tqdm(iter(train), desc=f"epoch {epoch}")):
			qoptim.zero_grad()
			aoptim.zero_grad()

			embeddings = embed_batch(prepare_batch(batch, device, tokenizer), qembedder, aembedder, float_mode)
			loss = criterion(*embeddings)

			total_loss += loss.item()
			writer.add_scalar('data/loss_dynamic', loss.item(), step)
			step += 1
			loss.backward()

			qscheduler.step()
			ascheduler.step()
			qoptim.step()
			aoptim.step()

		# ToDo do something with score
		score = metrics.get_mean_score_on_data(metric, test, \
											(qembedder, aembedder), \
											tokenizer, float_mode)

		writer.add_scalar('data/score', score, epoch)
		writer.add_scalar('data/total_loss', total_loss, epoch)
		_log.info(f'score:{score:9.4f} | loss:{total_loss:9.4f} ')
		checkpoint_name = checkpoint_dir + f"epoch:{epoch:2d} {metric_func}:{score:9.4f} {criterion_func}:{total_loss:9.4f}/"
		save_model((qembedder, aembedder), tokenizer, checkpoint_name)

	_log.info('Fine-tuning is compleated')
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	# ToDo save model

ex.run()
writer.close()