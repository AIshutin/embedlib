import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pytorch_pretrained_bert import BertTokenizer, BertForQuestionAnswering, BertModel
from pytorch_pretrained_bert.optimization import BertAdam

import numpy as np
import pickle
import tqdm

import metrics
import losses
from utils import get_optimizer_params, embed_batch, prepare_batch
from datasets import UbuntuCorpus

from sacred import Experiment
import logging

ex = Experiment()

@ex.config
def config():
	test_split = 0.2
	checkpoint_dir = "checkpoints"
	learning_rate = 5e-5 # 5e-5, 3e-5, 2e-5 are recommended in the paper
	epochs = 3
	warmup = 0.1

	metric_func = 'calc_mrr'
	criterion_func = 'hinge_loss'
	batch_size = 10 # 16, 32 are recommended in the paper

	max_dataset_size = int(1e5)

	# BERT config
	max_seq_len = 512
	bert_type = 'bert-base-uncased'
	cache_dir = '../pretrained-' + bert_type

@ex.capture
def get_data(_log, data_path, tokenizer, test_split, max_seq_len, batch_size, max_dataset_size):
	corpus = UbuntuCorpus(tokenizer, '../dialogs', max_seq_len, max_dataset_size)
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
def get_model(bert_type, cache_dir):
	tokenizer = BertTokenizer.from_pretrained(bert_type, cache_dir=cache_dir)
	qembedder = BertModel.from_pretrained(bert_type, cache_dir=cache_dir)
	aembedder = BertModel.from_pretrained(bert_type, cache_dir=cache_dir)
	model = (qembedder, aembedder)

	return tokenizer, model

@ex.capture
def save_model(model, checkpoint_dir):
	pass

@ex.automain
def train(_log, epochs, batch_size, learning_rate, warmup):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	tokenizer, (qembedder, aembedder) = get_model()
	qembedder.to(device)
	aembedder.to(device)

	train, test = get_data(_log, '.', tokenizer)

	metric = get_metric()
	criterion = get_criterion()

	num_train_optimization_steps = len(train) * epochs

	qoptim = BertAdam(get_optimizer_params(qembedder),
						lr=learning_rate,
						warmup=warmup,
						t_total=num_train_optimization_steps)

	aoptim = BertAdam(get_optimizer_params(aembedder),
						lr=learning_rate,
						warmup=warmup,
						t_toal=num_train_optimization_steps)

	score_before_finetuning = metrics.get_mean_score_on_data(metric, test, \
															(qembedder, aembedder), \
															tokenizer)

	_log.info("***** Running training *****")
	_log.info("  Num steps = %d", num_train_optimization_steps)
	_log.info(f"Score before fine-tuning: {score_before_finetuning:9.4f}")

	for epoch in range(epochs):
		total_loss = 0
		qembedder.train()
		aembedder.train()

		for bidx, batch in enumerate(tqdm.tqdm(iter(train), desc=f"epoch {epoch}")):
			qoptim.zero_grad()
			aoptim.zero_grad()

			embeddings = embed_batch(prepare_batch(batch, device, tokenizer), qembedder, aembedder)
			loss = criterion(*embeddings)

			total_loss += loss.item()
			loss.backward()

			qoptim.step()
			aoptim.step()

		# ToDo do something with score
		score = metrics.get_mean_score_on_data(metric, test, \
											(qembedder, aembedder), \
											tokenizer)
		_log.info(f'score {score:9.4f} | loss:bert {total_loss:9.4f} ')

	_log.info('Fine-tuning is compleated')
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	# ToDo save model
	save_model((qembedder, aembedder))

ex.run()
