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

from sacred import Experiment
ex = Experiment()

@ex.config
def config():
	test_split = 0.2
	checkpoint_dir = "checkpoints"
	learning_rate = 5e-5
	metric_func = 'calc_mrr'
	criterion_func = 'hinge_loss'
	# BERT config
	max_seq_len = 512
	bert_type = 'bert-base-uncased'
	cache_dir = './pretrained-' + bert_type

@ex.capture
def get_data(_log, test_split, data_path):
	corpus = UbuntuCorpus(tokenizer)
	_log.info('Corpus length: ' + len(corpus))

	return train, test

@ex.capture
def get_metric(metric_func):
	metric = getattr(metrics, metric_func)
	return metric

@ex.capture
def get_criterion(criterion_func):
	criterion = getattr(losses, criterion_func)
	return criterion

@ex.capture
def get_model():
	tokenizer = BertTokenizer.from_pretrained(bert_type, cache_dir=cache_dir)

	return tokenizer, model

@ex.automain
def train(_log, dataset):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	tokenizer, model = get_model()
	train, test = get_data()

	metric = get_metric()
	criterion = get_criterion()


def get_optimizer_params(model):
  param_optimizer = list(model.named_parameters())
  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

  optimizer_grouped_parameters = [
     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
			{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]

  return optimizer_grouped_parameters

test_size = int(len(corpus) * .33)
train_size = len(corpus) - test_size
train_corpus, test_corpus = torch.utils.data.random_split(corpus, [train_size, test_size])

def print_batch(batch):
	for i in range(len(batch[0])):
		print(i, batch[0][i])
		print('>>> ', batch[1][i])
	print()


def train(epochs):
	batch_size = 10 # 16, 32 are recommended in the paper
	trainloader = DataLoader(corpus, batch_size=batch_size, shuffle=True) #
	testloader = trainloader # DataLoader(test_corpus, batch_size=batch_size, shuffle=True)
	num_train_optimization_steps = len(corpus) * epochs

	lr = 5e-5 # 5e-5, 3e-5, 2e-5 are recommended in the paper
	warmup = 0.1

	qoptim = BertAdam(get_optimizer_params(qembedder),
						lr=lr,
						warmup=warmup,
						t_total=num_train_optimization_steps)

	aoptim = BertAdam(get_optimizer_params(aembedder),
						lr=lr,
						warmup=warmup,
						t_toal=num_train_optimization_steps)
	criterion = hinge_loss

	total = right = 0
	with torch.no_grad():
		for batch in testloader:
			total += len(batch[0])
			print_batch(batch)
			embeddings = embed_batch(prepare_batch(batch), qembedder, aembedder)
			right += calc_acc(*embeddings)

	qembedder.train()
	aembedder.train()

	logger.info("***** Running training *****")
	logger.info("  Num steps = %d", num_train_optimization_steps)
	logger.info(f"Before training: right: {right} of {total}")

	start_training = time.time()
	for epoch in range(epochs):
		total_loss = 0
		qembedder.train()
		aembedder.train()
		for bidx, batch in enumerate(tqdm.tqdm(iter(trainloader), desc=f"epoch {epoch}")):
			qoptim.zero_grad()
			aoptim.zero_grad()
			#print('batch_index', bidx)

			embeddings = embed_batch(prepare_batch(batch), qembedder, aembedder)
			loss = criterion(*embeddings)
			total_loss += loss.item()
			loss.backward()

			qoptim.step()
			aoptim.step()

		total = right = 0
		qembedder.eval()
		aembedder.eval()
		with torch.no_grad():
			for batch in testloader:
				total += len(batch[0])
				print_batch(batch)

				embeddings = embed_batch(prepare_batch(batch), qembedder, aembedder)
				right += calc_acc(*embeddings)

		print(f'right: {right} of {total} | loss: {total_loss} ')

	end_training = time.time()
	logger.info(f'Training is compleated. Time: {int(end_training - start_training)}')
	torch.cuda.empty_cache()

qembedder = BertModel.from_pretrained(bert_type, cache_dir=cache_dir).to(device)
aembedder = BertModel.from_pretrained(bert_type, cache_dir=cache_dir).to(device)
train(3) # 2, 3, 4 epochs are recommended in the paper
