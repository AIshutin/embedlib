# -*- coding: utf-8 -*-
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForQuestionAnswering, BertModel
from torch import nn
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms
from pytorch_pretrained_bert.optimization import BertAdam
import numpy as np
import PIL
import pickle
from torch.utils.data import Dataset, DataLoader
import os
import csv
import random
from sklearn.utils import shuffle
import time
import tqdm
import re

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

os.system('chmod +x download_datasets.sh')
os.system('./download_datasets.sh')

device = torch.device('cuda:0') if torch.cuda.device_count() >= 1 else torch.device('cpu')
bert_type = 'bert-base-uncased'

max_seq_len = 512 # BERT restriction
cache_dir = './pretrained-' + bert_type
tokenizer = BertTokenizer.from_pretrained(bert_type, cache_dir=cache_dir)


def remove_urls (vTEXT):
	# r'http\S+'
	# r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b'
	vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '[link]', vTEXT, flags=re.MULTILINE)
	return(vTEXT)

def prepare_batch(batch):
	(quests, answs) = batch
	quests = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(el)) for el in quests]
	answs = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(el)) for el in answs]
  
	quest_segments = [torch.tensor([[0 for i in range(len(quests[j]))]]) for j in range(len(quests))]
	answ_segments = [torch.tensor([[0 for i in range(len(answs[j] ))]]) for j in range(len(answs))]
  
	quests = [torch.tensor([el]) for el in quests]
	answs = [torch.tensor([el]) for el in answs]
  
	return ((quests, quest_segments), (answs, answ_segments))

def get_embedding(embeddings):
	'''
	using default bert-as-service strategy to get fixed-size vector
	1. considering only -2 layer
	2. "REDUCE_MEAN take the average of the hidden state of encoding layer on the time axis" @bert-as-service  
	'''
	embeddings = embeddings[-2]
	result = torch.sum(embeddings, dim=1)

	return result.to(device)

def embed_batch(batch, qembedder, aembedder):
	((quests, quest_segments), (answs, answ_segments)) = batch
  
	tmp_quest = [get_embedding(qembedder(quests[i].to(device), quest_segments[i].to(device))[0]) for i in range(len(quests))]
	tmp_answ = [get_embedding(aembedder(answs[i].to(device), answ_segments[i].to(device))[0]) for i in range(len(answs))]
  
	qembeddings = torch.cat(tmp_quest)
	aembeddings = torch.cat(tmp_answ)
	
	return (qembeddings, aembeddings)

class UbuntuCorpus(Dataset):
	def __init__(self, tokenizer, rootdir='./dialogs'):
		super(UbuntuCorpus, self).__init__()
		dialogs = []
		_cnt = 30 # debug constant
		thr = 30

		qa_pairs = []
		
		for subdir in os.listdir(rootdir):
			for dialog in os.listdir(rootdir + '/' + subdir):
				path = rootdir + '/' + subdir +'/' + dialog 
				with open(path) as tsvfile:
					reader = csv.reader(tsvfile, delimiter='\t')
					rows = [(row[1], row[-1]) for row in reader]
					replicas = []
					authors = set()
					author = -1
					for row in rows:
						if author == row[0]:
							replicas[-1].append(row[1])
						else:
							author = row[0]
							authors.add(author)
							replicas.append([row[1]])
					assert(len(authors) <= 2)
				'''
				Answer replic is a replic without ?
			 	Question replic is a replic with ? followed by answer replic

			 	Both must be longer than thr (after link replacemenets)
				
				And due to BERT restrictions both in tokenized form must be shorter than max_seq_len
				'''
		  
				for i in range(len(replicas)):
					replicas[i] = '[CLS] ' + remove_urls(' '.join(replicas[i]))

				for i in range(len(replicas) - 1):
					if replicas[i].count('?') > 0 and replicas[i + 1].count('?') == 0 \
					  and min(len(replicas[i]), len(replicas[i + 1])) >= thr \
					  and len(tokenizer.tokenize(replicas[i])) <= max_seq_len \
					  and len(tokenizer.tokenize(replicas[i + 1])) <= max_seq_len:
						qa_pairs.append([replicas[i], replicas[i + 1]])
						_cnt -= 1
						if _cnt <=0:
							break
				if _cnt <= 0:
					break
			if _cnt <=0:
				break
		'''for el in qa_pairs:
		  print('>>', el[0])
		  print('>>>', el[1])
		  print()'''
		
		self.qa_pairs = qa_pairs
  
	def __len__(self):
		return len(self.qa_pairs)
  
	def __getitem__(self, ind):
		return (self.qa_pairs[ind][0], self.qa_pairs[ind][1])#answ)      
		
corpus = UbuntuCorpus(tokenizer) # full corpus, 1,917,802 qa pairs 
print('Corpus length:', len(corpus))


def bce_loss(X, Y, conf_true=0.9, conf_false=0.1): 
	'''на вход пришел батч размера n,
	мы векторизовали контексты (X)
	и ответы (Y) и хотим сделать n*n
	независимых классификаций
	'''
	n = X.shape[0]

	logits = torch.mm(X, Y.transpose(0, 1)) # считаем таблицу умножения
	identity = torch.eye(n, device=X.device)
  
	non_diagonal = torch.ones_like(logits) - identity
	targets = identity * conf_true + non_diagonal * conf_false
	#получаем матрицу с conf_true на диагонали и conf_false где-либо ещё
  
	weights = identity + non_diagonal / (n - 1)
	# ^ чтобы не было перекоса в сторону негативов
	return F.binary_cross_entropy_with_logits(logits, targets, weights) * n

def cosine_similarity_table(X, Y):
	X = F.normalize(X)
	Y = F.normalize(Y)
	return torch.mm(X, Y.transpose(0, 1))

def hinge_loss(X, Y, margin=0.1):
	batch_size = X.shape[0]
	similarities = cosine_similarity_table(X, Y)
	#^ см. ниже
  
	identity = torch.eye(batch_size, device=X.device)
	non_diagonal = torch.ones_like(similarities) - identity

	targets = identity - non_diagonal
	weights = identity + non_diagonal / (batch_size - 1)
  
	#всё то же самое, но лосс другой: учитываем только то, что не превосходит margin
	losses = torch.pow(F.relu(margin - targets * similarities), 2)
	return torch.mean(losses * weights)


def triplet_loss(X, Y, margin=0.1):
	# https://omoindrot.github.io/triplet-loss
	
	batch_size = X.shape[0]
	similarities = cosine_similarity_table(X, Y)
	
	# qa-pair is positive
	# q-another a pair is negative
	# q-q pair is negative
	# a-a pair is negative
	
	# Approach
	# qa versus a + qa versus q

	right_conf = torch.eye(batch_size, device=X.device) * similarities
	max_confq, _ = similarities.max(1)
	max_confa, _ = similarities.max(0)
	total_loss = F.relu(max_confq - right_conf + margin) + F.relu(max_confa - right_conf + margin)    
	
	return total_loss.mean()

def calc_acc(X, Y):
	'''на вход пришел батч размера n,
	мы векторизовали контексты (X)
	и ответы (Y)'''
	
	csim = cosine_similarity_table(X, Y)
	confidence, predictions = csim.max(-1)
	print(predictions)
	right = 0
	for i in range(predictions.shape[0]):
		right += predictions[i] == i
	return right

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

class ApplyFunc(nn.Module):
    def __init__(self, func):
        super(ApplyFunc, self).__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)

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