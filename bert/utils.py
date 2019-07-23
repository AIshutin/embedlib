import re
import os
import torch
from pytorch_transformers import BertTokenizer, BertForQuestionAnswering
from pytorch_transformers import BertModel, BertConfig

def remove_urls (vTEXT):
	# r'http\S+'
	# r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b'
	vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '[link]', vTEXT, flags=re.MULTILINE)
	return vTEXT

def prepare_batch(batch, device, tokenizer):
	(quests, answs) = batch
	quests = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(el)) for el in quests]
	answs = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(el)) for el in answs]

	quest_segments = [torch.tensor([[0 for i in range(len(quests[j]))]], device=device) \
															for j in range(len(quests))]
	answ_segments = [torch.tensor([[0 for i in range(len(answs[j] ))]], device=device) \
															for j in range(len(answs))]

	quests = [torch.tensor([el], device=device) for el in quests]
	answs = [torch.tensor([el], device=device) for el in answs]

	return ((quests, quest_segments), (answs, answ_segments))

def get_embedding(embeddings):
	'''
	using bert-as-service-like strategy to get fixed-size vector
	1. considering only -1 layer (NOT -2 as in bert-as-service)
	2. "REDUCE_MEAN take the average of the hidden state of encoding layer on the time axis" @bert-as-service
	'''
	#print('get_embedding', embeddings.shape)
	result = torch.sum(embeddings, dim=1)

	return result

def embed_batch(batch, qembedder, aembedder, float_mode):
	((quests, quest_segments), (answs, answ_segments)) = batch

	tmp_quest = [get_embedding(qembedder(quests[i], quest_segments[i])[0]) for i in range(len(quests))]
	tmp_answ = [get_embedding(aembedder(answs[i], answ_segments[i])[0]) for i in range(len(answs))]

	qembeddings = torch.cat(tmp_quest)
	aembeddings = torch.cat(tmp_answ)

	if float_mode == 'fp16':
		return (qembeddings.half(), aembeddings.half())

	return (qembeddings, aembeddings)

def print_batch(batch):
	for i in range(len(batch[0])):
		print(i, batch[0][i])
		print('>>> ', batch[1][i])
	print()

def load_model(checkpoint_dir):
	print(checkpoint_dir )
	qembedder = BertModel.from_pretrained(checkpoint_dir + 'qembedder/')
	aembedder = BertModel.from_pretrained(checkpoint_dir + 'aembedder/')
	tokenizer = BertTokenizer(checkpoint_dir)
	return (qembedder, aembedder), tokenizer

def save_model(model, tokenizer, checkpoint_dir):
	os.system(f'mkdir "{checkpoint_dir}"')
	qname = checkpoint_dir + 'qembedder/'
	aname = checkpoint_dir + 'aembedder/'
	os.system(f'mkdir "{qname}"')
	os.system(f'mkdir "{aname}"')
	model[0].save_pretrained(qname)
	model[1].save_pretrained(aname)
	tokenizer.save_pretrained(checkpoint_dir)