import re
import torch

def remove_urls (vTEXT):
	# r'http\S+'
	# r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b'
	vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '[link]', vTEXT, flags=re.MULTILINE)
	return vTEXT

def prepare_batch(batch, device, tokenizer):
	(quests, answs) = batch
	#print(quests)
	quests = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(el)) for el in quests]
	answs = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(el)) for el in answs]
	#print(quests)
	#print(quests)
	#print(quests[0].shape)

	#quest_segments = [torch.zeros(len(quests[i]), 1, device=device) for i in range(len(quests))]
	#answ_segments = [torch.zeros(len(answs[i]), 1, device=device) for i in range(len(answs))]

	quest_segments = [torch.tensor([[0 for i in range(len(quests[j]))]], device=device) \
															for j in range(len(quests))]
	answ_segments = [torch.tensor([[0 for i in range(len(answs[j] ))]], device=device) \
															for j in range(len(answs))]

	quests = [torch.tensor([el], device=device) for el in quests]
	answs = [torch.tensor([el], device=device) for el in answs]

	#print(quests)
	#print(quest_segments)

	return ((quests, quest_segments), (answs, answ_segments))

def get_embedding(embeddings):
	'''
	using default bert-as-service strategy to get fixed-size vector
	1. considering only -2 layer
	2. "REDUCE_MEAN take the average of the hidden state of encoding layer on the time axis" @bert-as-service
	'''
	embeddings = embeddings[-2]
	result = torch.sum(embeddings, dim=1)

	return result

def embed_batch(batch, qembedder, aembedder):
	((quests, quest_segments), (answs, answ_segments)) = batch

	tmp_quest = [get_embedding(qembedder(quests[i], quest_segments[i])[0]) for i in range(len(quests))]
	tmp_answ = [get_embedding(aembedder(answs[i], answ_segments[i])[0]) for i in range(len(answs))]

	qembeddings = torch.cat(tmp_quest)
	aembeddings = torch.cat(tmp_answ)

	return (qembeddings, aembeddings)

def print_batch(batch):
	for i in range(len(batch[0])):
		print(i, batch[0][i])
		print('>>> ', batch[1][i])
	print()

def get_optimizer_params(model):
  param_optimizer = list(model.named_parameters())
  no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

  optimizer_grouped_parameters = [
     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
			{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]

  return optimizer_grouped_parameters
