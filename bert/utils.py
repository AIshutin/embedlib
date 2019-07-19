def remove_urls (vTEXT):
	# r'http\S+'
	# r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b'
	vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '[link]', vTEXT, flags=re.MULTILINE)
	return vTEXT

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
