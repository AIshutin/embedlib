import numpy as np
from losses import cosine_similarity_table
from utils import prepare_batch, embed_batch
import torch

def calc_accuracy(X, Y):
	csim = cosine_similarity_table(X, Y)
	confidence, predictions = csim.max(-1)
	correct = 0
	# ToDo: numpy
	for i in range(predictions.shape[0]):
		correct += predictions[i] == i
	return correct / X.shape[0]

def calc_mrr(X, Y):
	csim = cosine_similarity_table(X, Y)
	n = csim.shape[0]
	mrr = 0
	for i in range(n):
		arr = [(el.item(), idx) for (idx, el) in enumerate(list(csim[i]))]
		arr.sort(reverse=True)
		for j in range(len(arr)):
			if arr[j][1] == i:
				mrr += 1 / (j + 1)
				break
	mrr /= n
	assert(mrr <= 1.01)
	return mrr

def get_mean_score_on_data(metric, data, model, tokenizer):
	qembedder, aembedder = model
	device = next(qembedder.parameters()).device

	qembedder.eval()
	aembedder.eval()
	score = 0
	testbatch_cnt = 0
	with torch.no_grad():
		for batch in data:
			embeddings = embed_batch(prepare_batch(batch, device, tokenizer), qembedder, aembedder)
			score += metric(*embeddings)
			testbatch_cnt += 1
	score /= testbatch_cnt
	return score
