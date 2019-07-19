import numpy as np
from losses import cosine_similarity_table


def calc_accuracy(X, Y):
	csim = cosine_similarity_table(X, Y)
	confidence, predictions = csim.max(-1)
	correct = 0
	# todo: numpy
	for i in range(predictions.shape[0]):
		correct += predictions[i] == i
	return correct


def calc_mrr(X, Y):
	csim = cosine_similarity_table(X, Y)
	n = csim.shape[0]
	mrr = 0
	for i in range(n):
		arr = [(el.item(), idx) for (idx, el) in enumerate(list(csim[i])])))
		arr.sort(reverse=True)
		for j in range(len(arr)):
			if arr[j][1] == i:
				mrr += 1 / (j + 1)
				break
	mrr /= n
	return mrr
