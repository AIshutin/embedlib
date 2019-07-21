import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_similarity_table(X, Y):
	X = F.normalize(X)
	Y = F.normalize(Y)
	return torch.mm(X, Y.transpose(0, 1))


def bce_loss(X, Y, conf_true=0.9, conf_false=0.1):
	n = X.shape[0]

	logits = torch.mm(X, Y.transpose(0, 1))
	identity = torch.eye(n, device=X.device)

	non_diagonal = torch.ones_like(logits) - identity
	targets = identity * conf_true + non_diagonal * conf_false

	weights = identity + non_diagonal / (n - 1)
	return F.binary_cross_entropy_with_logits(logits, targets, weights) * n


def hinge_loss(X, Y, margin=0.1):
	batch_size = X.shape[0]
	similarities = cosine_similarity_table(X, Y)

	identity = torch.eye(batch_size, device=X.device)
	non_diagonal = torch.ones_like(similarities) - identity

	targets = identity - non_diagonal
	weights = identity + non_diagonal / (batch_size - 1)

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
