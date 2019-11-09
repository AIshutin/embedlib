import numpy as np
from .losses import cosine_similarity_table
import torch
from random import *

def calc_accuracy(X, Y):
    csim = cosine_similarity_table(X, Y)
    confidence, predictions = csim.max(-1)
    correct = 0
    # ToDo: numpy/pytorch
    for i in range(predictions.shape[0]):
        correct += predictions[i] == i
    return correct / X.shape[0]

def calc_random_accuracy(batch_size):
    return 1 / N_batch

def calc_mrr(X, Y, silent=True):
    print('cosine')
    with torch.no_grad():
        print('ln21', 'cosine_similarity_table' )
        csim = cosine_similarity_table(X, Y).detach().cpu().numpy()
        n = csim.shape[0]
        mrr = 0
        for i in range(n):
            print('it1')
            ind = csim[i][i]
            csim[i].sort()
            print('it1..')
            mrr += 1 / (csim[i].shape[0] - np.searchsorted(csim[i], ind))
            print('it1!!')
        mrr /= n
        print('done')
        assert(mrr <= 1.01)
    return mrr

def calc_placeholder_metric(X, Y):
    return 0

def calc_random_mrr(batch_size):
    mrrs = 0
    for i in range(1, 1 + batch_size):
        mrrs += 1 / i
    return mrrs / batch_size

def get_mean_on_data(metric, data, model):
    if type(metric) != type(list()) and type(metric) != type(tuple()):
        metric = [metric]
    results = [None] * len(metric)
    model.eval()
    testbatch_cnt = 0
    with torch.no_grad():
        for batch in data:
            print('in')
            embeddings = model(batch)
            print('metric')
            for i in range(len(metric)):
                curr = metric[i](*embeddings)
                if results[i] is None:
                    results[i] = curr
                else:
                    results[i] += curr
            testbatch_cnt += 1
            print('out')
    return [el / testbatch_cnt for el in results]
