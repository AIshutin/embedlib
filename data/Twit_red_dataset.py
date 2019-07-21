import torch
from pytorch_pretrained_bert import BertTokenizer, BertForQuestionAnswering, BertModel
from torch import nn
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms
from pytorch_pretrained_bert.optimization import BertAdam
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import PIL
from IPython.display import HTML
import pickle
from torch.utils.data import Dataset, DataLoader
import os
import csv
import random
from sklearn.utils import shuffle
import time
from tqdm import tqdm_notebook

from itertools import groupby
class twitt_dataset(Dataset):
    def __init__(self, tokenizer, pth = 'corp (1).txt'):
        '''
            Gets Path to TXT file in format
            [CLS] Qestion [SEP] \n
            [CLS] Answer [SEP]\n
            \n
            ...
        '''
        super(twitt_dataset, self).__init__()
        with open(pth, 'r') as f:
            reps = f.readlines()
        dgs = [list(group) for k, group in groupby(reps[:1000001], lambda x: x == '\n') if not k]
        for i in range(len(dgs)):
            dgs[i][0] = dgs[i][0].strip('\n')
            dgs[i][1] = dgs[i][1].strip('\n')
        self.qa_s = dgs
    def __len__(self):
        return len(self.qa_s)
    def __getitem__(self,idx):
        return (self.qa_s[idx][0], self.qa_s[idx][1])
