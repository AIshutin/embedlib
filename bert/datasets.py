from torch.utils.data import Dataset
import os
import csv
from utils import remove_urls
from itertools import groupby
import torch

class UbuntuCorpus(Dataset):
    def __init__(self, tokenizer, dir='./dialogs', max_seq_len=512, _cnt=30):
        super().__init__()
        dialogs = []
        thr = 30

        qa_pairs = []

        for subdir in os.listdir(dir):
            for dialog in os.listdir(dir + '/' + subdir):
                path = dir + '/' + subdir +'/' + dialog
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
                        _score_cnt -= 1
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
        return (self.qa_pairs[ind][0], self.qa_pairs[ind][1])

class TokenizedQABatch:
    def __init__(self, data):
        self.quests = data[0]
        self.answs = data[1]
        assert(len(data[0]) == len(data[1]))

    def pin_memory(self):
        for i in range(len(quests)):
            self.quests[i].pin_memory()
            self.answs[i].pin_memory()
        return self

def collate_wrapper(batch):
    return TokenizedQABatch(batch)

class TwittCorpus(Dataset):
    def __init__(self, tokenizer, path='corp.txt', max_seq_len=512, max_dataset_size=100):
        '''
        Gets Path to TXT file in format
        [CLS] Qestion [SEP] \n
        [CLS] Answer [SEP]\n
        \n
        ...
        '''
        super().__init__()
        with open(path, 'r') as f:
            reps = f.readlines()
        dgs = [list(group) for k, group in groupby(reps, lambda x: x == '\n') if not k]
        good = []
        for el in dgs:
            el[0] = el[0].replace('[SEP]', '').rstrip()
            el[1] = el[1].replace('[SEP]', '').rstrip()
            qtok = tokenizer.encode(el[0])
            atok = tokenizer.encode(el[1])
            if max(len(qtok), len(atok)) <= max_seq_len:
                good.append([qtok, atok])
                if len(good) == max_dataset_size:
                    break
        self.qa_s = good

    def __len__(self):
        return len(self.qa_s)

    def __getitem__(self, idx):
        return (self.qa_s[idx][0], self.qa_s[idx][1])
