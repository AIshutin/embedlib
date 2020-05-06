from torch.utils.data import Dataset
import os
import csv
from .utils import remove_urls
from itertools import groupby
import torch
import json

LIBPATH = os.path.dirname(os.path.dirname(__file__))

class UbuntuCorpus(Dataset):
    def __init__(self, _cnt=30, dir='../dialogs'):
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
                '''

                for i in range(len(replicas)):
                    replicas[i] = remove_urls(' '.join(replicas[i]))

                for i in range(len(replicas) - 1):
                    if replicas[i].count('?') > 0 and replicas[i + 1].count('?') == 0 \
                      and min(len(replicas[i]), len(replicas[i + 1])) >= thr:
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
        self.quests = []
        self.answs = []
        for el in data:
            self.quests.append(el[0])
            self.answs.append(el[1])
        #assert(len(data[0]) == len(data[1]))

    def pin_memory(self):
        for i in range(len(quests)):
            self.quests[i].pin_memory()
            self.answs[i].pin_memory()
        return self

def collate_wrapper(batch):
    return TokenizedQABatch(batch)

class SberQuAD(Dataset):
    files = ['dev-1.1.json', 'train-1.1.json']

    def parse_file(self, path):
        result = []
        data = json.load(open(path))
        print(data.keys())
        data = data['data']['paragraphs']
        for paragraph in data:
            answer = paragraph['context']
            for qas in paragraph['qas']:
                question = qas['question']
                result.append((question, answer))
        return result

    def __init__(self, path=os.path.join(LIBPATH, 'sberquad')):
        self.data = []
        for file in self.files:
            self.data += self.parse_file(os.path.join(path, file))

    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        return len(self.data)

class TwittCorpus(Dataset):
    def __init__(self, max_dataset_size=100, path='../rucorp.txt'):
        '''
        Gets Path to TXT file in format
        [CLS] Question [SEP] \n
        [CLS] Answer [SEP]\n
        \n
        ...
        '''
        super().__init__()
        with open(path, 'r') as f:
            reps = f.readlines()
        """
        state
        0 - q
        1 - a
        """
        state = 0
        dialogs = []
        quest = answ = ""
        start_tag = "[CLS]"
        end_tag = "[SEP]"
        for line in reps:
            line = line.strip()
            if state == 0:
                quest = quest + line
                if end_tag in line:
                    state = 1
            else:
                answ = answ + line
                if end_tag in line:
                    dialogs.append([quest, answ])
                    quest = answ = ""
                    state = 0
        good = []
        for el in dialogs:
            try:
                for k in range(2):
                    el[k] = el[k].replace('[SEP]', '').replace('[CLS]', '').rstrip()
                    if el[k] == '':
                        el[k] = '-'
            except Exception as exp:
                print(exp)
                print(el)
                raise exp
            good.append([el[0], el[1]])
            if len(good) == max_dataset_size:
                break
        self.qa_s = good

    def __len__(self):
        return len(self.qa_s)

    def __getitem__(self, idx):
        return (self.qa_s[idx][0], self.qa_s[idx][1])


corpus_handler = {'en-ubuntu-corpus': (UbuntuCorpus, 'dialogs'), \
                'en-twitt-corpus': (TwittCorpus, 'corp.txt'), \
                'ru-opendialog-corpus': (TwittCorpus, 'rucorp-subtitles.txt'), \
                'ru-toloka': (TwittCorpus, 'rucorp.txt'),
                'sberquad': (SberQuAD, 'sberquad')
                }

class CorpusData(Dataset):
    def __init__(self, names, max_dataset_size=100):
        super(CorpusData, self).__init__()
        datasets = []
        for el in names:
            datasets.append(corpus_handler[el][0](path=os.path.join(LIBPATH, corpus_handler[el][1]), \
                                  max_dataset_size=max_dataset_size))

        self.data = torch.utils.data.ConcatDataset(datasets)
        self.max_dataset_size = max_dataset_size

    def __len__(self):
        return min(len(self.data), self.max_dataset_size)

    def __getitem__(self, idx):
        return self.data[idx]
