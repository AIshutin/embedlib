from torch.utils.data import Dataset
import os
import csv
from .utils import remove_urls
from itertools import groupby
import torch

class UbuntuCorpus(Dataset):
    def __init__(self, tokenizer, _cnt=30, dir='../dialogs', max_seq_len=512):
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
                    replicas[i] = remove_urls(' '.join(replicas[i]))

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


class TwittCorpus(Dataset):
    def __init__(self, tokenizer, max_dataset_size=100, path='../rucorp.txt', max_seq_len=512):
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
                el[0] = el[0].replace('[SEP]', '').rstrip()
                el[1] = el[1].replace('[SEP]', '').rstrip()
                el[0] = el[0].replace('[CLS]', '').rstrip()
                el[1] = el[1].replace('[CLS]', '').rstrip()
            except Exception as exp:
                print(exp)
                print(el)
                raise exp
            qtok = tokenizer.encode(el[0])
            atok = tokenizer.encode(el[1])
            if max(len(qtok), len(atok)) <= max_seq_len:
                good.append([el[0], el[1]])
                if len(good) == max_dataset_size:
                    break
        self.qa_s = good

    def __len__(self):
        return len(self.qa_s)

    def __getitem__(self, idx):
        return (self.qa_s[idx][0], self.qa_s[idx][1])


corpus_handler = {'en-ubuntu-corpus': (UbuntuCorpus, '../dialogs'), \
                'en-twitt-corpus': (TwittCorpus, '../corp.txt'), \
                'ru-opendialog-corpus': (TwittCorpus, '../rucorp-subtitles.txt'), \
                'ru-toloka': (TwittCorpus, '../rucorp.txt')}

class CorpusData(Dataset):
    def __init__(self, names, tokenizer, max_dataset_size=100, max_seq_len=512):
        super(CorpusData, self).__init__()
        datasets = [corpus_handler[el][0](tokenizer, max_dataset_size, corpus_handler[el][1], max_seq_len) \
                    for el in names]
        self.data = torch.utils.data.ConcatDataset(datasets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
