import torch
from pytorch_transformers import BertTokenizer, BertForQuestionAnswering
from pytorch_transformers import BertModel, BertConfig
import os
import json

class BERTLike(torch.nn.Module):
    max_seq_len = 512
    __name__ = 'BERTLike'

    def prepare_batch(self, batch, device):
        quests, answs = batch.quests, batch.answs

        assert(len(quests) == len(answs))

        quests = [torch.tensor([self.tokenizer.encode(el)], device=device) for el in quests]
        answs = [torch.tensor([self.tokenizer.encode(el)], device=device) for el in answs]

        return (quests, answs)

    def get_embedding(self, embeddings):
        '''
        using bert-as-service-like strategy to get fixed-size vector
        1. considering only -1 layer (NOT -2 as in bert-as-service)
        2. "REDUCE_MEAN take the average of the hidden state of encoding layer on the time axis" @bert-as-service
        '''
        return torch.sum(embeddings, dim=1)

    def embed_batch(self, batch):
        quests, answs = batch

        tmp_quest = [self.get_embedding(self.qembedder(quests[i])[0]) for i in range(len(quests))]
        tmp_answ = [self.get_embedding(self.aembedder(answs[i])[0]) for i in range(len(answs))]

        qembeddings = torch.cat(tmp_quest)
        aembeddings = torch.cat(tmp_answ)

        assert(qembeddings.shape == aembeddings.shape)

        if self.float_mode == 'fp16':
            return (qembeddings.half(), aembeddings.half())

        return (qembeddings, aembeddings)

    def load_from(self, cache_dir):
        self.tokenizer = BertTokenizer.from_pretrained(cache_dir)
        self.qembedder = BertModel.from_pretrained(f'{cache_dir}qembedder/')
        self.aembedder = BertModel.from_pretrained(f'{cache_dir}aembedder/')

    def __init__(self, lang, bert_type, float_mode='fp32', cache_dir=None):
        super().__init__()
        self.lang = lang
        self.bert_type = bert_type
        if lang == 'en':
            if cache_dir is None:
                cache_dir = f'../pretrained-{bert_type}/'
                self.tokenizer = BertTokenizer.from_pretrained(bert_type, cache_dir=cache_dir)
                self.qembedder = BertModel.from_pretrained(bert_type, cache_dir=cache_dir)
                self.aembedder = BertModel.from_pretrained(bert_type, cache_dir=cache_dir)
            else:
                self.load_from(cache_dir)
        elif lang == 'ru':
            if cache_dir is None:
                cache_dir = f'../ru{bert_type}/'
            self.load_from(cache_dir)
        else:
            raise Exception('BERTlike model: unknown language.')

        self.float_mode = float_mode
        if float_mode == 'fp16':
            self.qembedder.half()
            self.aembedder.half()

    def save_to(self, folder):
        if folder[-1] != '/':
            folder = folder + '/'
        os.system(f'mkdir "{folder}"')
        qname = f'{folder}qembedder/'
        aname = f'{folder}aembedder/'
        os.system(f'mkdir "{qname}"')
        os.system(f'mkdir "{aname}"')
        self.qembedder.save_pretrained(qname)
        self.aembedder.save_pretrained(aname)
        self.tokenizer.save_pretrained(folder)
        config = {'float_mode': self.float_mode, 'name': self.__name__, \
                 'lang': self.lang, 'bert_type': self.bert_type}
        with open(f'{folder}model_config.json', 'w') as file:
            json.dump(config, file)

    def forward(self, batch):
        device = next(self.qembedder.parameters()).device
        return self.embed_batch(self.prepare_batch(batch, device))

    def train(self):
        self.qembedder.train()
        self.aembedder.train()

    def eval(self):
        self.qembedder.eval()
        self.aembedder.eval()
