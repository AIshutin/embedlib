import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import BertModel, BertConfig, AutoTokenizer, AutoModel
from transformers import RobertaModel, RobertaTokenizer
import transformers
import pickle
import re

import os
import json
import json
import six

import numpy
import gc
import youtokentome as yttm
from .tokenizers import RubertaTokenizer

class EmbedderModel(torch.nn.Module):
    max_seq_len = 512
    __name__ = "EmbedderModel"

    def prepare_halfbatch(self, batch, device):
        token = '[CLS]'

        for i in range(len(batch)):
            batch[i] = batch[i].strip()
            assert(batch[i][:len(token)] != token)
            if batch[i] == '':
                batch[i] = '0'

        if self.config['batch_mode']:
            mx_len = 0
            encoded = [self.tokenizer.encode(el, add_special_tokens=True) for el in batch]
            for el in encoded:
                mx_len = max(mx_len, len(el))
            attention_mask = [[1] * len(encoded[i]) + [0] * (mx_len - len(encoded[i])) \
                                                        for i in range(len(encoded))]
            for i in range(len(encoded)):
                encoded[i] += [self.tokenizer.pad_token_id] * (mx_len - len(encoded[i])) # [PAD] token

            encoded = torch.tensor(encoded, device=device)
            attention_mask = torch.tensor(attention_mask, device=device)
            return (encoded, attention_mask)
        else:
            encoded = [torch.tensor([self.tokenizer.encode(el)], device=device) for el in batch]
            return encoded

    def get_embedding(self, embeddings):
        '''
        using bert-as-service-like strategy to get fixed-size vector
        1. considering only -1 layer (NOT -2 as in bert-as-service)
        2. "REDUCE_MEAN take the average of the hidden state of encoding layer on the time axis" @bert-as-service
        '''
        return torch.sum(embeddings, dim=1)

    def embed_halfbatch(self, batch, is_quest=True):
        if is_quest:
            embedder = self.qembedder
        else:
            embedder = self.aembedder

        if self.config['batch_mode']:
            embeddings = self.get_embedding(embedder(batch[0], attention_mask=batch[1])[0])
        else:
            tmp_embeds = [self.get_embedding(embedder(el)[0]) for el in batch]
            embeddings = torch.cat(tmp_embeds)

        if self.config['float_mode'] == 'fp16':
            return embeddings.half()
        else:
            return embeddings

    def load_from(self, cache_dir):
        self.tokenizer = pickle.load(open(cache_dir + 'tokenizer.pkl', 'rb'))
        if 'qembedder' in self.config['models']:
            self.qembedder = AutoModel.from_pretrained(cache_dir + 'qembedder/')
        if 'aembedder' in self.config['models']:
            self.aembedder = AutoModel.from_pretrained(cache_dir + 'aembedder/')

    def __init__(self):
        super().__init__()
        gc.collect()
        self.eval()

    def save_to(self, folder):
        if folder[-1] != '/':
            folder = folder + '/'
        os.system(f'mkdir "{folder}"')
        qname = f'{folder}qembedder/'
        aname = f'{folder}aembedder/'
        os.system(f'mkdir "{qname}"')
        os.system(f'mkdir "{aname}"')
        if 'qembedder' in self.config['models']:
            self.qembedder.save_pretrained(qname)
        if 'aembedder' in self.config['models']:
            self.aembedder.save_pretrained(aname)
        pickle.dump(self.tokenizer, open(folder + 'tokenizer.pkl', 'wb'))
        with open(f'{folder}model_config.json', 'w') as file:
            json.dump(self.config, file)

    def qembedd(self, quests):
        device = next(self.qembedder.parameters()).device
        return self.embed_halfbatch(self.prepare_halfbatch(quests, device), is_quest=True)

    def aembedd(self, answs):
        device = next(self.qembedder.parameters()).device
        return self.embed_halfbatch(self.prepare_halfbatch(answs, device), is_quest=False)

    def forward(self, batch):
        device = next(self.qembedder.parameters()).device
        quests, answs = batch.quests, batch.answs
        assert(len(quests) == len(answs))
        return (self.qembedd(quests), self.aembedd(answs))

    def has_qembedder(self):
        return 'qembedder' in self.config['models']

    def has_aembedder(self):
        return 'aembedder' in self.config['models']


class BERTLike(EmbedderModel):
    __name__ = 'BERTLike'

    def __init__(self, lang, bert_type, float_mode='fp32', \
            models=['aembedder', 'qembedder'], cache_dir=None, version='???'):

        self.config = {'models': []}
        super().__init__()
        self.config = {'lang': lang,
                       'bert_type': bert_type,
                       'models': models,
                       'version': version,
                       'float_mode': float_mode,
                       'name': self.__name__,
                       'batch_mode': False}


        if lang == 'en':
            if cache_dir is None:
                cache_dir = '../pretrained-' + bert_type + '/'
                self.tokenizer = AutoTokenizer.from_pretrained(bert_type, cache_dir=cache_dir)
                if 'qembedder' in models:
                    self.qembedder = BertModel.from_pretrained(bert_type, cache_dir=cache_dir)
                if 'aembedder' in models:
                    self.aembedder = BertModel.from_pretrained(bert_type, cache_dir=cache_dir)
            else:
                self.load_from(cache_dir)
        elif lang == 'ru':
            if cache_dir is None:
                cache_dir = '../ru' + bert_type + '/'
            self.load_from(cache_dir)
        else:
            raise Exception('BERTlike model: unknown language.')

        gc.collect()
        if float_mode == 'fp16':
            if 'qembedder' in models:
                self.qembedder.half()
                gc.collect()
            if 'aembedder' in models:
                self.aembedder.half()
                gc.collect()

        self.eval()

    def eval(self):
        if 'qembedder' in self.config['models']:
            self.qembedder.eval()
        if 'aembedder' in self.config['models']:
            self.aembedder.eval()
        self.config['batch_mode'] = True

    def train(self):
        if 'qembedder' in self.config['models']:
            self.qembedder.train()
        if 'aembedder' in self.config['models']:
            self.aembedder.train()
        self.config['batch_mode'] = False

class GPT2Like(EmbedderModel):
    __name__ = 'GPT2Like'

    '''def prepare_halfbatch(self, batch, device):
        for i in range(len(batch)):
            batch[i] = batch[i].strip()
            assert(batch[i][:len('[CLS]')] != '[CLS]')
            if len(batch[i]) == 0:
                batch[i] = '0'

        if self.batch_mode:
            mx_len = 0
            encoded = [self.tokenizer.encode(el) for el in batch]
            for el in encoded:
                mx_len = max(mx_len, len(el))
            attention_mask = [[1] * len(encoded[i]) + [0] * (mx_len - len(encoded[i])) \
                                                        for i in range(len(encoded))]
            for i in range(len(encoded)):
                encoded[i] += [0] * (mx_len - len(encoded[i])) # [PAD] token

            encoded = torch.tensor(encoded, device=device)
            attention_mask = torch.tensor(attention_mask, device=device)
            return (encoded, attention_mask)
        else:
            encoded = [torch.tensor([self.tokenizer.encode(el)], device=device) for el in batch]
            #print(encoded)
            return encoded'''

    def __init__(self, lang, float_mode='fp32', \
            models=['aembedder', 'qembedder'], cache_dir=None, version='???'):
        super().__init__()

        self.config = {'lang': lang,
                       'models': models,
                       'version': version,
                       'float_mode': float_mode,
                       'name': self.__name__,
                       'batch_mode': False}

        if lang == 'en':
            if cache_dir is None:
                cache_dir = f'../pretrained-{self.__name__}-{self.lang}' +  '/'
                transformer_name = "microsoft/DialoGPT-small"
                self.tokenizer = AutoTokenizer.from_pretrained(transformer_name, cache_dir=cache_dir)
                if 'qembedder' in models:
                    self.qembedder = AutoModel.from_pretrained(transformer_name, cache_dir=cache_dir)
                if 'aembedder' in models:
                    self.aembedder = AutoModel.from_pretrained(transformer_name, cache_dir=cache_dir)
            else:
                self.load_from(cache_dir)
        elif lang == 'ru':
            if cache_dir is None:
                cache_dir = f'../ru-{self.__name__}/'
                model = AutoModel.from_pretrained(cache_dir)
                if 'qembedder' in models:
                    self.qembedder = model
                if 'aembedder' in models:
                    self.aembedder = model
                vocab_path = os.path.join(cache_dir, 'vocab_50000.bpe')
                self.file = open(vocab_path, 'rb').read()
                self.tokenizer = RubertaTokenizer(os.path.join(cache_dir, 'vocab_50000.bpe'))
            else:
                self.load_from(cache_dir)
            self.tokenizer.encode('Привет, мир!') # Test
        else:
            raise Exception(f'{self.__name__} model: unknown language.')

        gc.collect()
        if float_mode == 'fp16':
            if 'qembedder' in models:
                self.qembedder.half()
                gc.collect()
            if 'aembedder' in models:
                self.aembedder.half()
                gc.collect()
        self.eval()

class RoBERTaLike(EmbedderModel):
    __name__ = 'RoBERTaLike'

    '''def prepare_halfbatch(self, batch, device):
        token = '[CLS]'

        for i in range(len(batch)):
            batch[i] = batch[i].strip()
            if batch[i][:len(token)] != token:
                batch[i] = token + ' ' + batch[i]

        if self.batch_mode:
            mx_len = 0
            encoded = [self.tokenizer.encode(el) for el in batch]
            for el in encoded:
                mx_len = max(mx_len, len(el))
            attention_mask = [[1] * len(encoded[i]) + [0] * (mx_len - len(encoded[i])) \
                                                        for i in range(len(encoded))]
            for i in range(len(encoded)):
                encoded[i] += [self.tokenizer.pad_token_id] * (mx_len - len(encoded[i])) # [PAD] token

            encoded = torch.tensor(encoded, device=device)
            attention_mask = torch.tensor(attention_mask, device=device)
            return (encoded, attention_mask)
        else:
            encoded = [torch.tensor([self.tokenizer.encode(el)], device=device) for el in batch]
            return encoded'''

    def __init__(self, lang, float_mode='fp32', \
            models=['aembedder', 'qembedder'], cache_dir=None, version='???'):
        super().__init__()

        self.config = {'lang': lang,
                       'models': models,
                       'version': version,
                       'float_mode': float_mode,
                       'name': self.__name__,
                       'batch_mode': False}

        if lang == 'en':
            if cache_dir is None:
                cache_dir = f'../pretrained-{self.__name__}'
                bert_type = 'roberta-base'
                self.tokenizer = RobertaTokenizer.from_pretrained(bert_type, cache_dir=cache_dir)
                if 'qembedder' in models:
                    self.qembedder = RobertaModel.from_pretrained(bert_type, cache_dir=cache_dir)
                if 'aembedder' in models:
                    self.aembedder = RobertaModel.from_pretrained(bert_type, cache_dir=cache_dir)
            else:
                self.load_from(cache_dir)
        elif lang == 'ru':
            if cache_dir is None:
                cache_dir = '../ruberta_base/'
                vocab_path = os.path.join(cache_dir, 'vocab_50000.bpe')
                self.file = open(vocab_path, 'rb').read()
                self.tokenizer = RubertaTokenizer(os.path.join(cache_dir, 'vocab_50000.bpe'))
                self.tokenizer.encode('Привет, мир!') # Test

                if 'qembedder' in models:
                    self.qembedder = RobertaModel.from_pretrained(cache_dir)
                if 'aembedder' in models:
                    self.aembedder = RobertaModel.from_pretrained(cache_dir)
            else:
                self.load_from(cache_dir)
        else:
            raise Exception('BERTlike model: unknown language.')

        gc.collect()
        self.float_mode = float_mode
        if float_mode == 'fp16':
            if 'qembedder' in models:
                self.qembedder.half()
                gc.collect()
            if 'aembedder' in models:
                self.aembedder.half()
                gc.collect()

        self.eval()
