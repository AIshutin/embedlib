import torch
from pytorch_transformers import BertTokenizer, BertForQuestionAnswering
from pytorch_transformers import BertModel, BertConfig
import os
import json

class BERTLike(torch.nn.Module):
    max_seq_len = 512
    __name__ = 'BERTLike'

    def prepare_halfbatch(self, batch, device):
        mx_len = 0
        encoded = [self.tokenizer.encode(el) for el in batch]
        for el in encoded:
            mx_len = max(mx_len, len(el))
        attention_mask = [[1] * len(encoded[i]) + [0] * (mx_len - len(encoded[i])) \
                                                    for i in range(len(encoded))]
        for i in range(len(encoded)):
            encoded[i] += self.tokenizer.encode('[PAD]') * (mx_len - len(encoded[i])) # [PAD] token

        encoded = torch.tensor(encoded, device=device)
        attention_mask = torch.tensor(attention_mask, device=device)
        return (encoded, attention_mask)

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

        embeddings = self.get_embedding(embedder(batch[0], attention_mask=batch[1])[0])

        if self.float_mode == 'fp16':
            return embeddings.half()
        else:
            return embeddings

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

    def train(self):
        self.qembedder.train()
        self.aembedder.train()

    def eval(self):
        self.qembedder.eval()
        self.aembedder.eval()
