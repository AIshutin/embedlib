import torch
from pytorch_transformers import BertTokenizer, BertForQuestionAnswering
from pytorch_transformers import BertModel, BertConfig
import pickle

import os
import json

import numpy
import gc

class BERTLike(torch.nn.Module):
    max_seq_len = 512
    __name__ = 'BERTLike'

    def prepare_halfbatch(self, batch, device):
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
                encoded[i] += self.tokenizer.encode('[PAD]') * (mx_len - len(encoded[i])) # [PAD] token

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

        if self.batch_mode:
            embeddings = self.get_embedding(embedder(batch[0], attention_mask=batch[1])[0])
        else:
            tmp_embeds = [self.get_embedding(embedder(el)[0]) for el in batch]
            embeddings = torch.cat(tmp_embeds)

        if self.float_mode == 'fp16':
            return embeddings.half()
        else:
            return embeddings

    def load_from(self, cache_dir):
        self.tokenizer = BertTokenizer.from_pretrained(cache_dir)
        if 'qembedder' in self.models:
            self.qembedder = BertModel.from_pretrained(f'{cache_dir}qembedder/')
        if 'aembedder' in self.models:
            self.aembedder = BertModel.from_pretrained(f'{cache_dir}aembedder/')

    def __init__(self, lang, bert_type, float_mode='fp32', \
            models=['aembedder', 'qembedder'], cache_dir=None, version='???'):
        super().__init__()

        self.lang = lang
        self.bert_type = bert_type
        self.models = models
        self.version = version
        print(cache_dir, lang, bert_type)
        if lang == 'en':
            if cache_dir is None:
                cache_dir = f'../pretrained-{bert_type}/'
                self.tokenizer = BertTokenizer.from_pretrained(bert_type, cache_dir=cache_dir)
                if 'qembedder' in models:
                    self.qembedder = BertModel.from_pretrained(bert_type, cache_dir=cache_dir)
                if 'aembedder' in models:
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
            if 'qembedder' in models:
                self.qembedder.half()
                gc.collect()
            if 'aembedder' in models:
                self.aembedder.half()
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
        if 'qembedder' in'LASERembedder':
            self.qembedder.save_pretrained(qname)
        if 'aembedder' in self.models:
            self.aembedder.save_pretrained(aname)
        self.tokenizer.save_pretrained(folder)
        config = {'float_mode': self.float_mode, 'name': self.__name__, \
                 'lang': self.lang, 'bert_type': self.bert_type, \
                 'models': self.models, 'version': self.version}
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

    def has_qembedder(self):
        return 'qembedder' in self.models

    def has_aembedder(self):
        return 'aembedder' in self.models

    def train(self):
        if self.has_qembedder():
            self.qembedder.train()
        if self.has_aembedder():
            self.aembedder.train()
        self.batch_mode = False

    def eval(self):
        if self.has_qembedder():
            self.qembedder.eval()
        if self.has_aembedder():
            self.aembedder.eval()
        #if self.lang == 'ru':
        #    self.batch_mode = False
        #else:
        self.batch_mode = True

class LASERprotoembedder(torch.nn.Module):
    max_seq_len = 1791791791
    __name__ = "LASERprotoembedder"
    def __init__(self, **kwargs):
        super().__init__()
        from laserembeddings import Laser
        import pickle

        self.laser = Laser()
        self.lang = kwargs['lang']
        self.version = kwargs['version']
        if 'name' not in kwargs:
            kwargs['name'] = self.__name__
        self._config = kwargs

    def save_to(self, folder):
        if folder[-1] != '/':
            folder = folder + '/'
        os.system(f'mkdir "{folder}"')
        with open(f'{folder}embedder', 'wb') as f:
            pickle.dump(self.embedder, f)
        with open(f'{folder}model_config.json', 'w') as file:
            json.dump(self._config, file)

    def eval(self):
        self.embedder.eval()

    def train(self):
        self.embedder.train()

    def add_embedder(self, embedder):
        self.embedder = embedder

    def forward(self, batch):
        quests, answs = batch.quests, batch.answs
        assert(len(quests) == len(answs))
        return (self.qembedd(quests), self.aembedd(answs))

    def qembedd(self, quests):
        device = next(self.embedder.parameters()).device
        laser = torch.tensor(self.laser.embed_sentences(quests, lang=self.lang), device=device)
        res = self.embedder(laser)
        return res # .view(-1, 1024, 1,)

    aembedd = qembedd
    tokenizer = str

class LASERembedder(LASERprotoembedder):
    __name__ = 'LASERembedder'

    def __init__(self, lang, cache_dir=None, lay_num=5, version='???'):
        super().__init__(lang=lang, version=version, lay_num=lay_num, name=self.__name__)

        if cache_dir is None:
            modules = []
            for i in range(1, 1 + lay_num):
                modules.append(torch.nn.Linear(1024, 1024))
                if i != lay_num:
                    #modules.append(torch.nn.BatchNorm1d(1024))
                    modules.append(torch.nn.LayerNorm(1024))
                    modules.append(torch.nn.ReLU())
                    modules.append(torch.nn.Dropout(0.1))
            self.add_embedder(torch.nn.Sequential(*modules))
        else:
            self.add_embedder(pickle.load(open(f'{cache_dir}embedder', 'rb')))

class TransformerInputReshaper(torch.nn.Module):
    """
    (batch_size, laser_embed) --> (1, batch_size, laser_embed)
    """
    def __init__(self):
        super().__init__()

    def forward(self, data):
        return data.view(1, -1, data.shape[-1])

class TransformerOutputReshaper(torch.nn.Module):
    """
    (1, batch_size, laser_embed) --> (batch_size, laser_embed)
    """
    def __init__(self):
        super().__init__()

    def forward(self, data):
        return data[0]

class LASERtransformer_embedder(LASERprotoembedder):
    __name__ = 'LASERtransformer_embedder'

    def __init__(self, lang, cache_dir=None, lay_num=3, version='???'):
        super().__init__(lang=lang, version=version, lay_num=lay_num, name=self.__name__)

        if cache_dir is None:
            net = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=1024,
                                                                                nhead=8,
                                                                                dim_feedforward=1024),
                                            lay_num)
            modules = [TransformerInputReshaper(), net, TransformerOutputReshaper()]
            self.add_embedder(torch.nn.Sequential(*modules))
        else:
            self.add_embedder(pickle.load(open(f'{cache_dir}embedder', 'rb')))


class USEncoder(torch.nn.Module): # In progress
    def __init__(self, float_mode='fp32', lang='en'):
        import tensorflow as tf
        import tensorflow_hub as hub
        import sentencepiece as spm

        self.embedder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")
        self.qembedder = torch.nn.Sequntial([torch.nn.Linear(300, 300), \
                                            torch.nn.ReLU()])
        self.aembedder = torch.nn.Sequntial([torch.nn.Linear(300, 300), \
                                            torch.nn.ReLU()])
        with tf.Session() as sess:
            spm_path = sess.run(module(signature="spm_path"))

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_path)
        print("SentencePiece model loaded at {}.".format(spm_path))


        self.input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
        self.embeddings = module(
            inputs=dict(
                values=self.input_placeholder.values,
                indices=self.input_placeholder.indices,
                dense_shape=self.input_placeholder.dense_shape))

        # Reduce logging output.
        tf.logging.set_verbosity(tf.logging.ERROR)

    def process_to_IDs_in_sparse_format(self, sentences):
        # An utility method that processes sentences with the sentence piece processor
        # 'sp' and returns the results in tf.SparseTensor-similar format:
        # (values, indices, dense_shape)
        ids = [self.sp.EncodeAsIds(x) for x in sentences]
        max_len = max(len(x) for x in ids)
        dense_shape=(len(ids), max_len)
        values=[item for sublist in ids for item in sublist]
        indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
        return (values, indices, dense_shape)

    def tf_encode(self, text):
        values, indices, dense_shape = self.process_to_IDs_in_sparse_format(text)

        with tf.Session() as session:
          session.run([tf.global_variables_initializer(), tf.tables_initializer()])
          message_embeddings = session.run(
              self.encodings,
              feed_dict={self.input_placeholder.values: values,
                        self.input_placeholder.indices: indices,
                        self.input_placeholder.dense_shape: dense_shape})

          '''for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
            print("Message: {}".format(messages[i]))
            print("Embedding size: {}".format(len(message_embedding)))
            message_embedding_snippet = ", ".join(
                (str(x) for x in message_embedding[:3]))
            print("Embedding: [{}, ...]\n".format(message_embedding_snippet))'''

        return message_embeddings

    def forward(self, batch):
        device = next(self.qembedder.parameters()).device
        quests, answs = batch.quests, batch.answs
        qembedd = self.tf_encode(quests)
        aembedd = self.tf_encode(answs)

        print('qembedd', qembedd.shape)

        qembeddings = self.qembedder(qembedd)
        aembeddings = self.aembedder(aembedd)

        return (torch.tensor(qembeddings, device=device), torch.tensor(aembeddings, device=device))
