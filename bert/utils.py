import re
import os
import torch
from pytorch_transformers import BertTokenizer, BertForQuestionAnswering
from pytorch_transformers import BertModel, BertConfig

def remove_urls (vTEXT):
    # r'http\S+'
    # r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b'
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '[link]', vTEXT, flags=re.MULTILINE)
    return vTEXT

def prepare_batch(batch, device):
    quests, answs = batch.quests, batch.answs

    assert(len(quests) == len(answs))

    quests = [torch.tensor([el], device=device) for el in quests]
    answs = [torch.tensor([el], device=device) for el in answs]

    return (quests, answs)

def get_embedding(embeddings):
    '''
    using bert-as-service-like strategy to get fixed-size vector
    1. considering only -1 layer (NOT -2 as in bert-as-service)
    2. "REDUCE_MEAN take the average of the hidden state of encoding layer on the time axis" @bert-as-service
    '''
    return torch.sum(embeddings, dim=1)

def embed_batch(batch, qembedder, aembedder, float_mode):
    quests, answs = batch

    tmp_quest = [get_embedding(qembedder(quests[i])[0]) for i in range(len(quests))]
    tmp_answ = [get_embedding(aembedder(answs[i])[0]) for i in range(len(answs))]

    qembeddings = torch.cat(tmp_quest)
    aembeddings = torch.cat(tmp_answ)

    assert(qembeddings.shape == aembeddings.shape)

    if float_mode == 'fp16':
        return (qembeddings.half(), aembeddings.half())

    return (qembeddings, aembeddings)

def print_batch(batch, tokenizer):
    quests, answs = batch.quests, batch.answs

    for i in range(len(quests)):
        print(i, tokenizer.decode(quests[i]))
        print('>>> ', tokenizer.decode(answs[i]))
    print()

def load_model(checkpoint_dir):
    print(checkpoint_dir)
    qembedder = BertModel.from_pretrained(checkpoint_dir + 'qembedder/')
    aembedder = BertModel.from_pretrained(checkpoint_dir + 'aembedder/')
    tokenizer = BertTokenizer.from_pretrained(checkpoint_dir)
    return (qembedder, aembedder), tokenizer

def save_model(model, tokenizer, checkpoint_dir):
    os.system(f'mkdir "{checkpoint_dir}"')
    qname = checkpoint_dir + 'qembedder/'
    aname = checkpoint_dir + 'aembedder/'
    os.system(f'mkdir "{qname}"')
    os.system(f'mkdir "{aname}"')
    model[0].save_pretrained(qname)
    model[1].save_pretrained(aname)
    tokenizer.save_pretrained(checkpoint_dir)
