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

def print_batch(batch, tokenizer):
    quests, answs = batch.quests, batch.answs

    for i in range(len(quests)):
        print(i, tokenizer.decode(quests[i]))
        print(">>> ", tokenizer.decode(answs[i]))
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
