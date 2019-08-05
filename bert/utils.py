import re
import os
import torch
from pytorch_transformers import BertTokenizer, BertForQuestionAnswering
from pytorch_transformers import BertModel, BertConfig
import json
import models

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
    if checkpoint_dir[-1] != '/':
        checkpoint_dir = checkpoint_dir + '/'
    print(f"loading model from {checkpoint_dir}")
    config = json.load(open(f'{checkpoint_dir}model_config.json'))
    name = config['name']
    config.pop('name')
    config['cache_dir'] = checkpoint_dir
    print(f"CONFIG {config}")
    return getattr(models, name)(**config)
