from models import BERTLike
from utils import load_model
from utils import mem_report
import json
import torch
import time

def chop_last_attention_layer(tower, num=1):
    tower.encoder.layer = tower.encoder.layer[:-num]
    print(tower.encoder.layer)
    return tower

model = BERTLike(bert_type='bert-base-uncased', lang='en', float_mode='fp16')
num_last_lays = 1
model.qembedder = chop_last_attention_layer(model.qembedder, num_last_lays)
model.aembedder = chop_last_attention_layer(model.aembedder, num_last_lays)

dirname = '../11-attentions/'
model.save_to(dirname)

def decrease_attention_lays_in_config(dirname, num=num_last_lays):
    states = json.load(open(f'{dirname}qembedder/config.json'))
    states['num_hidden_layers'] -= num
    #states['num_attention_heads'] -= 1
    json.dump(states, open(f'{dirname}qembedder/config.json', 'w'))
    json.dump(states, open(f'{dirname}aembedder/config.json', 'w'))

decrease_attention_lays_in_config(dirname, num_last_lays)
del model
model = load_model(dirname)
#device = torch.device('cuda:0')
#model.to(device)
mem_report()
time.sleep(120)
