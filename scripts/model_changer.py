from embedlib.models import BERTLike
from embedlib.utils import load_model
from embedlib.utils import mem_report
import json
import torch
import time
import gc

def chop_last_attention_layer(tower, num=1):
    tower.encoder.layer = tower.encoder.layer[:-num]
    print(tower.encoder.layer)
    return tower

prev_model_dir =  "../rubert-base-uncased"
if prev_model_dir is None:
    model = BERTLike(bert_type='bert-base-uncased', lang='en', float_mode='fp32')
else:
    model = load_model(prev_model_dir)

gc.collect()
mem_report()

num_last_lays = 6
model.qembedder = chop_last_attention_layer(model.qembedder, num_last_lays)
model.aembedder = chop_last_attention_layer(model.aembedder, num_last_lays)

dirname = f'../ru-{12 - num_last_lays}-attentions/'
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
gc.collect()
mem_report()
#time.sleep(120)
