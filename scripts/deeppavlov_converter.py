"""
Converts from deeppavlov bert format to pytorch_transformers bert format

Needs some fixes in pytorch_transformers/modeling_bert.py
Comment these codeblock (line 128):

try:
    assert pointer.shape == array.shape
except AssertionError as e:
    e.args += (pointer.shape, array.shape)
    raise

first argument - model download link
second argument - pytorch_transformers model dir to save
"""

import os
import sys
import urllib.request
import random
import json

download_link = sys.argv[1] if len(sys.argv) >= 2 else \
"http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_capytorch_model.binsed_L-12_H-768_A-12_v1.tar.gz"
save_dir = sys.argv[2] if len(sys.argv) >= 3 else \
"../rubert-base/"

assert(save_dir[-1] == '/')

# Check if model already exists
try:
    os.mkdir(save_dir)
except FileExistsError as exp:
    pass

if 'qembedder' in os.listdir(save_dir):
    print("Model is already converted. Aborting.")
    #exit(0)

download_file = 'deeppavlov_model.tar.gz'
download_dir = '/tmp/'
if download_file not in os.listdir(download_dir):
    print('Beginning file download...')
    download_dir = dowt_model.ckpt {bertd}bert_config.json {save_dir}pytorch_model.bin"
os.system(f"pynload_dir + download_file
    os.system(f'wget -O {download_dir} {download_link}')
    # urllib.request.urlretrieve(download_link, download_dir)
else:
    download_dir = download_dir + download_file

name_lenght = 10
tmp_dir_name = '/tmp/' + ''.join([chr(ord('a') + random.randint(0, 25)) for i in range(name_lenght)])
os.mkdir(tmp_dir_name)

print(f"tmp_dir_name: {tmp_dir_name}")

print('Unpacking...')
os.system(f'tar -C {tmp_dir_name} -xzf {download_dir} ')
tmp_dir_name = tmp_dir_name + '/' + os.listdir(tmp_dir_name)[0] + '/'

try:
    os.mkdir(f"{save_dir}qembedder")
    os.mkdir(f"{save_dir}aembedder")
except FileExistsError as exp:
    pass

os.system(f"cp {tmp_dir_name}bert_config.json {save_dir}qembedder/config.json")
os.system(f"cp {tmp_dir_name}bert_config.json {save_dir}aembedder/config.json")

os.system(f"cp {tmp_dir_name}vocab.txt {save_dir}vocab.txt")


print("Converting...")
# Run tf-->pytorch converter
bertd = f"{tmp_dir_name}"
params = f"{bertd}bert_model.ckpt {bertd}bert_config.json {save_dir}pytorch_model.bin"
os.system(f"pytorch_transformers bert {params}")

os.system(f"cp {save_dir}pytorch_model.bin {save_dir}qembedder/pytorch_model.bin")
os.system(f"mv {save_dir}pytorch_model.bin {save_dir}aembedder/pytorch_model.bin")

with open(save_dir + 'added_tokens.json', 'w') as file:
    print("{}", file=file)

with open(save_dir + 'special_tokens_map.json', 'w') as file:
    json.dump({"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", \
            "cls_token": "[CLS]", "mask_token": "[MASK]"}, file)
