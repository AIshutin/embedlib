import re
import os
import torch
from pytorch_transformers import BertTokenizer, BertForQuestionAnswering
from pytorch_transformers import BertModel, BertConfig
import json
from . import models
import gc

def remove_urls (vTEXT):
    # r'http\S+'
    # r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b'
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '[link]', vTEXT, flags=re.MULTILINE)
    return vTEXT

def print_batch(batch):
    quests, answs = batch.quests, batch.answs

    for i in range(len(quests)):
        print(i, quests[i])
        print(">>> ", answs[i])
    print()

def load_model(checkpoint_dir):
    if checkpoint_dir[-1] != '/':
        checkpoint_dir = checkpoint_dir + '/'
    #print(f"loading model from {checkpoint_dir}")
    config = json.load(open(checkpoint_dir + 'model_config.json'))
    if 'name' in config:
        name = config['name']
        config.pop('name')
    else:
        name = config['__name__']
        config.pop('__name__')
    config['cache_dir'] = checkpoint_dir
    #print(f"CONFIG {config}", name)
    return getattr(models, name)(**config)

## MEM utils ##
def mem_report():
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        print('Storage on %s' %(mem_type))
        print('-'*LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print('%s\t\t%s\t\t%.2f' % (
                element_type,
                size,
                mem) )
        print('-'*LEN)
        print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem) )
        print('-'*LEN)
        return total_mem

    LEN = 65
    print('='*LEN)
    objects = gc.get_objects()
    print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    res = 0
    res += _mem_report(cuda_tensors, 'GPU')
    res += _mem_report(host_tensors, 'CPU')
    print('='*LEN)
    return res
