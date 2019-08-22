from utils import load_model
from datasets import CorpusData, collate_wrapper
from torch.utils.data import Dataset, DataLoader, Subset
from utils import print_batch
from metrics import calc_mrr
from utils import mem_report
import metrics
import sys
import torch
import time
import gc

device = torch.device('cpu') # 'cuda:0' if torch.cuda.is_available() else

if len(sys.argv) >= 2:
    checkpoint = sys.argv[1]
else:
    checkpoint = '../11-attentions'

model = load_model(checkpoint).to(device)

max_dataset_size = int(1e1)
batch_size = 16

dataset_name = 'ru-opendialog-corpus' # 'en-twitt-corpus' # 'ru-opendialog-corpus'
verbal = False

if verbal:
    full_dataset= CorpusData([dataset_name], model.tokenizer, max_dataset_size + 100)
    print(f"full: {len(full_dataset)}")
    data = Subset(full_dataset, range(max_dataset_size, max_dataset_size + 100))
    print(f"subset: {len(data)}")
    loader = DataLoader(data, batch_size=batch_size, collate_fn=collate_wrapper)

    for batch in loader:
        print_batch(batch)
        embeddings = model(batch)
        print(f'MRR: {calc_mrr(embeddings[0], embeddings[1], False)}')
        print()
        print()
else:
    full_dataset = CorpusData([dataset_name], model.tokenizer, max_dataset_size)
    gc.collect()
    mem_report()
    time.sleep(120)
    loader = DataLoader(full_dataset, batch_size=batch_size, collate_fn=collate_wrapper)
    print(metrics.get_mean_on_data([calc_mrr], loader, model))
    print(metrics.calc_random_mrr(batch_size))
