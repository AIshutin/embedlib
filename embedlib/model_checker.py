from utils import load_model
from datasets import CorpusData, collate_wrapper
from torch.utils.data import Dataset, DataLoader, Subset
from utils import print_batch
from metrics import calc_mrr
import sys
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if len(sys.argv) >= 2:
    checkpoint = sys.argv[1]
else:
    checkpoint = '../rubert-base-uncased'

model = load_model(checkpoint).to(device)

max_dataset_size = int(1e2)
batch_size = 16

full_dataset= CorpusData(['en-twitt-corpus'], model.tokenizer, max_dataset_size + 100) # 'ru-opendialog-corpus'
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
