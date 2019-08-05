from utils import load_model
from datasets import TwittCorpus, collate_wrapper
from torch.utils.data import Dataset, DataLoader, Subset
from utils import print_batch, embed_batch, prepare_batch
from metrics import calc_mrr
import sys
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if len(sys.argv) >= 2:
    checkpoint = sys.argv[1]
else:
    checkpoint = 'checkpoints/epoch: 2 calc_mrr:   0.9948 hinge_loss:   3.7534/'

(qembedder, aembedder), tokenizer = load_model(checkpoint)

qembedder.to(device)
aembedder.to(device)

max_dataset_size = int(1e5)
batch_size = 16
float_mode = 'fp32'

full_dataset= TwittCorpus(tokenizer, max_dataset_size + 100)
print(f"full: {len(full_dataset)}")
data = Subset(full_dataset, range(max_dataset_size, max_dataset_size + 100))
print(f"subset: {len(data)}")
loader = DataLoader(data, batch_size=batch_size, collate_fn=collate_wrapper)

for batch in loader:
    print_batch(batch, tokenizer)
    embeddings = embed_batch(prepare_batch(batch, device), qembedder, aembedder, float_mode)
    calc_mrr(embeddings[0], embeddings[1], False)

    print()
    print()
