import torch
import torch.nn.functional as F
from pytorch_transformers import BertModel, BertTokenizer

class Embedder:
    """Takes model type and path to checkpoint dir and creates a functor that can return numpy embeddings."""

    def __init__(self, model, path):
        assert model == 'bert-base-en'

        self.model = BertModel.from_pretrained(f'{path}/qembedder')
        self.tokenizer = BertTokenizer.from_pretrained(path)

    def __call__(self, sentence):
        input_ids = torch.tensor([self.tokenizer.encode(sentence)])
        with torch.no_grad():
            embedding = self.model(input_ids)[0]
            embedding = torch.sum(embedding, dim=1)[0]
            embedding = F.normalize(embedding, dim=0)
            return embedding

