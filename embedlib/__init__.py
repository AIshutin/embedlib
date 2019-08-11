import torch
from utils import load_model

class Embedder:
    """Takes model type and path to checkpoint dir and creates a functor that can return numpy embeddings."""

    def __init__(self, model, path):
        self.model = load_from(f'{path}')

    def __call__(self, sentence, is_question=True):
        if type(sentence) != type([]):
            batch = [sentence]
        else:
            batch = sentence

        if is_question:
            return self.qembedd(batch)
        else:
            return self.aembedd(batch)
