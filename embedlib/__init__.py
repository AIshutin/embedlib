import torch
from utils import load_model
from utils import mem_report
import gc
import time

class Embedder:
    """Takes model type and path to checkpoint dir and creates a functor that can return numpy embeddings."""

    def __init__(self, model, path):
        self.model = load_model(f'{path}')

    def __call__(self, sentence, is_question=True):
        if type(sentence) != type([]):
            batch = [sentence]
        else:
            batch = sentence

        if is_question:
            return self.model.qembedd(batch)
        else:
            return self.model.aembedd(batch)

if __name__ == '__main__':
    model = Embedder(None, 'checkpoints/the_first_weights')
    model.model.to(torch.device('cpu'))
    gc.collect()
    assert(mem_report() <= 600)
    model('I love cats')
    time.sleep(120)
