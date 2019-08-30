import torch
import torch.nn.functional as F
from .utils import load_model
from .utils import mem_report
import numpy
import gc
import time

class Embedder:
    """Takes model type and path to checkpoint dir and creates a functor that can return numpy embeddings."""

    def __init__(self, model, path):
        self.model = load_model(f'{path}')
        self.model.eval()

    def normalize(self, vectors):
        return F.normalize(vectors, dim=1)

    def __call__(self, sentence, is_question=True):
        if type(sentence) != type([]):
            batch = [sentence]
        else:
            batch = sentence

        if is_question:
            return self.normalize(self.model.qembedd(batch)).detach().cpu().numpy()
        else:
            return self.normalize(self.model.aembedd(batch)).detach().cpu().numpy()

def similarity(a, b):
    #print(a.shape)
    #print(b.shape)
    return numpy.dot(a, numpy.transpose(b))

if __name__ == '__main__':
    model = Embedder(None, '../ru-2-attentions')#'checkpoints/the_first_weights')
    #model.model.to(torch.device('cpu'))
    gc.collect()
    assert(mem_report() <= 600)
    time.sleep(120)
    model('I love cats')
