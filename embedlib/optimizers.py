from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from torch.optim import Adam

class LASERprotoembedderOptimizer:
    def __init__(self, model, num_train_optimization_steps, num_warmup_steps, learning_rate=1e-4, warmup=0.1):
        self.optim = Adam(model.embedder.parameters(), lr=learning_rate)

    def step(self):
        self.optim.step()
        pass

    def zero_grad(self):
        self.optim.zero_grad()
        pass

class LASERembedderOptimizer(LASERprotoembedderOptimizer):
    pass

class LASERtransformer_embedderOptimizer(LASERprotoembedderOptimizer):
    pass

class BERTLikeOptimizer: # (torch.optim.Optimizer):
    # https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD example
    def __init__(self, model, num_train_optimization_steps, num_warmup_steps, learning_rate=5e-5, warmup=0.1):
        self.qoptim = AdamW(model.qembedder.parameters(), lr=learning_rate, correct_bias=False)
        self.qscheduler = WarmupLinearSchedule(self.qoptim, warmup_steps=num_warmup_steps, \
                                        t_total=num_train_optimization_steps)

        self.aoptim = AdamW(model.aembedder.parameters(), lr=learning_rate, correct_bias=False)
        self.ascheduler = WarmupLinearSchedule(self.aoptim, warmup_steps=num_warmup_steps, \
                                        t_total=num_train_optimization_steps)

     #def __setstate__(self, state):
     # super().__setstate__(state)

    def step(self, closure=None):
        assert(closure is None)

        self.qoptim.step()
        self.aoptim.step()
        self.qscheduler.step()
        self.ascheduler.step()

    def zero_grad(self):
        self.qoptim.zero_grad()
        self.aoptim.zero_grad()

class USEncoderOptimizer:
    def __init__(self, model, learning_rate=1e-4, **kwarg):
        self.qoptim = Adam(model.qembedder.parameters(),\
                                        lr=learning_rate)
        self.aoptim = Adam(model.aembedder.parameters(),\
                                        lr=learning_rate)

    def step(self, closure=None):
        assert(closure is None)

        self.qoptim.step()
        self.aoptim.step()

    def zero_grad(self):
        self.qoptim.zero_grad()
        self.aoptim.zero_grad()
