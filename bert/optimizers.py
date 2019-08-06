from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

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

        self.qscheduler.step()
        self.ascheduler.step()
        self.qoptim.step()
        self.aoptim.step()

    def zero_grad(self):
        self.qoptim.zero_grad()
        self.aoptim.zero_grad()
