from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
try:
    from apex.optimizers import FP16_Optimizer
    from apex.optimizers import FusedAdam
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
    # sudo pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

class WarmupLinearLRWatcher:
    def __init__(self, num_train_optimization_steps, num_warmup_steps, learning_rate, warmup):
        self.lr = learning_rate
        self.warmup = warmup
        self.total = num_train_optimization_steps
        self.total_w = num_warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def get_lr(self):
        if self.current_step < self.total_w:
            return self.lr * (self.current_step + 1) / self.total_w
        elif self.current_step == self.total_w:
            return self.lr
        else:
            total_decay = self.total - self.total_w
            return self.lr * (self.total - self.current_step) / (total_decay)

class BERTLikeOptimizer: # (torch.optim.Optimizer):
    # https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD example
    def __init__(self, model, num_train_optimization_steps, num_warmup_steps, learning_rate=5e-5, warmup=0.1):
        # https://github.com/huggingface/pytorch-transformers/blob/examples/examples/run_bert_squad.py for help with fp16
        self.float_mode = model.float_mode
        self.learning_rate = learning_rate
        self.global_step = 0
        self.warmup = warmup
        self.total_steps = num_train_optimization_steps

        '''parameters = []
        for el in model.qembedder.parameters():
            parameters += [el]
        for el in model.aembedder.parameters():
            parameters += [el]
        self.parameters = model.qembedder.parameters()
        parameters = model.qembedder.parameters()'''


        param_optimizer = list(model.qembedder.named_parameters())
        param_optimizer += list(model.aembedder.named_parameters())

        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        self.params = param_optimizer

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        if self.float_mode != 'fp16':
            self.optim = AdamW(parameters, lr=learning_rate, correct_bias=False)
            self.scheduler = WarmupLinearSchedule(self.optim, warmup_steps=num_warmup_steps, \
                                            t_total=num_train_optimization_steps,\
                                            warmup=warmup)
        else:
            self.optim = FusedAdam(optimizer_grouped_parameters,
                                  lr=learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            self.optim = FP16_Optimizer(self.optim, dynamic_loss_scale=True)

            self.scheduler = WarmupLinearLRWatcher(num_train_optimization_steps=num_train_optimization_steps,\
                                                num_warmup_steps=num_warmup_steps,\
                                                learning_rate=learning_rate,\
                                                warmup=warmup)

    def step(self, loss, closure=None):
        # loss should be not backwarded
        assert(closure is None)
        if self.float_mode == 'fp16':
            self.optim.backward(loss)
            # modify learning rate with special warm up BERT uses
            # if args.fp16 is False, BertAdam is used that handles this automatically
            lr_this_step = self.scheduler.get_lr()
            assert(lr_this_step != 0)

            for el in self.params:
                if el[1].grad is None:
                    print(el[0], type(el[1].grad))
                    raise Exception('grad is None')

            for el in self.optim.optimizer.param_groups:
                param_group['lr'] = lr_this_step
                assert(param_group['params'][0].grad is not None)

            for param_group in self.optim.param_groups:
                param_group['lr'] = lr_this_step
                assert(param_group['params'][0].grad is not None)
        else:
            loss.backward()

        self.scheduler.step()
        self.optim.step()
        self.global_step += 1

    def zero_grad(self):
        self.optim.zero_grad()
