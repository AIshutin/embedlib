from utils import mem_report
import torch
import time
import gc
device = torch.device('cpu')
tensor = torch.randn(1024, 1024, device=device)
mem_report()
half = tensor.half()
del tensor
gc.collect() 
mem_report()
