from utils import load_model
from utils import mem_report
import time
import gc

model = load_model('checkpoints/the_first_weights')
gc.collect()
mem_report()
time.sleep(111)
