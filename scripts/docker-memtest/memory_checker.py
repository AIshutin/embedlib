from embedlib.utils import load_model
from embedlib.utils import mem_report
import time
import gc

model = load_model('/app/model')
gc.collect()
mem_report()
time.sleep(120)
