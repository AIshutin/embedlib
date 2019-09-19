import embedlib
from knn_metric import get_mean_knn_metric
from mem_test import get_model_mem
from speed_test import get_model_perfomance
import sys
import json
import torch

model_folder = sys.argv[1]
if model_folder[-1] != '/':
    model_folder = model_folder + '/'
model_config = json.load(open(f'{model_folder}model_config.json'))
model_version = model_config['version']

embedder = embedlib.Embedder(model_version, model_folder)
device = torch.device('cpu')#'cuda:0') # 'cuda:0' if torch.cuda.is_available() else

tests = [("Metric score {:9.3f}", get_mean_knn_metric, (embedder,)),
         ("Processing speed {:7.3f} its/sec", get_model_perfomance, (embedder,)),
         ("Memory usage {:9.3f}Mb", get_model_mem, (model_folder,))]

for (id, test) in enumerate(tests):
    res = test[1](*test[-1])
    print(test[0].format(res))
