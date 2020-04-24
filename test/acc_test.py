import embedlib
from common import run_train
import random
import logging
import os
import random
import shutil
import copy

DIR = os.getcwd()
if DIR[-1] != '/':
    DIR = DIR + '/'
DIR = DIR + 'scripts/checkpoints'

class TestMe:
    pass

class RunTracker:
    def __init__(self, **run_params):
        self.params = run_params

    def __call__(self, tmpdir, max_dataset_size, model_name):
        mypath = os.path.abspath(__file__)
        print(tmpdir)
        project_dir = os.path.split(os.path.split(mypath)[0])[0]
        print(f"project_dir is {project_dir}")
        run_train(traindir=f'{project_dir}/scripts/', checkpoint_dir=tmpdir,
                                                      max_dataset_size=max_dataset_size,
                                                      model_name=model_name,
                                                      **self.params)
        assert(len(os.listdir(tmpdir)) != 0)
        last_tmp = -1
        last_checkpoint = []
        for el in os.listdir(tmpdir):
            print(el)
            dirpath = f'{tmpdir}/{el}'
            for file in os.listdir(dirpath):
                path = f'{dirpath}/{file}'
                if not os.path.isfile(path):
                    continue
                print(path)
                tm = os.path.getmtime(path)
                if tm >= last_tmp:
                    last_checkpoint = dirpath
                    last_tmp = tm

        print(f"The last checkpoint is {last_checkpoint}")
        checkpoint_name = os.path.split(last_checkpoint)[-1]
        print(checkpoint_name)
        parts = checkpoint_name.split()
        not_empty = []
        for el in parts:
            if len(el) != 0:
                not_empty.append(el)
        score = None
        score_func = 'mrr'
        if 'metric_name' in self.params:
            score_func = self.params['metric_name']

        next = False
        for el in not_empty:
            if next:
                score = float(el)
                next = False

            if (score_func + ':') in el:
                next = True
                assert(score is None)
        return {score_func: score}


def base_func(self, request, max_dataset_size, model_name):
    name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    print(name)
    name = name[:name.rfind('[')]

    kwargs = {'tmpdir': DIR,
              'max_dataset_size': max_dataset_size,
              'model_name': model_name}
    full_name = name[len('test_'):] + f'|model_name={model_name}|max_dataset_size={max_dataset_size}'
    print(full_name)
    request.session.results[full_name] = getattr(self, name[5:])(**kwargs)

params = {
    'batch_size': (16, 32),
    'criterion_func': ('hinge_loss', 'triplet_loss'),
    'metric_name': ('mrr',),
    'multigpu': (False,)
}

def bruteforce_params(full_params=params, chosen=[]):
    if len(full_params) == 0:
        strings = [f"{key}={val}" for key, val in chosen]
        name = "|".join(strings)
        setattr(TestMe, f'test_{name}', base_func)
        dct_options = {}
        for el in chosen:
            dct_options[el[0]] = el[1]
        setattr(TestMe, name, RunTracker(**dct_options))
        return
    key = random.choice(list(full_params.keys()))
    values = full_params[key]
    full_params = copy.deepcopy(full_params)
    full_params.pop(key)
    for el in values:
        chosen.append((key, el))
        bruteforce_params(full_params, copy.deepcopy(chosen))
        chosen.pop(-1)
bruteforce_params()
