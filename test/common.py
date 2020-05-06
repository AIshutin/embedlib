import os
import logging
import pytest

def run_train(traindir='../scripts/', **kwargs):
    print(traindir)
    command = [f'cd {traindir} && python3 train.py with']
    for key in kwargs:
        command.append(f'{key}={kwargs[key]}')
    command = ' '.join(command)
    logging.info(command)
    print(command)
    code = os.system(command)
    assert(code == 0)
