import os
import shutil
import docker
import random

def parse_time_logs(data):
    array = []
    for line in data.split('\n'):
        if line == '':
            continue
        num = float(line.rstrip().split()[-1])
        array.append(num)
    mean_inference_time = sum(array) / len(array)
    return mean_inference_time

def parse_memory_logs(data):
    array = []
    for line in data.split('\n')[2:]:
        if line == '':
            continue
        _, mem, _ = line.split()
        array.append(float(mem))
    array = array[int(len(array)/4):] # skip irrelevant init memory consumption
    mean_memory_consumption = sum(array) / len(array)
    return mean_memory_consumption

def docker_measuring(tmpdir, batch_size=32):
    tmpdir = '/home/aishutin/docker23'
    os.mkdir(tmpdir)
    print(tmpdir)
    shutil.copyfile('requirements.txt', os.path.join(tmpdir, 'requirements.txt'))
    shutil.copyfile('setup.py', os.path.join(tmpdir, 'setup.py'))
    shutil.copyfile('test/memory_checker.py', os.path.join(tmpdir, 'memory_checker.py'))
    shutil.copytree('embedlib', os.path.join(tmpdir, 'embedlib'))
    shutil.copyfile('test/text_samples.json', os.path.join(tmpdir, 'text_samples.json'))
    shutil.copyfile('test/performance_docker', os.path.join(tmpdir, 'Dockerfile'))
    shutil.copytree('../the_first_weights', os.path.join(tmpdir, 'model'))
    client = docker.from_env()
    tag = 'embedlib-tests' # str(random.randint(1, 1000000000))
    build_log = client.images.build(path=str(tmpdir),
                        tag=tag,
                        buildargs={'_BATCH_SIZE': str(batch_size)},
                        quiet=False)
    print('build_log', build_log)
    output = client.containers.run(tag)
    output = output.decode('utf-8')
    _, time_logs, memory_logs = output.split('-' * 5)
    time_val = parse_time_logs(time_logs)
    memory_val = parse_memory_logs(memory_logs)
    return time_val, memory_val
