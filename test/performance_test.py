import os
import shutil
import docker
import random

def test_me(tmpdir, max_dataset_size, model_name, batch_size=32):
    print(tmpdir)
    shutil.copyfile('requirements.txt', os.path.join(tmpdir, 'requirements.txt'))
    shutil.copyfile('setup.py', os.path.join(tmpdir, 'setup.py'))
    shutil.copytree('embedlib', os.path.join(tmpdir, 'embedlib'))
    shutil.copyfile('test/text_samples.json', os.path.join(tmpdir, 'text_samples.json'))
    shutil.copyfile('test/performance_docker', os.path.join(tmpdir, 'Dockerfile'))
    print('----')
    client = docker.from_env()
    tag = str(random.randint(1, 1000000000))
    result = client.images.build(path=tmpdir,
                                tag=tag,
                                buildargs={'_BATCH_SIZE': batch_size})
    output = client.containers.run(tag)
    print(output)
