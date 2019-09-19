import docker
import time
import os
import shutil

client = docker.from_env()
embedlib_source_path = '/home/aishutin/DL/qa-ml/'
waittime = 20 # in seconds

memory_checker_py = "from embedlib.utils import load_model\n" \
                    "from embedlib.utils import mem_report\n" \
                    "import time\n" \
                    "import gc\n" \
                    "model = load_model('/app/model')\n" \
                    "gc.collect()\n" \
                    "mem_report()\n" \
                    "time.sleep(35)\n"

dockerfile = "FROM python:3.7\n" \
              "RUN pip3 install torch torchvision\n" \
              "RUN pip3 install numpy\n" \
              "RUN pip3 install laserembeddings\n" \
              "RUN python3 -m laserembeddings download-models\n" \
              "\n" \
              "COPY embedlib /app/embedlib/embedlib\n" \
              "COPY setup.py /app/embedlib/\n" \
              "RUN pip3 install /app/embedlib/\n" \
              "COPY model /app/model/\n" \
              "COPY memory_checker.py /app/memory_checker.py\n" \
              'CMD ["python3", "/app/memory_checker.py"]\n'

docker_dir = '/tmp/automemtest_via_docker/'

def compose_container(model_folder, docker_folder=docker_dir):
    name = "automem"
    try:
        shutil.rmtree(docker_folder)
    except FileNotFoundError as exp:
        pass

    os.mkdir(docker_folder)
    shutil.copytree(model_folder, f'{docker_folder}model')
    shutil.copyfile(f'{embedlib_source_path}setup.py', f'{docker_folder}setup.py')
    shutil.copytree(f'{embedlib_source_path}embedlib/', f'{docker_folder}embedlib')
    with open(f'{docker_folder}memory_checker.py', 'w') as file:
        print(memory_checker_py, file=file)
    with open(f'{docker_folder}Dockerfile', 'w') as file:
        print(dockerfile, file=file)
    client.images.build(path=docker_folder, tag=name)
    return name

def get_model_stats(cname, waittime=waittime, mem_only=True):
    container = client.containers.run(cname, detach=True)
    #print(container)
    #print(container.id)
    time.sleep(waittime)
    stats = client.containers.get(container.id).stats(stream=False)
    if mem_only:
        return stats["memory_stats"]["usage"]
    else:
        return stats

def get_model_mem(model_folder, waittime=waittime):
    cname = compose_container(model_folder)
    return get_model_stats(cname) / 1024 / 1024


if __name__ == '__main__':
    #compose_container('../10lays-laser/', 'autodocker/')
    print(get_model_stats("automem") / 1024 / 1024)
