# To install cuda docker
wget -O - -q 'https://gist.githubusercontent.com/allenday/c875eaf21a2b416f6478c0a48e428f6a/raw/f7feca1acc1a992afa84f347394fd7e4bfac2599/install-docker-ce.sh' | sudo bash
wget https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i nvidia-docker*.deb
sudo nvidia-docker-plugin &
sudo nvidia-docker run --rm nvidia/cuda nvidia-smi


#wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz
#gunzip -c openmpi-4.0.0.tar.gz | tar xf -
#cd openmpi-4.0.0 && ./configure --prefix=/usr/local && make all install
pip3 install --no-cache-dir horovod
