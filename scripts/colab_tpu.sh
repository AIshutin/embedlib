#!/bin/bash
DIST_BUCKET="gs://tpu-pytorch/wheels"
TORCH_WHEEL="torch-1.15-cp36-cp36m-linux_x86_64.whl"
TORCH_XLA_WHEEL="torch_xla-1.15-cp36-cp36m-linux_x86_64.whl"
TORCHVISION_WHEEL="torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl"
# Install Colab TPU compat PyTorch/TPU wheels and dependencies
pip uninstall -y torch torchvision
gsutil cp "$DIST_BUCKET/$TORCH_WHEEL" .
gsutil cp "$DIST_BUCKET/$TORCH_XLA_WHEEL" .
gsutil cp "$DIST_BUCKET/$TORCHVISION_WHEEL" .
pip install "$TORCH_WHEEL"
pip install "$TORCH_XLA_WHEEL"
pip install "$TORCHVISION_WHEEL"
sudo apt-get install libomp5
pip install sacred
pip install tensorboardX
pip install pymongo dnspython

# Cerebra internal code
echo "Installing embedlib"
pip install libs/
echo "Training"
cd scripts && python3 train.py with tpu=True
