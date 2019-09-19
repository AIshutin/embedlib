from embedlib.utils import load_model
import torch
import sys
device = torch.device('cpu')
model = load_model(sys.argv[1]).to(device)
model.save_to(sys.argv[2])
