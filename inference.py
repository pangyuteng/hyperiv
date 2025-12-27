import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_util import OptionDataset
from hyperiv_util import SetEmbeddingNetwork, HyperNetwork
from trainer_util import trainer
import torch.optim as optim


iv_network = torch.nn.Sequential(
    torch.nn.Linear(2, 16),
    torch.nn.Tanh(),
    torch.nn.Linear(16, 16),
    torch.nn.Tanh(),
    torch.nn.Linear(16, 1),
    torch.nn.Softplus()
)

n_params = sum([p.numel() for p in iv_network.parameters()])

input_dim = 3
output_dim = n_params
print(input_dim,output_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

hyper_model = SetEmbeddingNetwork(input_dim, output_dim).to(device)

pth_file = "spx_hyperiv.pth"
hyper_model.load_state_dict(torch.load(pth_file,weights_only=True))
hyper_model.eval()

x = np.random.rand(128,3)
x = torch.from_numpy(x).to(device).float()
out = hyper_model(x)
print(out.shape)

#torch.save(hyper_model.state_dict(), pth_file)
