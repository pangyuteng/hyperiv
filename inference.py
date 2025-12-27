import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_util import OptionDataset
from hyperiv_util import SetEmbeddingNetwork, HyperNetwork
from trainer_util import trainer
import torch.optim as optim



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

iv_network = torch.nn.Sequential(
    torch.nn.Linear(2, 16),
    torch.nn.Tanh(),
    torch.nn.Linear(16, 16),
    torch.nn.Tanh(),
    torch.nn.Linear(16, 1),
    torch.nn.Softplus()
).to(device)
n_params = sum([p.numel() for p in iv_network.parameters()])

input_dim = 3
output_dim = n_params
print(input_dim,output_dim)


hyper_model = SetEmbeddingNetwork(input_dim, output_dim).to(device)

model = HyperNetwork(hyper_model, iv_network)

hyper_pth_file = "spx_hyper.pth"
iv_pth_file = "spx_iv.pth"
model_pth_file = 'spx_model.pth'

model.load_state_dict(torch.load(model_pth_file,weights_only=True))
model.eval()

z = np.random.rand(1,9,3)
z = torch.from_numpy(z).to(device).float()
x = np.random.rand(1,3,2)
x = torch.from_numpy(x).to(device).float()
y_pred = model(z, x).squeeze(-1)
print(y_pred.shape)
print(y_pred)