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

def load_model(model_pth_file):


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

    model.load_state_dict(torch.load(model_pth_file,weights_only=True))
    model.eval()
    return model

if __name__ == "__main__":
    #h5_file = "/mnt/hd2bak/scratch/spx_w_ref.h5"
    #raise ValueError()
    model_pth_file = './workdir/spx_model_00001.pth'
    h5_file = "/mnt/hd2bak/scratch/spx_w_ref_sm.h5"

    model = load_model(model_pth_file)
    # a few points
    z = np.random.rand(1,8,3)
    z = torch.from_numpy(z).to(device).float()
    # grid
    x = np.random.rand(1,40,2)
    x = torch.from_numpy(x).to(device).float()
    y_pred = model(z, x).squeeze(-1)
    y_pred = y_pred.cpu().detach().numpy()
    print(y_pred.shape)

    df = pd.read_hdf(h5_file, 'df')

    N = 20
    B = 128
    train_dates = df["date"].unique()
    train_dataset = OptionDataset(df, N=N, sample=True)
    train_dataloader = DataLoader(train_dataset, batch_size=B, shuffle=True, drop_last=True)
    for row_data in train_dataloader:
        z,x,y_true = row_data
        z = z.to(device).float()
        x = x.to(device).float()
        #print(z.shape,x.shape,y_true.shape)
        y_pred = model(z, x).squeeze(-1)
        y_pred = y_pred.cpu().detach().numpy().flatten()
        y_true = y_true.cpu().detach().numpy().flatten()
        print('y_pred',y_pred[:5],y_pred.shape)
        print('y_true',y_true[:5],y_true.shape)
        print(np.mean(np.square(y_pred-y_true)))
        print("----")
        