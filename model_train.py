import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_util import OptionDataset
from hyperiv_util import SetEmbeddingNetwork, HyperNetwork
from trainer_util import trainer
import torch.optim as optim
import datetime

train_split = '2024-10-08'
h5_file = "/mnt/hd2bak/scratch/spx_w_ref_sm.h5"
model_dir = "workdir"
num_epochs = 500
loss_csv_file = os.path.join(model_dir,'loss.csv')

if False: # for debugging
    train_split = "2025-12-24"
    h5_file = "spx_w_ref.h5"
    num_epochs = 5


os.makedirs(model_dir,exist_ok=True)
df = pd.read_hdf(h5_file, 'df')

N = 1024
B = 128

train_dates = df[df["date"] < train_split]["date"].unique()
train_dataset = OptionDataset(df[df["date"].isin(train_dates)], N=N, sample=True)
train_dataloader = DataLoader(train_dataset, batch_size=B, shuffle=True, drop_last=True)

test_dates = df[df["date"] >= train_split]["date"].unique()
test_dataset = OptionDataset(df[df["date"].isin(test_dates)], N=N, sample=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

iv_network = torch.nn.Sequential(
    torch.nn.Linear(2, 16),
    torch.nn.Tanh(),
    torch.nn.Linear(16, 16),
    torch.nn.Tanh(),
    torch.nn.Linear(16, 1),
    torch.nn.Softplus()
)

n_params = sum([p.numel() for p in iv_network.parameters()])
print(n_params)

input_dim = 3
output_dim = n_params

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

hyper_model = SetEmbeddingNetwork(input_dim, output_dim).to(device)

model = HyperNetwork(hyper_model, iv_network)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5)

mylist = []
for epoch in range(num_epochs):
    train_loss_mse, train_loss_mae, train_loss_cal, train_loss_g, train_loss_integral = trainer(train_dataloader, model, device, optimizer, is_train=True)
    print(f'Epoch {epoch+1}/{num_epochs}, Train MSE: {train_loss_mse:.8f}, Train MAE: {train_loss_mae:.8f}, Train CAL: {train_loss_cal:.8f}, Train G: {train_loss_g:.8f}, Train Integral: {train_loss_integral:.8f}')
    lr_scheduler.step()
    test_loss_mse, test_loss_mae, test_loss_cal, test_loss_g, test_loss_integral = trainer(test_dataloader, model, device, optimizer, is_train=False)
    print(f'Epoch {epoch+1}/{num_epochs}, Test MSE: {test_loss_mse:.8f}, Test MAE: {test_loss_mae:.8f}, Test CAL: {test_loss_cal:.8f}, Test G: {test_loss_g:.8f}, Test Integral: {test_loss_integral:.8f}')

    hyper_pth_file = os.path.join(model_dir,f"spx_hyper_{epoch:05d}.pth")
    iv_pth_file = os.path.join(model_dir,f"spx_iv_{epoch:05d}.pth")
    model_pth_file = os.path.join(model_dir,f'spx_model_{epoch:05d}.pth')
    torch.save(hyper_model.state_dict(), hyper_pth_file)
    torch.save(iv_network.state_dict(), iv_pth_file)
    torch.save(model.state_dict(), model_pth_file)

    myrow = dict(
        epoch=epoch,
        train_loss_mse=train_loss_mse,
        train_loss_mae=train_loss_mae,
        train_loss_cal=train_loss_cal,
        train_loss_g=train_loss_g,
        train_loss_integral=train_loss_integral,
        test_loss_mse=test_loss_mse,
        test_loss_mae=test_loss_mae,
        test_loss_cal=test_loss_cal,
        test_loss_g=test_loss_g,
        test_loss_integral=test_loss_integral,
    )
    mylist.append(myrow)
    lossdf = pd.DataFrame(mylist)
    lossdf.to_csv(loss_csv_file,index=False)


