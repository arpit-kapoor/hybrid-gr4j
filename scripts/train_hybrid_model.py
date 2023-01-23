# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
import numpy as np
import datetime as dt

import os
import sys
sys.path.append("..")

from model.hybrid.hgr4j_ann import HyGR4JNN
from model.utils.training import EarlyStopper
from model.utils.evaluation import evaluate
from model.optim.pso_np import PSO
from data.utils import read_dataset_from_file, get_station_list

# %%
data_dir = '/home/z5370003/data/camels/aus/'
sub_dir = 'no-scale-seq'
station_id = '318076'
run_dir = '/home/z5370003/results/hygr4j'

n_epochs = 150
n_samples = 5
pop_size = 20
lr = 1e-2

# %%
print(f"Reading data for station_id: {station_id}")
train_ds, val_ds = read_dataset_from_file(data_dir, 
                                          sub_dir, 
                                          station_id=station_id)

# %%
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

plot_dir = os.path.join(run_dir, 'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

t_train, X_train, y_train = train_ds.tensors
t_val, X_val, y_val = val_ds.tensors

# %%
model = HyGR4JNN(0.0)
min_pos = 100.0
max_pos = 1200.0
mu = np.array([817.253])
std = (max_pos - mu)/2

opt = optim.Adam(model.parameters(), lr=lr)
mse_loss = nn.MSELoss()

early_stopper = EarlyStopper(patience=5, min_delta=0.05)

# %%
def fit_fn(params):
    model.set_x1(params[0])
    y_hat = model(X_train)
    return mse_loss(y_hat, y_train).detach().numpy()

# %%
print(f"Initialize PSO.. time: {dt.datetime.now().time()}")
dims = 1
pso = PSO(
    pop_size=pop_size,
    fitness_function=fit_fn,
    num_params=dims,
    mu=mu, std=std,
    max_limits=max_pos*np.ones(dims),
    min_limits=min_pos*np.ones(dims),
    max_limits_vel=5*np.ones(dims),
    min_limits_vel=-5*np.ones(dims)
)
print(f"Train Model.. time: {dt.datetime.now().time()}")

# %%
X_train = torch.nan_to_num(X_train)
X_val = torch.nan_to_num(X_val)

# %%
best_val = torch.inf

for i in range(n_epochs):

    start_ts = dt.datetime.now()

    if i % n_samples == 0:

        swarm, best_pos, best_fit = pso.swarm, pso.best_swarm_pos, pso.best_swarm_err

        swarm, best_pos, best_fit = pso.evolve(swarm, best_pos, best_fit)

        pso_ts = dt.datetime.now()
        pso_time = pso_ts - start_ts

        model.set_x1(best_pos[0])

    opt.zero_grad()
    model.train()

    y_hat = model(X_train)
    
    loss = mse_loss(y_hat, y_train)
    loss.backward()

    opt.step()

    train_ts = dt.datetime.now()
    train_time = train_ts - pso_ts

    model.eval()
    y_val_hat = model(X_val)
    
    val_loss = mse_loss(y_val_hat, y_val)

    print(f"Epoch: {i+1}: train loss: {loss.detach().numpy():.4f} val loss: {val_loss.detach().numpy():.4f} best_pos: {best_pos[0]:.2f} timestamp: {dt.datetime.now().time()}")

    if val_loss < best_val:
        torch.save(model, os.path.join(run_dir, 'best_model.pt'))

    if early_stopper.early_stop(val_loss):
        break

# %%
model = torch.load(os.path.join(run_dir, 'best_model.pt'))

model.eval()

y_hat = model(X_train)
loss = mse_loss(y_hat, y_train)

y_val_hat = model(X_val)
val_loss = mse_loss(y_val_hat, y_val)

# %%
P = X_train[:, 0].detach().numpy()
E = X_train[:, 1].detach().numpy()
Q = y_train.detach().numpy()
Q_hat = y_hat.detach().numpy()

evaluate(P, E, Q, Q_hat)

# %%
P = X_val[:, 0].detach().numpy()
E = X_val[:, 1].detach().numpy()
Q = y_val.detach().numpy()
Q_hat = y_val_hat.detach().numpy()

evaluate(P, E, Q, Q_hat)

