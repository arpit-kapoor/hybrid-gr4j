# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
import numpy as np
import datetime as dt

import os
import sys
sys.path.append("/project")

from model.hybrid.hgr4j_ann import HyGR4JNN
from model.utils.training import EarlyStopper
from model.utils.evaluation import evaluate
from model.optim.pso_np import PSO
from data.utils import read_dataset_from_file, get_station_list

# %%
data_dir = '/data/camels/aus/'
sub_dir = 'no-scale'
station_id = '318076'
run_dir = '/project/results/hygr4j'

n_epochs = 2
n_samples = 1
pop_size = 10
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
model.set_x1(817.2529155367329)

opt = optim.Adam(model.parameters(), lr=lr)
mse_loss = nn.MSELoss()

early_stopper = EarlyStopper(patience=5, min_delta=0.01)

# %%
def fit_fn(params):
    model.set_x1(params[0])
    y_hat = model(X_train)
    return mse_loss(y_hat, y_train).detach().numpy()

# %%
dims = 1
pso = PSO(
    pop_size=pop_size,
    fitness_function=fit_fn,
    num_params=dims,
    max_limits=900*np.ones(dims),
    min_limits=200*np.ones(dims),
    max_limits_vel=5*np.ones(dims),
    min_limits_vel=-5*np.ones(dims)
)

# %%
X_train = torch.nan_to_num(X_train)
X_val = torch.nan_to_num(X_val)

# %%
best_val = torch.inf

for i in range(n_epochs):

    start_ts = dt.datetime.now()

    swarm, best_pos, best_fit = pso.swarm, pso.best_swarm_pos, pso.best_swarm_err

    for j in range(n_samples):
        swarm, best_pos, best_fit = pso.evolve(swarm, best_pos, best_fit)

    pso_ts = dt.datetime.now()
    pso_time = pso_ts - start_ts

    opt.zero_grad()
    model.train()

    model.set_x1(best_pos[0])
    y_hat = model(X_train)
    
    loss = mse_loss(y_hat, y_train)
    loss.backward()

    opt.step()

    train_ts = dt.datetime.now()
    train_time = train_ts - pso_ts

    model.eval()
    y_val_hat = model(X_val)
    
    val_loss = mse_loss(y_val_hat, y_val)

    print(f"Epoch: {i+1}: train loss: {loss.detach().numpy():.4f} val loss: {val_loss.detach().numpy():.4f} best_pos: {best_pos}")

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

print(loss, val_loss)

# %%
P = X_train[:, 0].detach().numpy()
E = X_train[:, 1].detach().numpy()
Q = y_train.detach().numpy()
Q_hat = y_hat.detach().numpy()

print(evaluate(P, E, Q, Q_hat))

# %%
P = X_val[:, 0].detach().numpy()
E = X_val[:, 1].detach().numpy()
Q = y_val.detach().numpy()
Q_hat = y_val_hat.detach().numpy()

print(evaluate(P, E, Q, Q_hat))

