import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from OptimalConstraintKrusell import (Calibration, Simulation, WorkSpace,  initial_value_fun,
                                       IndexedDataset, LossFunction, InitializedModel, DeepSet,
                                      xi_fun, xi_fun_with_grad, adjust_learning_rate)
from train import train_model

cal = Calibration()
torch.set_default_dtype(torch.float64)
device = cal.device
print(f"\ndevice = {device}")
torch.set_default_device(cal.device_str)
torch.manual_seed(cal.seed_torch)
torch.cuda.manual_seed(cal.seed_torch)

res_net = DeepSet(g_dim=2,
                    phi_hidden_dim=cal.phi_hidden_dim,
                    phi_out_dim=cal.phi_out_dim,
                    rho_hidden_dim=cal.rho_hidden_dim,
                    n_phi_layers=cal.n_phi_layers,
                    n_rho_layers=cal.n_rho_layers).to(device)

model = res_net
if cal.is_train:
    print("Training model started.")
    train_model(model, cal)
    print("Training model finished.")
else:
    print("Loading parameters")
    model.load_state_dict(torch.load("model.pth", weights_only=True))
print("Compute ergodic distribution...\n")
model.eval()
sml_dist = Simulation(1)
sml_dist.A0[:] = cal.hat_A
sml_dist.initial_dist()
ws_dist = WorkSpace(model, 1)
ws_dist.g = sml_dist.g0.clone()
ws_dist.A[:] = cal.hat_A
for t in range(cal.Nt * cal.num_epochs):
    ws_dist.dWj_other = (torch.normal(0, ws_dist.dt, size=(1,ws_dist.Nj), device=ws_dist.device_str))
    ws_dist.simulate_dist_one_step(xi_fun, ws_dist.dWj_other)
print("Done\n")

print("Compute ergodic policies:\n")
ws_test = WorkSpace(model, cal.Ng)
ws_test.a = cal.a0_low + (100 - cal.a0_low) * torch.pow(torch.linspace(0.0, 1.0, cal.Ng), 1.0 / cal.a_cur)
ws_test.z[:] = cal.hat_z
ws_test.g = ws_dist.g.repeat(cal.Ng,1,1)
ws_test.A[:] = cal.hat_A

grid = ws_test.a.cpu().numpy()
xi = xi_fun(model, ws_test.z, ws_test.a, ws_test.g, ws_test.A)
cons_grid = ws_test.consumption(xi).cpu().numpy()

ws_test.capital_labor()
ws_test.prices()
ws_test.saving(xi_fun)
saving_grid = ws_test.s.cpu().numpy()
print("Done\n")
np.savetxt("dist_a.csv", ws_dist.g[:,:,1].cpu().numpy(), delimiter=",")
np.savetxt("dist_z.csv", ws_dist.g[:,:,0].cpu().numpy(), delimiter=",")
np.savetxt("grid.csv", ws_test.a.cpu().numpy(), delimiter=",")
np.savetxt("c_grid.csv", cons_grid, delimiter=",")

# 绘制结果
plt.figure(figsize=(12, 5))
plt.plot(grid, cons_grid, label='consumption')
#plt.plot(grid, saving_grid, label='saving')
plt.xlabel('asset')
plt.title('Policies')
plt.legend()
plt.tight_layout()
plt.show()