import numpy as np
from math import sqrt
import torch
class Calibration:
    gamma = 2
    alpha = 0.36
    delta = 0.08 / 4
    delta_diff = 0.1
    deathrate = 0.005
    rho = 0.01
    theta = 1-0.6
    hat_z = 1.038
    eta = 1-0.6
    hat_A = 1.038
    sigma = sqrt(0.2 ** 2 * (1 - (1-theta) ** 2))
    sigma_z = sqrt(0.2 ** 2 * (1 - (1-eta) ** 2))
    dt = 1e-3
    Nt = 10
    Nt_xi = 20
    T = Nt * dt
    Nj = 100 #num of agents to approximate distribution
    Ns = 100 #num of samples
    Ng = 10000 #num of grid points
    loss_min = 1e-8
    z0_low = 0.2
    z0_high = 1.8
    a0_low = 0.01
    a0_high = 1000
    a_cur = 0.15
    A0_low = 0.2
    A0_high = 1.8
    KYratio_low = 7.0
    KYratio_high = 15.0
    L = 1

    is_competitive = 1
    is_train = 1

    seed_numpy = 42
    seed_torch = 42

    device_str = "cpu"

    device = torch.device(device_str)
    # parameters on net work
    num_agents = Nj
    num_epochs = 20

    learning_rate = 1e-3
    batch_size = 512

    n_phi_layers = 2  # number of hidden layers in phi
    n_rho_layers = 4  # number of hidden layers in rho
    phi_hidden_dim = 64
    phi_out_dim = 32
    rho_hidden_dim = 128

    def simulate_z_one_step(self, z_cur, dWj):
        return torch.minimum(
            torch.maximum(z_cur + self.theta * (
                    self.hat_z - z_cur) * self.dt + self.sigma_z * dWj,
                          torch.tensor(self.z0_low)),
            torch.tensor(self.z0_high))

    def simulate_A_one_step(self, A_cur, dWk):
        return torch.minimum(
            torch.maximum(A_cur + self.eta * (
                        self.hat_A - A_cur) * self.dt + self.sigma * dWk,
                          torch.tensor(self.A0_low)),
            torch.tensor(self.A0_high))

def initial_value_fun( z, a, g, A):
    Nj = 40
    alpha = 0.36
    gamma = 2
    rho = 0.04
    L = 1
    K = g.sum(dim=1) / Nj
    Y = A * K ** alpha * L ** (1-alpha)
    r = alpha * Y / K
    w = (1-alpha) * Y / L
    return (w * z + r * a) ** (1-gamma) / (1-gamma) / rho



