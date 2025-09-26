import numpy as np
from math import sqrt
class Calibration:
    gamma = 2
    alpha = 0.36
    delta = 0.08
    delta_diff = 0.1
    rho = 0.04
    theta = 1-0.6
    hat_z = 1.038
    eta = 1-0.6
    hat_A = 1.038
    sigma = sqrt(0.2 ** 2 * (1 - (1-theta) ** 2))
    sigma_z = sqrt(0.2 ** 2 * (1 - (1-eta) ** 2))
    dt = 1
    T = 50
    Nt = int(T/dt)
    Nj = 80
    Nk = 50

    z0_low = 0.2
    z0_high = 1.8
    a0_low = 0
    a0_high = 100
    A0_low = 0.2
    A0_high = 1.8

    # parameters on net work
    num_agents = Nj
    num_inputs = num_agents + 3
    num_layers = 2
    num_neurons = 512
    num_outputs = 1
    num_epochs = 1000
    learning_rate = 1e-3
    batch_size = 128
