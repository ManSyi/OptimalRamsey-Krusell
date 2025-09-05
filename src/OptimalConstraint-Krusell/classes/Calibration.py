import numpy as np

class Calibration:
    gamma = 0
    alpha = 0
    delta = 0
    delta_diff = 0
    rho = 0
    theta = 0
    hat_z = 0
    eta = 0
    hat_A = 0
    sigma = 0
    sigma_z = 0
    dt = 1
    T = 20
    Nt = int(T/dt)
    Nj = 10
    Nk = 8
    z0_low = 0
    z0_high = 0
    a0_low = 0
    a0_high = 0
    A0_low = 0
    A0_high = 0