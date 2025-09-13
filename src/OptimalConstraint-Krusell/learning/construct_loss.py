import numpy as np
from ..classes import Calibration, Simulation, WorkSpace
from ..utilities import *
import torch
import torch.nn as nn
from train_network import xi_fun_with_grad, xi_fun
class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, xi_fun:xi_fun, criterion, xi_fun_with_grad:xi_fun_with_grad, index, cal:Calibration, ws:WorkSpace):
        sum = 0.0
        for ti in range(cal.Nt-1):

            ws.dWj = (torch.normal(0, (ti + 1) * cal.dt, size=ws.z.shape)
                      - torch.normal(0, ti * cal.dt, size=ws.z.shape))
            ws.dWk = (torch.normal(0, (ti + 1) * cal.dt, size=ws.A.shape)
                      - torch.normal(0, ti * cal.dt, size=ws.A.shape))

            ws.K, ws.L = capital_labor(ws.a, ws.z)
            ws.Y, ws.r, ws.w, ws.Lambda = prices(ws.A, cal.alpha, cal.delta, ws.K, ws.L,
                                                 ws.xi, ws.a, ws.z)

            ws.z, ws.z_drift = siml_z(ws.z, cal.theta, cal.hat_z, cal.dt, cal.sigma_z, ws.dWj,
                                      (cal.z0_low, cal.z0_high))
            ws.A, ws.A_drift = siml_A(ws.A, cal.eta, cal.hat_A, cal.dt, cal.sigma, ws.dWk,
                                      (cal.A0_low, cal.A0_high))

            finite_diff(cal, ws, ti, xi_fun)

            ws.xi = siml_xi(ws.xi, cal.rho, ws.r, cal.dt, ws.Lambda,
                                         - cal.sigma_z * ws.xi_z,
                                         - cal.sigma * ws.xi_A, ws.dWj, ws.dWk)

            upwind(cal, ws, ti, xi_fun)
            sum = sum + criterion(ws.xi.T.flatten().index_select(0, index[:, 0] + index[:, 1]),
                                  xi_fun_with_grad(ws.model, ws.z, ws.a, ws.a, ws.A, index[:, 0] + index[:, 1])).item()

        return  sum / (cal.Nt - 1)




