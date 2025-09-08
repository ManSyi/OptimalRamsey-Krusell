import numpy as np
from ..classes import Calibration, Simulation, WorkSpace
from ..utilities import *
import torch
import torch.nn as nn

def simulate(cal, inputs, ws, xi_fun):
    for t in range(cal.Nt):
        ws.K = capital(ws.a[t,:,:])
        ws.Y, ws.r, ws.w, ws.Lambda = prices(ws.A[t,:], cal.alpha, cal.delta, ws.K, ws.L,
                                             ws.xi[t,:,:], ws.a[t,:,:], ws.z[t,:])
        ws.z_drift = cal.theta * (cal.z_hat - ws.z[t,:])
        ws.A_drift = cal.eta * (cal.A_hat - ws.A[t,:])
        ws.z[t + 1,:] = ws.z[t,:] + ws.z_drift * cal.dt + cal.sigma_z * inputs.dWj[t,:]
        ws.A[t + 1,:] = ws.A[t,:] + ws.A_drift * cal.dt + cal.sigma * inputs.dWk[t,:]

        finite_diff(cal, ws, t, xi_fun)

        ws.xi[t+1,:,:] = siml_xi(ws.xi[t,:,:], cal.rho, ws.r, cal.dt, ws.Lambda,
                                 - cal.sigma_z * ws.xi_z[:,:],
                                 - cal.sigma * ws.xi_A[:,:], inputs.dWj[t,:], inputs.dWk[t,:])

        upwind(cal,ws, t, xi_fun)


class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,xi_fun, cal:Calibration, ws:WorkSpace):
        return torch.from_numpy(np.linalg.norm(
            ws.xi[1:, :, :] - xi_fun(ws.model, ws.z[1:, :], ws.a[1:, :, :], ws.a[1:, :, :], ws.A[1:, :])) / (
                    cal.Nj * cal.Nk * (cal.Nt - 1)))
