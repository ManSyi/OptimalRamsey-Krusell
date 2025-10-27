import numpy as np
from ..classes import Calibration, Simulation, WorkSpace
from ..utilities import *
import torch
import torch.nn as nn
from .network import xi_fun_with_grad, xi_fun
class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, xi_fun, criterion, xi_fun_with_grad,
                ws:WorkSpace, sml:Simulation):

        ws.xi_cur = xi_fun_with_grad(ws.model, ws.z, ws.a, ws.g, ws.A)


        ws.capital_labor()
        ws.prices()
        ws.saving(xi_fun_with_grad)
        ws.dist_drift(xi_fun_with_grad)
        ws.shadow_price()


        ws.Exi = -(ws.Lambda - (ws.rho - ws.r) * ws.xi_cur)

        ws.a = ws.simulate_a_one_step(ws.a, ws.s)
        ws.ga = ws.simulate_a_one_step(ws.g[:, :, 1], ws.ga_drift)

        ws.dWj = (torch.normal(0, ws.dt, size=(ws.max_batch_size,), device=ws.device_str))
        ws.dWj_other = (torch.normal(0, ws.dt, size=(ws.max_batch_size, ws.Nj), device=ws.device_str))
        ws.dWk = (torch.normal(0, ws.dt, size=(ws.max_batch_size,), device=ws.device_str))
        ws.z = ws.simulate_z_one_step(ws.z, ws.dWj)
        ws.gz = ws.simulate_z_one_step(ws.g[:, :, 0], ws.dWj_other)
        ws.A = ws.simulate_A_one_step(ws.A, ws.dWk)
        ws.Exi0 = (xi_fun_with_grad(ws.model, ws.z, ws.a, torch.stack((ws.gz,ws.ga),dim=2), ws.A) - ws.xi_cur) / ws.dt

        ws.dWj = (torch.normal(0, ws.dt, size=(ws.max_batch_size,), device=ws.device_str))
        ws.dWj_other = (torch.normal(0, ws.dt, size=(ws.max_batch_size, ws.Nj), device=ws.device_str))
        ws.dWk = (torch.normal(0, ws.dt, size=(ws.max_batch_size,), device=ws.device_str))
        ws.z = ws.simulate_z_one_step(ws.z, ws.dWj)
        ws.gz = ws.simulate_z_one_step(ws.g[:, :, 0], ws.dWj_other)
        ws.A = ws.simulate_A_one_step(ws.A, ws.dWk)
        ws.Exi1 = (xi_fun_with_grad(ws.model, ws.z, ws.a, torch.stack((ws.gz,ws.ga),dim=2), ws.A) - ws.xi_cur) / ws.dt


        loss = torch.mean((ws.Exi0 - ws.Exi) * (ws.Exi1 - ws.Exi))


        return  loss




