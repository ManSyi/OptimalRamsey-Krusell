import numpy as np
from ..classes import Calibration, Simulation, WorkSpace
from ..utilities import *
import torch
import torch.nn as nn
import datetime
from .train_network import xi_fun_with_grad, xi_fun
class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, xi_fun:xi_fun, criterion, xi_fun_with_grad:xi_fun_with_grad, index, cal:Calibration, ws:WorkSpace, sml:Simulation):
        loss_sum = 0.0

        ws.initial_start(sml)
        for ti in range(cal.Nt-1):

            ws.simulate_one_step(ti, xi_fun)

            # select = index[:, 0] + index[:, 1] * cal.Nj
            # x1 = ws.xi.T.flatten().index_select(0, select)
            # x2 = xi_fun_with_grad(ws.model, ws.z, ws.a, ws.a, ws.A, select).squeeze()
            # loss_sum = loss_sum + criterion(x1, x2)

        ws.initial_end(xi_fun)

        for ti in reversed(range(cal.Nt - 1)):
            ws.simulate_one_step_reverse(ti)

            select = index[:, 0] + index[:, 1] * cal.Nj
            x1 = ws.xi.T.flatten().index_select(0, select)
            x2 = xi_fun_with_grad(ws.model, ws.z_path[:,ti], ws.a_path[:,:,ti], ws.a_path[:,:,ti], ws.A_path[:,ti], select).squeeze()
            loss_sum = loss_sum + criterion(x1, x2)

        return  loss_sum / (cal.Nt - 1)




