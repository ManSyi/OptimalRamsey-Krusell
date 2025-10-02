import numpy as np
from ..classes import Calibration, Simulation, WorkSpace
from ..utilities import *
import torch
import torch.nn as nn
from .train_network import xi_fun_with_grad, xi_fun
class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, xi_fun, criterion, xi_fun_with_grad,
                index,a,g, ws:WorkSpace, sml:Simulation):
        loss_sum = 0.0

        ws.sample_size = a.shape[0] # or index.shape[0]

        ws.a_path[:ws.sample_size, 0] = a
        ws.g_path[:ws.sample_size, :, 0] = g

        # ws.a[:ws.sample_size] = a
        # ws.g[:ws.sample_size,:] = g

        # ws.xi[:ws.sample_size] = xi_fun(ws.model, sml.z_path[index, -1], a, g, sml.A_path[index, -1])

        for ti in range(ws.Nt-1):
            ws.z[:ws.sample_size] = sml.z_path[index, ti]
            ws.A[:ws.sample_size] = sml.A_path[index, ti]
            ws.dWj[:ws.sample_size] = sml.dWj_path[index, ti]
            ws.dWk[:ws.sample_size] = sml.dWk_path[index, ti]
            ws.simulate_one_step(ti, xi_fun)
            # xi_est = xi_fun_with_grad(ws.model, sml.z_path[index, ti], ws.a[:ws.sample_size],
            #                           ws.g[:ws.sample_size, :], sml.A_path[index, ti])
            # loss_sum = loss_sum + criterion(ws.xi[:ws.sample_size], xi_est)

        ws.xi[:ws.sample_size] = xi_fun(ws.model, sml.z_path[index, -1], ws.a_path[:ws.sample_size, -1],
                       ws.g_path[:ws.sample_size,:,-1], sml.A_path[index, -1])

        for ti in reversed(range(ws.Nt - 1)):
            ws.simulate_one_step_reverse(ti)

            # Only calculate the actual sample size
            xi_est = xi_fun_with_grad(ws.model, sml.z_path[index,ti], ws.a_path[:ws.sample_size,ti],
                                      ws.g_path[:ws.sample_size,:,ti], sml.A_path[index,ti])
            loss_sum = loss_sum + criterion(ws.xi[:ws.sample_size], xi_est)

        return  loss_sum / (ws.Nt - 1)




