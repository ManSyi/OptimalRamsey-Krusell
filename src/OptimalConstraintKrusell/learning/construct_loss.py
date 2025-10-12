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
                index,z,a,g,A, ws:WorkSpace, sml:Simulation):

        loss_sum = 0.0

        # ws.sample_size = a.shape[0] # or index.shape[0]
        # ws.a[:ws.sample_size] = a
        # ws.g[:ws.sample_size,:] = g
        #
        # for ti in range(ws.Nt-1):
        #     ws.z[:ws.sample_size] = sml.z_path[index, ti]
        #     ws.A[:ws.sample_size] = sml.A_path[index, ti]
        #     ws.simulate_invariant_dist_one_step(xi_fun)
        #
        # ti = 0
        # ws.z[:ws.sample_size] = z
        # ws.A[:ws.sample_size] = A
        # ws.a[:ws.sample_size] = a
        #
        # ws.dWj[:ws.sample_size] = sml.dWj_path[index, ti]
        # ws.dWk[:ws.sample_size] = sml.dWk_path[index, ti]
        #
        # ws.xi[:ws.sample_size] = xi_fun(ws.model, z, a, ws.g[:ws.sample_size, :], A)
        # ws.simulate_xi_one_step(xi_fun)
        # xi_est = xi_fun_with_grad(ws.model, sml.z_path[index, ti + 1], ws.a[:ws.sample_size],
        #                           ws.g[:ws.sample_size, :], sml.A_path[index, ti + 1])
        # loss_sum = criterion(ws.xi[:ws.sample_size], xi_est)

        ws.sample_size = a.shape[0] # or index.shape[0]
        ws.xi[:ws.sample_size] = xi_fun(ws.model, z, a, g, A)
        ws.a[:ws.sample_size] = a
        ws.g[:ws.sample_size, :] = g
        loss_sum = 0
        for ti in range(ws.Nt - 1):
            ws.z[:ws.sample_size] = sml.z_path[index, ti]
            ws.A[:ws.sample_size] = sml.A_path[index, ti]
            ws.dWj[:ws.sample_size] = sml.dWj_path[index, ti]
            ws.dWk[:ws.sample_size] = sml.dWk_path[index, ti]
            ws.simulate_xi_one_step(xi_fun)
            # loss_sum = loss_sum + criterion(ws.xi[:ws.sample_size], xi_fun_with_grad(ws.model, sml.z_path[index, ti + 1], ws.a[:ws.sample_size],
            #                           ws.g[:ws.sample_size, :], sml.A_path[index, ti + 1]))
            # loss_sum = loss_sum + criterion(torch.tensor(1.0), xi_fun_with_grad(ws.model, sml.z_path[index, ti + 1], ws.a[:ws.sample_size],
            #                           ws.g[:ws.sample_size, :], sml.A_path[index, ti + 1]) / ws.xi[:ws.sample_size])

            # loss_sum = loss_sum + torch.norm( xi_fun_with_grad(ws.model, sml.z_path[index, ti + 1], ws.a[:ws.sample_size],
            #                            ws.g[:ws.sample_size, :], sml.A_path[index, ti + 1]) / ws.xi[:ws.sample_size] - 1.0)

            loss_sum = loss_sum + torch.mean(torch.pow(xi_fun_with_grad(ws.model, sml.z_path[index, ti + 1], ws.a[:ws.sample_size],
                                       ws.g[:ws.sample_size, :], sml.A_path[index, ti + 1]) / ws.xi[:ws.sample_size] - 1.0, 2))


        return  loss_sum




