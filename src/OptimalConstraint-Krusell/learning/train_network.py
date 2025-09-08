import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from ..classes import Calibration as cal
from typing import Callable
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, num_agents):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1 + 1+ num_agents + 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        value = self.linear_relu_stack(x)
        return value

model = NeuralNetwork(cal.Nj)


def xi_fun(model: NeuralNetwork, z,a,g,A):
    z = np.array(z)
    a = np.array(a)
    g = np.array(g)
    A = np.array(A)
    z_shape = z.shape
    a_shape = a.shape
    g_shape = g.shape
    A_shape = A.shape
    z_dims = len(z_shape)
    a_dims = len(a_shape)
    g_dims = len(g_shape)
    A_dims = len(A_shape)



    if z_dims == 0 and A_dims == 0:
        ng = g.size## z 和 A 均是标量
        sample_x = np.zeros((ng + 3))
        sample_x[0] = z
        sample_x[1] = a
        sample_x[2:-1] = g
        sample_x[-1] = A
    elif z_dims == 0 and A_dims == 1:
        nA = A.size
        ng = np.size(g,0)
        sample_x = np.zeros((nA,ng + 3))
        sample_x[:,0] = z
        sample_x[:,1] = a
        sample_x[:,2:-1] = g.T
        sample_x[:,-1] = A
    elif z_dims == 1 and A_dims == 0:
        ng = g.size
        nz = z.size
        sample_x = np.zeros((nz, ng + 3))
        sample_x[:, 0] = z
        sample_x[:,1] = a
        sample_x[:,2:-1] = g
        sample_x[:,-1] = A
    elif z_dims == 1 and A_dims == 1:
        nA = np.size(A,0)
        nz = np.size(z, 0)
        ng = np.size(g,0)
        sample_x = np.zeros((nA * nz,ng + 3))
        sample_x[:,0] = np.tile(z,nA)
        sample_x[:,1] = a.flatten(order='F')
        sample_x[:,2:-1] = np.repeat(g.T,nz, axis=0)
        sample_x[:,-1] = np.repeat(A,nz, axis=0)
    elif z_dims == 2 and A_dims == 2:
        nt = np.size(z, 0)
        nz = np.size(z, 1)
        nA = np.size(A, 1)
        ng = np.size(g, 1)
        sample_x = np.zeros((nA * nz * nt, ng + 3))
        sample_x[:,0] = np.tile(z.flatten(order='C'), nA)
        sample_x[:,1] = np.reshape(a,(-1,nA)).flatten(order='F')
        sample_x[:,2:-1] = np.repeat(np.reshape(np.swapaxes(g,1,2),(-1,nz)), nz,axis=0)
        sample_x[:, -1] = np.tile(a, (nz,1)).flatten(order='F')
    else:
        raise Exception("Dimensions don't match!")

    return model(sample_x).numpy()