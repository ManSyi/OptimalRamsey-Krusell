import numpy as np
from .calibration import Calibration
import torch

class Simulation(Calibration):
    def __init__(self):
        super().__init__()
        self.z0 = torch.zeros(self.Nj)
        self.A0 = torch.zeros(self.Nk)
        self.a0 = torch.zeros(self.Nj)
        self.g0 = torch.zeros(self.Nj)
        self.g =  torch.zeros(self.Nj, self.Nk)
        self.sample_index = torch.zeros(self.Nj * self.Nk, 2, dtype=torch.int32)
        self.sample_index[:,0] = torch.arange(self.Nj, dtype=torch.int32).repeat(self.Nk)
        self.sample_index[:,1] = torch.arange(self.Nk, dtype=torch.int32).repeat_interleave(self.Nj, dim=0)
    @staticmethod
    def toSample_x(z,a,g,A):
        z_dims = len(z.shape)
        A_dims = len(A.shape)
        g_dims = len(g.shape)
        a_dims = len(a.shape)

        if z_dims == 0 and A_dims == 0:
            ng = g.shape[0]  ## z 和 A 均是标量
            sample_x = torch.zeros((ng + 3))
            sample_x[0] = z
            sample_x[1] = a
            sample_x[2:-1] = g
            sample_x[-1] = A
        elif z_dims == 0 and A_dims == 1:
            nA = A.shape[0]
            ng = g.shape[0]
            sample_x = torch.zeros((nA, ng + 3))
            sample_x[:, 0] = z
            sample_x[:, 1] = a
            sample_x[:, 2:-1] = g.T
            sample_x[:, -1] = A
        elif z_dims == 1 and A_dims == 0:
            ng = g.shape[0]
            nz = z.shape[0]
            sample_x = torch.zeros((nz, ng + 3))
            sample_x[:, 0] = z
            sample_x[:, 1] = a
            sample_x[:, 2:-1] = g
            sample_x[:, -1] = A
        elif z_dims == 1 and A_dims == 1:
            nA = A.shape[0]
            nz = z.shape[0]
            ng = g.shape[0]
            sample_x = torch.zeros((nA * nz, ng + 3))
            sample_x[:, 0] = z.repeat(nA)
            if a_dims == 2:
                sample_x[:, 1] = a.T.flatten()
            elif a_dims == 1:
                sample_x[:, 1] = a.repeat(nA)
            if g_dims == 2:
                sample_x[:, 2:-1] = g.T.repeat_interleave(nz, dim=0)
            elif g_dims == 1:
                sample_x[:, 2:-1] = g.repeat(nA * nz, 1)

            sample_x[:, -1] = A.repeat_interleave(nz, dim=0)

        else:
            raise Exception("Dimensions don't match!")
        return sample_x

    @staticmethod
    def toOutput_y(sample_y, z_dims, A_dims, z,A):
        if z_dims == 0 and A_dims == 0:
            return sample_y.squeeze()
        elif z_dims == 0 and A_dims == 1:
            return sample_y.squeeze()
        elif z_dims == 1 and A_dims == 0:
            return sample_y.squeeze()
        elif z_dims == 1 and A_dims == 1:
            nA = A.shape[0]
            nz = z.shape[0]
            return reshape_fortran(sample_y, (nz, nA))
        else:
            raise Exception("Dimensions don't match!")

    def initial_sample(self):
        self.z0.uniform_(self.z0_low, self.z0_high)
        self.A0.uniform_(self.A0_low, self.A0_high)
        self.a0.uniform_(self.a0_low, self.a0_high / 2)
        self.g0= self.a0
        return self.toSample_x(self.z0,self.a0,self.g0,self.A0)



def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))
