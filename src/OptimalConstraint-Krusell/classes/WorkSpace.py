import numpy as np
from .Calibration import Calibration
from .Simulation import Simulation
from ..learning import NeuralNetwork
import torch

class WorkSpace(Calibration):
    def __init__(self, model: NeuralNetwork, num_sample_z, num_sample_A):
        super().__init__(self)
        self.a = torch.zeros((num_sample_z, num_sample_A))
        self.xi = torch.zeros((num_sample_z, num_sample_A))

        self.xi_z = torch.zeros((num_sample_z, num_sample_A))
        self.xi_A = torch.zeros((num_sample_z, num_sample_A))
        self.xi_a = torch.zeros((num_sample_z, num_sample_A))
        self.c =  torch.zeros((num_sample_z, num_sample_A))
        self.cF =  torch.zeros((num_sample_z, num_sample_A))
        self.cB =  torch.zeros((num_sample_z, num_sample_A))
        self.c0 =  torch.zeros((num_sample_z, num_sample_A))
        self.sF =  torch.zeros((num_sample_z, num_sample_A))
        self.sB =  torch.zeros((num_sample_z, num_sample_A))
        self.xiF =  torch.zeros((num_sample_z, num_sample_A))
        self.xiB =  torch.zeros((num_sample_z, num_sample_A))
        self.indF =  torch.zeros((num_sample_z, num_sample_A))
        self.indB =  torch.zeros((num_sample_z, num_sample_A))
        self.ind0 =  torch.zeros((num_sample_z, num_sample_A))
        self.K =  torch.zeros(num_sample_A)
        self.Y =  torch.zeros(num_sample_A)
        self.r =  torch.zeros(num_sample_A)
        self.w =  torch.zeros(num_sample_A)
        self.Lambda =  torch.zeros(num_sample_A)
        self.L = 0


        self.dWj =  torch.zeros(num_sample_z)
        self.dWk =  torch.zeros(num_sample_A)
        self.z =  torch.zeros(num_sample_z)
        self.A = torch.zeros(num_sample_A)
        self.z_drift = torch.zeros(num_sample_z)
        self.A_drift = torch.zeros(num_sample_A)

        self.model = model


    def initial_start(self, cal:Calibration, sml:Simulation, xi_fun):
        self.a = sml.a0
        self.A = sml.A0
        self.z = sml.z0
        self.xi = xi_fun(self.model, self.z, self.a, self.a, self.A)