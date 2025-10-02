import numpy as np
from .calibration import Calibration
from .simulation import Simulation
import torch
import copy

class WorkSpace(Calibration):
    def __init__(self, model):
        super().__init__()
        self.a = torch.zeros(self.batch_size)
        self.g = torch.zeros((self.batch_size, self.Nj))
        self.xi = torch.zeros(self.batch_size)

        self.xi_z = torch.zeros(self.batch_size)
        self.xi_A = torch.zeros(self.batch_size)

        self.varsigma = torch.zeros(self.batch_size)
        self.varsigma_z = torch.zeros(self.batch_size)
        self.xi_a = torch.zeros(self.batch_size)
        self.c =  torch.zeros(self.batch_size)
        self.c_other = torch.zeros((self.batch_size, self.Nj))
        self.s_other = torch.zeros((self.batch_size, self.Nj))
        self.xi_a_other = torch.zeros((self.batch_size, self.Nj))
        self.s = torch.zeros(self.batch_size)
        self.K = torch.zeros(self.batch_size)
        self.Y = torch.zeros(self.batch_size)
        self.r = torch.zeros(self.batch_size)
        self.w = torch.zeros(self.batch_size)
        self.r_k = torch.zeros(self.batch_size)
        self.w_k = torch.zeros(self.batch_size)
        self.Lambda = torch.zeros(self.batch_size)
        self.dWj =  torch.zeros(self.batch_size)
        self.dWk =  torch.zeros(self.batch_size)
        self.z =  torch.zeros(self.batch_size)
        self.A = torch.zeros(self.batch_size)



        self.a_path = torch.zeros((self.batch_size,self.Nt))
        self.g_path = torch.zeros((self.batch_size, self.Nj, self.Nt))
        
        self.xi_infinitesimal_path = torch.zeros((self.batch_size, self.Nt - 1))

        self.model = model
        self.sample_size = self.batch_size


    def derivative_z_A(self, xi_fun):

        self.xi_z = torch.maximum((xi_fun(self.model, self.z + self.delta_diff, self.a, self.g, self.A)
                                 - xi_fun(self.model, self.z - self.delta_diff, self.a, self.g, self.A))
                                  / (self.delta_diff * 2), torch.tensor(1e-5))

        self.xi_A = torch.maximum((xi_fun(self.model, self.z, self.a, self.g, self.A + self.delta_diff)
                     - xi_fun(self.model, self.z, self.a, self.g, self.A - self.delta_diff))
                                  / (self.delta_diff * 2), torch.tensor(1e-5))


    def derivative_a(self, xi_fun):

        self.xi_a = torch.maximum(xi_fun(self.model, self.z, self.a + self.delta_diff, self.g, self.A)
                  - xi_fun(self.model, self.z, self.a - self.delta_diff, self.g, self.A)
                                 / (self.delta_diff * 2), torch.tensor(1e-5))
        self.c = self.consumption(self.xi_a)
        self.s = self.r * self.a + self.w * self.z - self.c
        
        for j in range(self.Nj):
            self.xi_a_other[:,j] = torch.maximum(xi_fun(self.model, self.z, self.g[:,j] + self.delta_diff, self.g, self.A)
                  - xi_fun(self.model, self.z, self.g[:,j] - self.delta_diff, self.g, self.A)
                                 / (self.delta_diff * 2), torch.tensor(1e-5))

        self.c_other = self.consumption(self.xi_a_other)
        self.s_other = self.r.unsqueeze(1) * self.g + self.w.unsqueeze(1) * self.z.unsqueeze(1) - self.c_other


    def capital_labor(self):
        self.K = 1 / self.Nj * torch.maximum(self.g.sum(1), torch.tensor(0.01))
        self.L = 1.0

    def shadow_price(self):
        if not self.is_competitive:
            self.r_k = self.alpha * (self.alpha - 1) * self.A * self.K ** (self.alpha - 2) * self.L ** (1 - self.alpha)
            self.w_k = self.alpha * (1 - self.alpha) * self.A * self.K ** (self.alpha - 1) * self.L ** (-self.alpha)
            self.Lambda = 1 / self.Nj * (self.xi_a_other * (self.r_k.unsqueeze(1) * self.g + self.w_k.unsqueeze(1) * self.z.unsqueeze(1))).sum(dim=1)

    def prices(self):
        self.Y = self.A * self.K ** self.alpha * self.L ** (1 - self.alpha)
        # self.r = torch.maximum(self.alpha * self.Y / self.K - self.delta, torch.tensor(1e-5))
        self.r = self.alpha * self.Y / self.K - self.delta
        self.w = (1 - self.alpha) * self.Y / self.L



    def consumption(self, xip):
        return xip ** (-1 / self.gamma)

    def utils(self, c):
        return torch.maximum(c, torch.tensor(0.001)) ** (1-self.gamma) / (1-self.gamma)

    def risk_loadings(self):
        self.varsigma_z = - self.xi_z * self.sigma_z
        self.varsigma = - self.xi_A * self.sigma


    def simulate_one_step(self, ti, xi_fun):


        self.a = self.a_path[:, ti]
        self.g = self.g_path[:,:,ti]
        
        self.derivative_z_A(xi_fun)
        self.capital_labor()
        self.prices()
        self.derivative_a(xi_fun)
        self.risk_loadings()
        self.shadow_price()

        self.xi_infinitesimal_path[:,ti] = (-(self.utils(self.c) - self.Lambda * (self.K - self.a)) * self.dt
                                      - self.varsigma_z * self.dWj - self.varsigma * self.dWk)
        self.a_path[:,ti + 1] = torch.minimum(
            torch.maximum(self.a + self.s * self.dt, torch.tensor(self.a0_low)),
            torch.tensor(self.a0_high))
        
        self.g_path[:,:, ti + 1] = torch.minimum(
            torch.maximum(self.g + self.s_other * self.dt, torch.tensor(self.a0_low)),
            torch.tensor(self.a0_high))

    def simulate_one_step_reverse(self, ti):
        # xi_pre = f(xi_cur)
        self.xi = (1 / (1 + (self.rho + self.deathrate) * self.dt)
                   * (self.xi - self.xi_infinitesimal_path[:,ti] ))
