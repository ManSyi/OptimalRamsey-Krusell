import numpy as np
from .calibration import Calibration
from .simulation import Simulation
import torch
import copy

class WorkSpace(Calibration):
    def __init__(self, model, size):
        super().__init__()
        self.a = torch.zeros(size)
        self.g = torch.zeros((size, self.Nj, 2))
        self.ga = torch.zeros((size, self.Nj))
        self.gz = torch.zeros((size, self.Nj))
        self.z =  torch.zeros(size)
        self.A = torch.zeros(size)

        self.xi = torch.zeros(size)
        self.Exi = torch.zeros(size)
        self.Exi0 = torch.zeros(size)
        self.Exi1 = torch.zeros(size)


        self.c =  torch.zeros(size)
        self.ga_drift = torch.zeros((size, self.Nj))
        self.gz_drift = torch.zeros((size, self.Nj))
        self.xi_other = torch.zeros((size, self.Nj))
        self.s = torch.zeros(size)
        self.K = torch.zeros(size)
        self.Y = torch.zeros(size)
        self.r = torch.zeros(size)
        self.w = torch.zeros(size)
        self.r_k = torch.zeros(size)
        self.w_k = torch.zeros(size)
        self.Lambda = torch.zeros(size)
        self.dWj =  torch.zeros(size)
        self.dWj_other = torch.zeros((size, self.Nj))
        self.dWk =  torch.zeros(size)

        self.model = model
        self.max_batch_size = self.batch_size



    def capital_labor(self):
        self.K = torch.minimum(torch.maximum(1 / self.Nj * self.g[:,:,1].sum(dim=1), torch.tensor(self.a0_low)), torch.tensor(self.a0_high))
        self.L =  torch.minimum(torch.maximum(1 / self.Nj *self.g[:,:,0].sum(dim=1), torch.tensor(self.z0_low)), torch.tensor(self.z0_high))

    def shadow_price(self):
        if not self.is_competitive:
            self.r_k = self.alpha * (self.alpha - 1) * self.A * self.K ** (self.alpha - 2) * self.L ** (1 - self.alpha)
            self.w_k = self.alpha * (1 - self.alpha) * self.A * self.K ** (self.alpha - 1) * self.L ** (-self.alpha)
            self.Lambda = 1 / self.Nj * (self.xi_other * (self.r_k.unsqueeze(1) * self.g + self.w_k.unsqueeze(1) * self.z.unsqueeze(1))).sum(dim=1)

    def prices(self):
        self.Y = self.A * self.K ** self.alpha * self.L ** (1 - self.alpha)
        self.r = self.alpha * self.Y / self.K - self.delta
        self.w = (1 - self.alpha) * self.Y / self.L

    def saving(self, xi_fun):
        self.s = ((self.r + self.deathrate) * self.a + self.w * self.z
                  - self.consumption(xi_fun(self.model, self.z, self.a, self.g, self.A)))

    def dist_drift(self, xi_fun):
        self.xi_other= xi_fun(self.model, self.g[:,:,0], self.g[:,:,1], self.g, self.A)
        self.ga_drift = ((self.r.unsqueeze(1) + self.deathrate) * self.g[:,:,1] + self.w.unsqueeze(1) * self.g[:,:,0]
                         - self.consumption(self.xi_other))

    def consumption(self, xip):
        return xip ** (-1 / self.gamma)

    def utils(self, c):
        return torch.maximum(c, torch.tensor(0.001)) ** (1-self.gamma) / (1-self.gamma)


    def simulate_dist_one_step(self, xi_fun, dWj_other):
        self.capital_labor()
        self.prices()
        self.dist_drift(xi_fun)
        self.g[:, :, 0] = self.simulate_z_one_step(self.g[:, :, 0], dWj_other)
        self.g[:, :, 1] = self.simulate_a_one_step(self.g[:, :, 1], self.ga_drift)


    def simulate_states_one_step(self, xi_fun, dWj, dWk, dWj_other):
        """

        simulate individual state to get invariant distribution, given value function
        approximated by neural net work

        """
        self.capital_labor()
        self.prices()
        self.saving(xi_fun)
        self.dist_drift(xi_fun)

        self.z = self.simulate_z_one_step(self.z, dWj)
        self.A = self.simulate_A_one_step(self.A, dWk)
        self.a = self.simulate_a_one_step(self.a, self.s)
        self.g[:, :,0] = self.simulate_z_one_step(self.g[:, :,0], dWj_other)
        self.g[:, :,1] = self.simulate_a_one_step(self.g[:, :,1], self.ga_drift)





    def simulate_a_one_step(self, a_cur, s_cur):
        return torch.minimum(
            torch.maximum(a_cur + s_cur * self.dt, torch.tensor(self.a0_low)),
            torch.tensor(self.a0_high))



    def simulate_Exi_one_step(self):
        """
        Simulate E(dVa)/dt by one step, given the initial values (z,a,g,A), r, Lambda
        Returns: E(dVa)/dt

        """

        self.Exi = -(self.Lambda - (self.rho + self.deathrate - self.r) * self.xi)


