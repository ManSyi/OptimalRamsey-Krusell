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
        self.xi_a = torch.zeros(self.batch_size)
        self.xi_z = torch.zeros(self.batch_size)
        self.xi_A = torch.zeros(self.batch_size)

        self.varsigma = torch.zeros(self.batch_size)
        self.varsigma_z = torch.zeros(self.batch_size)

        self.c =  torch.zeros(self.batch_size)
        self.s_other = torch.zeros((self.batch_size, self.Nj))
        self.xi_other = torch.zeros((self.batch_size, self.Nj))
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

        # self.xi_z = torch.maximum((xi_fun(self.model, self.z + self.delta_diff, self.a, self.g, self.A)
        #                          - xi_fun(self.model, self.z - self.delta_diff, self.a, self.g, self.A))
        #                           / (self.delta_diff * 2), torch.tensor(1e-5))
        #
        # self.xi_A = torch.maximum((xi_fun(self.model, self.z, self.a, self.g, self.A + self.delta_diff)
        #              - xi_fun(self.model, self.z, self.a, self.g, self.A - self.delta_diff))
        #                           / (self.delta_diff * 2), torch.tensor(1e-5))

        self.xi_z = ((xi_fun(self.model, self.z + self.delta_diff, self.a, self.g, self.A)
                                 - xi_fun(self.model, self.z - self.delta_diff, self.a, self.g, self.A))
                                  / (self.delta_diff * 2))
        self.xi_A = ((xi_fun(self.model, self.z, self.a, self.g, self.A + self.delta_diff)
                     - xi_fun(self.model, self.z, self.a, self.g, self.A - self.delta_diff))
                                  / (self.delta_diff * 2))


    def derivative_a(self, xi_fun):

        # self.xi_a = torch.maximum(xi_fun(self.model, self.z, self.a + self.delta_diff, self.g, self.A)
        #           - xi_fun(self.model, self.z, self.a - self.delta_diff, self.g, self.A)
        #                          / (self.delta_diff * 2), torch.tensor(1e-5))
        # self.c = self.consumption(self.xi_a)
        # self.s = ((self.r + self.deathrate) * self.a + self.w * self.z
        #           - self.c)
        #
        # self.xi_a_other = torch.maximum(xi_fun(self.model, self.z, self.g + self.delta_diff, self.g, self.A)
        #           - xi_fun(self.model, self.z, self.g - self.delta_diff, self.g, self.A)
        #                          / (self.delta_diff * 2), torch.tensor(1e-5))
        # self.s_other = ((self.r.unsqueeze(1) + self.deathrate) * self.g + self.w.unsqueeze(1) * self.z.unsqueeze(1)
        #                 - self.consumption(self.xi_a_other))

        self.s = ((self.r + self.deathrate) * self.a + self.w * self.z
                  - self.consumption(xi_fun(self.model, self.z, self.a, self.g, self.A)))
        self.xi_other= xi_fun(self.model, self.z, self.g, self.g, self.A)
        self.s_other = ((self.r.unsqueeze(1) + self.deathrate) * self.g + self.w.unsqueeze(1) * self.z.unsqueeze(1)
                        - self.consumption(self.xi_other))



    def capital_labor(self):
        self.K = 1 / self.Nj * torch.maximum(self.g.sum(1), torch.tensor(0.01))
        self.L = 1.0

    def shadow_price(self):
        if not self.is_competitive:
            self.r_k = self.alpha * (self.alpha - 1) * self.A * self.K ** (self.alpha - 2) * self.L ** (1 - self.alpha)
            self.w_k = self.alpha * (1 - self.alpha) * self.A * self.K ** (self.alpha - 1) * self.L ** (-self.alpha)
            self.Lambda = 1 / self.Nj * (self.xi_other * (self.r_k.unsqueeze(1) * self.g + self.w_k.unsqueeze(1) * self.z.unsqueeze(1))).sum(dim=1)

    def prices(self):
        self.Y = self.A * self.K ** self.alpha * self.L ** (1 - self.alpha)
        self.r = self.alpha * self.Y / self.K - self.delta
        self.w = (1 - self.alpha) * self.Y / self.L



    def consumption(self, xip):
        return xip ** (-1 / self.gamma)

    def utils(self, c):
        return torch.maximum(c, torch.tensor(0.001)) ** (1-self.gamma) / (1-self.gamma)

    def risk_loadings(self):
        self.varsigma_z = - self.xi_z * self.sigma_z
        self.varsigma = - self.xi_A * self.sigma


    def simulate_invariant_dist_one_step(self, xi_fun):
        """

        simulate individual state to get invariant distribution, given value function
        approximated by neural net work

        """
        self.capital_labor()
        self.prices()

        self.s_other = ((self.r.unsqueeze(1) + self.deathrate) * self.g + self.w.unsqueeze(1) * self.z.unsqueeze(1)
                              - self.consumption(xi_fun(self.model, self.z, self.g, self.g, self.A)))
        self.g = torch.minimum(
            torch.maximum(self.g + self.s_other * self.dt, torch.tensor(self.a0_low)),
            torch.tensor(self.a0_high))



    def simulate_states_one_step(self, xi_fun):

        self.capital_labor()
        self.prices()
        self.derivative_a(xi_fun)

        self.a = torch.minimum(
            torch.maximum(self.a + self.s * self.dt, torch.tensor(self.a0_low)),
            torch.tensor(self.a0_high))
        self.g = torch.minimum(
            torch.maximum(self.g + self.s_other * self.dt, torch.tensor(self.a0_low)),
            torch.tensor(self.a0_high))



    def simulate_xi_one_step(self, xi_fun):

        """
        Simulate xi and state variable by one step, given the initial values.
        Returns: xi, a, g, in the next step

        """
        self.capital_labor()
        self.prices()
        self.derivative_a(xi_fun)

        self.shadow_price()
        self.derivative_z_A(xi_fun)
        self.risk_loadings()

        # self.xi = ((1 + (self.rho + self.deathrate)  * self.dt) * self.xi - (self.utils(self.c) - self.Lambda * (self.K - self.a) )* self.dt
        #               - self.varsigma_z * self.dWj - self.varsigma * self.dWk)

        self.xi = torch.maximum((self.xi - (self.Lambda - (self.rho + self.deathrate -self.r) * self.xi)
                   * self.dt - self.varsigma_z * self.dWj - self.varsigma * self.dWk), torch.tensor(1e-6))

        self.a = torch.minimum(
            torch.maximum(self.a + self.s * self.dt, torch.tensor(self.a0_low)),
            torch.tensor(self.a0_high))
        self.g = torch.minimum(
            torch.maximum(self.g + self.s_other * self.dt, torch.tensor(self.a0_low)),
            torch.tensor(self.a0_high))
