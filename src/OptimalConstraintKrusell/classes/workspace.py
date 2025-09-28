import numpy as np
from .calibration import Calibration
from .simulation import Simulation
import torch

class WorkSpace(Calibration):
    def __init__(self, model, num_sample_z, num_sample_A):
        super().__init__()
        self.a = torch.zeros((num_sample_z, num_sample_A))
        self.g = torch.zeros((num_sample_z, num_sample_A))
        self.xi = torch.zeros((num_sample_z, num_sample_A))

        self.xi_z = torch.zeros((num_sample_z, num_sample_A))
        self.xi_A = torch.zeros((num_sample_z, num_sample_A))

        self.varsigma = torch.zeros((num_sample_z, num_sample_A))
        self.varsigma_z = torch.zeros((num_sample_z, num_sample_A))
        self.xi_a = torch.zeros((num_sample_z, num_sample_A))
        self.c =  torch.zeros((num_sample_z, num_sample_A))
        self.cF =  torch.zeros((num_sample_z, num_sample_A))
        self.cB =  torch.zeros((num_sample_z, num_sample_A))
        self.c0 =  torch.zeros((num_sample_z, num_sample_A))
        self.sF =  torch.zeros((num_sample_z, num_sample_A))
        self.sB =  torch.zeros((num_sample_z, num_sample_A))
        self.xiF = torch.zeros((num_sample_z, num_sample_A))
        self.xiB = torch.zeros((num_sample_z, num_sample_A))
        self.xiC = torch.zeros((num_sample_z, num_sample_A))
        self.xip = torch.zeros((num_sample_z, num_sample_A))
        self.indF = torch.zeros((num_sample_z, num_sample_A))
        self.indB = torch.zeros((num_sample_z, num_sample_A))
        self.ind0 = torch.zeros((num_sample_z, num_sample_A))
        self.HF = torch.zeros((num_sample_z, num_sample_A))
        self.HB = torch.zeros((num_sample_z, num_sample_A))
        self.H0 = torch.zeros((num_sample_z, num_sample_A))
        self.K = torch.zeros(num_sample_A)
        self.Y = torch.zeros(num_sample_A)
        self.r = torch.zeros(num_sample_A)
        self.w = torch.zeros(num_sample_A)
        self.r_k = torch.zeros(num_sample_A)
        self.w_k = torch.zeros(num_sample_A)
        self.Lambda = torch.zeros(num_sample_A)
        self.L = 0


        self.dWj =  torch.zeros(num_sample_z)
        self.dWk =  torch.zeros(num_sample_A)
        self.z =  torch.zeros(num_sample_z)
        self.A = torch.zeros(num_sample_A)
        self.z_drift = torch.zeros(num_sample_z)
        self.A_drift = torch.zeros(num_sample_A)

        self.dWj_path = torch.zeros((num_sample_z, self.Nt - 1))
        self.dWk_path = torch.zeros((num_sample_A, self.Nt - 1))
        self.z_path = torch.zeros((num_sample_z, self.Nt))
        self.A_path = torch.zeros((num_sample_A, self.Nt))
        self.a_path = torch.zeros((num_sample_z, num_sample_A, self.Nt))
        self.z_drift_path = torch.zeros((num_sample_z, self.Nt - 1))
        self.A_drift_path = torch.zeros((num_sample_A, self.Nt - 1))
        self.xi_infinitesimal_path = torch.zeros((num_sample_z, num_sample_A, self.Nt - 1))

        self.model = model

    def set_exo_shocks(self, sml:Simulation):
        self.z_path[:, 0] = sml.z0
        self.A_path[:, 0] = sml.A0
        for ti in range(self.Nt - 1):
            self.dWj = (torch.normal(0, (ti + 1) * self.dt, size=self.z.shape, device="cuda")
                        - torch.normal(0, ti * self.dt, size=self.z.shape, device="cuda"))
            self.dWk = (torch.normal(0, (ti + 1) * self.dt, size=self.A.shape, device="cuda")
                        - torch.normal(0, ti * self.dt, size=self.A.shape, device="cuda"))

            self.z = self.z_path[:, ti]
            self.A = self.A_path[:, ti]
            self.siml_z()
            self.siml_A()

            self.z_drift_path[:, ti] = self.z_drift
            self.A_drift_path[:, ti] = self.A_drift
            self.dWj_path[:, ti] = self.dWj
            self.dWk_path[:, ti] = self.dWk
            self.z_path[:, ti + 1] = self.z
            self.A_path[:, ti + 1] = self.A

    def initial_end(self, xi_fun):
        self.xi = xi_fun(self.model, self.z_path[:,-1], self.a_path[:,:,-1], self.a_path[:,:,-1], self.A_path[:,-1])

    def initial_start(self, sml:Simulation):
        self.a_path[:,:,0] = sml.a0.repeat(self.Nk, 1).permute(1, 0)
        # self.A = self.A_path[:,0]
        # self.z = self.z_path[:,0]
        # self.z_drift = self.z_drift_path[:, 0]
        # self.A_drift = self.A_drift_path[:, 0]
        self.g = self.a_path[:,:,0]
        # self.xi = xi_fun(self.model, self.z, self.a,  self.a, self.A)
        #
        # self.dWj = self.dWj_path[:, 0]
        # self.dWk = self.dWk_path[:, 0]
        #
        #
        # self.upwind_z_A(xi_fun)
        # # initiate xip
        # self.capital_labor()
        # self.prices()
        # self.upwind(xi_fun)
        # self.risk_loadings()
        # self.shadow_price()
        # self.xi_drift_path[:,:,0] = self.utils() - self.Lambda.unsqueeze(0) * (self.K.unsqueeze(0) - self.a)
        # self.xi_volatility_path[:,:,0] = self.varsigma_z * self.dWj.unsqueeze(1) + self.varsigma * self.dWk.unsqueeze(0)


    def upwind_z_A(self, xi_fun):

        self.xiF = torch.maximum((xi_fun(self.model, self.z + self.delta_diff, self.a, self.a, self.A)
                                 - xi_fun(self.model, self.z, self.a, self.a, self.A)) / self.delta_diff, torch.tensor(1e-4))
        self.xiB = torch.maximum(-(xi_fun(self.model, self.z - self.delta_diff, self.a, self.a, self.A)
                                  - xi_fun(self.model, self.z, self.a, self.a, self.A)) / self.delta_diff, torch.tensor(1e-4))

        self.xi_z = self.xiF * (self.z_drift > 0).unsqueeze(1) + self.xiB * (self.z_drift < 0).unsqueeze(1)

        self.xiF = torch.maximum((xi_fun(self.model, self.z, self.a, self.a, self.A + self.delta_diff)
                    - xi_fun(self.model, self.z, self.a, self.a, self.A)) / self.delta_diff, torch.tensor(1e-4))
        self.xiB = torch.maximum(-(xi_fun(self.model, self.z, self.a, self.a, self.A - self.delta_diff)
                     - xi_fun(self.model, self.z, self.a, self.a, self.A)) / self.delta_diff, torch.tensor(1e-4))

        self.xi_A = self.xiF * (self.A_drift > 0).unsqueeze(0) + self.xiB * (self.A_drift < 0).unsqueeze(0)


    def upwind(self, xi_fun):
        self.xiF = torch.maximum(xi_fun(self.model, self.z, self.a + self.delta_diff, self.a, self.A)
                  - xi_fun(self.model, self.z, self.a, self.a, self.A) / self.delta_diff, torch.tensor(1e-4))
        self.xiB = torch.maximum(-(xi_fun(self.model, self.z, self.a - self.delta_diff, self.a, self.A)
                   - xi_fun(self.model, self.z, self.a, self.a, self.A)) / self.delta_diff, torch.tensor(1e-4))
        self.cF = self.consumption(self.xiF)
        self.cB = self.consumption(self.xiB)
        self.c0 = self.r.unsqueeze(0) * self.a + self.w.unsqueeze(0) * self.z.unsqueeze(1)
        self.sF = self.c0 - self.cF
        self.sB = self.c0 - self.cB

        self.HF = self.utils(self.cF) + self.sF * self.xiF
        self.HB = self.utils(self.cB) + self.sB * self.xiB
        self.H0 = self.utils(self.c0)



        self.indF = torch.logical_and(self.sF > 0, torch.logical_and(
            torch.logical_or(self.sB > 0, self.HF > self.HB), self.HF > self.H0))

        self.indB = torch.logical_and(self.sB < 0, torch.logical_and(
            torch.logical_or(self.sF < 0, self.HB > self.HF), self.HB > self.H0))

        # self.indF = self.sF > 0
        # self.indB = self.sB < 0
        self.ind0 = torch.logical_and(~self.indF, ~self.indB)
        self.c = (self.cF * self.indF + self.cB * self.indB
                + self.c0 * self.ind0)
        self.marginal_utils()


    def capital_labor(self):
        self.K = 1 / self.Nj * torch.maximum(self.a.sum(0), torch.tensor(0.01))
        self.L = 1 / self.Nj * self.z.sum(0)

    def shadow_price(self):
        self.Lambda = 1 / self.Nj * (self.xip * (self.r_k.unsqueeze(0) * self.a + self.w_k.unsqueeze(0) * self.z.unsqueeze(1))).sum(0)

    def prices(self):
        self.Y = torch.exp(self.A) * self.K ** self.alpha * self.L ** (1 - self.alpha)
        self.r = torch.maximum(self.alpha * self.Y / self.K - self.delta, torch.tensor(1e-5))
        self.w = (1 - self.alpha) * self.Y / self.L
        self.r_k = self.alpha * (self.alpha - 1) * torch.exp(self.A) * self.K ** (self.alpha - 2) * self.L ** (1 - self.alpha)
        self.w_k = self.alpha * (1 - self.alpha) * torch.exp(self.A) * self.K ** (self.alpha - 1) * self.L ** (-self.alpha)


    def consumption(self, xip):
        return xip ** (-1 / self.gamma)

    def marginal_utils(self):
        self.xip = torch.maximum(self.c, torch.tensor(0.001)) ** (-self.gamma)

    def utils(self, c):
        return torch.maximum(c, torch.tensor(0.001)) ** (1-self.gamma) / (1-self.gamma)

    def risk_loadings(self):
        self.varsigma_z = - self.xi_z * self.sigma_z
        self.varsigma = - self.xi_A * self.sigma

    def siml_xi(self):
        self.risk_loadings()
        self.xi = 1 / (1 - self.rho  * self.dt) * (
                    self.xi - (self.utils(self.c) - self.Lambda.unsqueeze(0) * (self.K.unsqueeze(0) - self.a) )* self.dt
                    - self.varsigma_z * self.dWj.unsqueeze(1) - self.varsigma * self.dWk.unsqueeze(0))
        # self.xi = (
        #             (1 + self.rho  * self.dt) * self.xi - (self.utils() - self.Lambda.unsqueeze(0) * (self.K.unsqueeze(0) - self.a) )* self.dt
        #             - self.varsigma_z * self.dWj.unsqueeze(1) - self.varsigma * self.dWk.unsqueeze(0))
    def siml_a(self):
        self.a = torch.minimum(torch.maximum(self.a + (self.sF * self.indF + self.sB * self.indB) * self.dt, torch.tensor(self.a0_low)),
                               torch.tensor(self.a0_high))

    def siml_z(self):
        self.z_drift = self.theta * (self.hat_z - self.z)
        self.z = torch.minimum(torch.maximum(self.z + self.z_drift * self.dt + self.sigma_z * self.dWj, torch.tensor(self.z0_low)),
                               torch.tensor(self.z0_high))

    def siml_A(self):
        self.A_drift = self.eta * (self.hat_A - self.A)
        self.A = torch.minimum(torch.maximum(self.A + self.A_drift * self.dt + self.sigma * self.dWk, torch.tensor(self.A0_low)),
                               torch.tensor(self.A0_high))

    def simulate_one_step(self, ti, xi_fun):
        # self.dWj = (torch.normal(0, (ti + 1) * self.dt, size=self.z.shape, device="cuda")
        #           - torch.normal(0, ti * self.dt, size=self.z.shape, device="cuda"))
        # self.dWk = (torch.normal(0, (ti + 1) * self.dt, size=self.A.shape, device="cuda")
        #           - torch.normal(0, ti * self.dt, size=self.A.shape, device="cuda"))

        # construct reference, not copy
        self.dWj = self.dWj_path[:, ti]
        self.dWk = self.dWk_path[:, ti]
        self.z = self.z_path[:, ti]
        self.A = self.A_path[:, ti]
        self.a = self.a_path[:,:, ti]
        self.z_drift = self.z_drift_path[:, ti]
        self.A_drift = self.A_drift_path[:, ti]

        self.upwind_z_A(xi_fun)
        self.capital_labor()
        self.prices()
        self.upwind(xi_fun)
        self.risk_loadings()
        self.shadow_price()

        self.xi_infinitesimal_path[:,:,ti] = (-(self.utils(self.c) - self.Lambda.unsqueeze(0) * (self.K.unsqueeze(0) - self.a)) * self.dt
                                      - self.varsigma_z * self.dWj.unsqueeze(1) - self.varsigma * self.dWk.unsqueeze(0))

        # self.xi_infinitesimal_path[:,:,ti] = (-self.Lambda.unsqueeze(0) * self.dt
        #                               - self.varsigma_z * self.dWj.unsqueeze(1) - self.varsigma * self.dWk.unsqueeze(0))

        self.a_path[:,:,ti + 1] = torch.minimum(
            torch.maximum(self.a + (self.sF * self.indF + self.sB * self.indB) * self.dt, torch.tensor(self.a0_low)),
            torch.tensor(self.a0_high))
        # self.siml_a()

        # self.capital_labor()
        # self.prices()
        # self.shadow_price()
        # self.upwind_z_A(xi_fun)
        # self.siml_xi()
        # self.upwind(xi_fun)
        # self.siml_a()


    def simulate_one_step_reverse(self, ti):
        # xi_pre = f(xi_cur)
        self.xi = (1 / (1 + (self.rho + self.deathrate) * self.dt)
                   * (self.xi - self.xi_infinitesimal_path[:,:,ti] ))
