import numpy as np
from .calibration import Calibration
import torch
from typing import Union, Sequence, Tuple, Optional

class Simulation(Calibration):
    def __init__(self):
        super().__init__()
        self.z0 = torch.zeros(self.Ns)
        self.A0 = torch.zeros(self.Ns)
        self.a0 = torch.zeros(self.Ns)
        self.g0 = torch.zeros((self.Ns,self.Nj))
        self.z_path = torch.zeros((self.Ns, self.Nt))
        self.A_path = torch.zeros((self.Ns, self.Nt))
        self.dWj_path = torch.zeros((self.Ns, self.Nt - 1))
        self.dWk_path = torch.zeros((self.Ns, self.Nt - 1))

    def initial_sample(self):
        self.z0.uniform_(self.z0_low, self.z0_high)
        self.A0.uniform_(self.A0_low, self.A0_high)
        self.a0.uniform_(self.a0_low, self.a0_high)
        self.initial_g()



    def initial_g(self):
        KYratio = torch.zeros(self.Ns)
        KYratio.uniform_(self.KYratio_low, self.KYratio_high)
        # Y = A * K ** alpha * L ** (1-alpha)
        # Y = A * (KYratio * Y) ** alpha * L ** (1-alpha)
        Y = (self.A0 * KYratio ** self.alpha * self.L ** (1-self.alpha)) ** (1 / (1-self.alpha))
        K = Y * KYratio
        g = torch.from_numpy(lhs_uniform(self.Ns,self.Nj, self.a0_low, self.a0_high, self.seed_numpy)).to(self.device)

        self.g0 = g * (K / (g.sum(dim=1) / self.Nj)).unsqueeze(dim=1)

    def set_exo_shocks(self):
        self.z_path[:, 0] = self.z0
        self.A_path[:, 0] = self.A0
        self.dWj_path= (torch.normal(0, self.dt, size=(self.Ns,self.Nt - 1), device=self.device_str))
        self.dWk_path= (torch.normal(0, self.dt, size=(self.Ns,self.Nt - 1), device=self.device_str))
        for ti in range(self.Nt - 1):
            self.A_path[:, ti + 1] = torch.minimum(
                torch.maximum(self.A_path[:, ti] + self.eta * (self.hat_A - self.A_path[:, ti]) * self.dt + self.sigma * self.dWk_path[:, ti],
                              torch.tensor(self.A0_low)),
                torch.tensor(self.A0_high))
            self.z_path[:, ti + 1] = torch.minimum(
                torch.maximum(self.z_path[:, ti] + self.theta * (self.hat_z - self.z_path[:, ti]) * self.dt + self.sigma_z * self.dWj_path[:, ti],
                              torch.tensor(self.z0_low)),
                torch.tensor(self.z0_high))

def lhs_uniform(N: int, J:int,
                a_low: float,
                a_high: float,
                seed: Optional[int] = None) -> np.ndarray:
    """
    Latin Hypercube Sampling (basic) for a J-dim vector g.

    Args:
        N: number of LHS samples to produce.
        J: number of elements in each LHS sample.
        a_low: scalar.
        a_high: scalar.
        seed: optional RNG seed for reproducibility.

    Returns:
        samples: numpy array shape (N, J) where each row is one sample g_i,
                 and each column is stratified by LHS over [a_low[j], a_high[j]].
    """
    rng = np.random.default_rng(seed)

    samples = np.zeros((N, J))

    for j in range(J):
        # create one sample in each of the N strata for dimension j
        # sample inside strata: (i + u_i) / N for i=0..N-1 where u_i ~ U(0,1)
        u = rng.random(N)
        pts = (np.arange(N) + u) / N   # each value in ((i)/N, (i+1)/N)
        # randomly permute which sample index gets which stratum
        rng.shuffle(pts)
        samples[:, j] = a_low + pts * (a_high - a_low)

    return samples
