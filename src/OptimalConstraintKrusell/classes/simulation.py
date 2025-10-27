import numpy as np
from .calibration import Calibration
import torch
from typing import Union, Sequence, Tuple, Optional

class Simulation(Calibration):
    def __init__(self, Ns):
        super().__init__()
        self.z0 = torch.zeros(Ns)
        self.A0 = torch.zeros(Ns)
        self.a0 = torch.zeros(Ns)
        self.g0 = torch.zeros((Ns,self.Nj, 2))

        self.z = torch.zeros(Ns)
        self.A = torch.zeros(Ns)
        self.a = torch.zeros(Ns)
        self.g = torch.zeros((Ns,self.Nj, 2))
        self.sample_size = Ns



    def initial_sample(self):
        self.z0.uniform_(self.z0_low, self.z0_high)
        self.A0.uniform_(self.A0_low, self.A0_high)
        self.a0.uniform_(self.a0_low, self.a0_high)

        #self.g0[:, :, 0] =self.z0.repeat(self.sample_size, 1)
        #self.g0[:, :, 1] = self.a0.repeat(self.sample_size, 1)
        self.initial_dist()

        self.z = self.z0.clone()
        self.a = self.a0.clone()
        self.g = self.g0.clone()
        self.A = self.A0.clone()



    def initial_dist(self):
        self.g0[:,:,0] = torch.from_numpy(lhs_uniform(self.sample_size, self.Nj, self.z0_low, self.z0_high, self.seed_numpy)).to(
            self.device)
        # self.g0[:, :, 0] = self.g0[:,:,0] / (self.g0[:,:,0].sum(dim=1)/self.Nj).unsqueeze(dim=1)

        KYratio = torch.zeros(self.sample_size)
        KYratio.uniform_(self.KYratio_low, self.KYratio_high)
        # Y = A * K ** alpha * L ** (1-alpha)
        # Y = A * (KYratio * Y) ** alpha * L ** (1-alpha)
        Y = (self.A0 * KYratio ** self.alpha * self.L ** (1-self.alpha)) ** (1 / (1-self.alpha))
        K = Y * KYratio
        g = torch.from_numpy(lhs_uniform(self.sample_size,self.Nj, self.a0_low, self.a0_high, self.seed_numpy)).to(self.device)

        self.g0[:,:,1] = g * (K / (g.sum(dim=1) / self.Nj)).unsqueeze(dim=1)




def lhs_uniform(N: int, J:int,
                low: float,
                high: float,
                seed: Optional[int] = None) -> np.ndarray:
    """
    Latin Hypercube Sampling (basic) for a J-dim vector g.

    Args:
        N: number of LHS samples to produce.
        J: number of elements in each LHS sample.
        low: scalar.
        high: scalar.
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
        samples[:, j] = low + pts * (high - low)

    return samples

