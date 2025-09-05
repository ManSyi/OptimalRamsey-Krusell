import numpy as np
from .Calibration import Calibration

class Simulation(Calibration):
    dWj = 0
    dWk = 0
    g0 = 0
    z0 = 0
    a0 = 0
    def __init__(self):
        Calibration.__init__(self)
        self.dWj = np.zeros((self.Nt, self.Nj))
        self.dWk = np.zeros((self.Nt, self.Nk))
        x_k = np.zeros((self.Nt, self.Nk))
        x_j = np.zeros((self.Nt, self.Nj))
        for i in range(1,self.Nt):
            t = i * self.dt
            x_k[i,:] = np.random.normal(0, t, size=(1,self.Nk))
            x_j[i, :] = np.random.normal(0, t, size=(1, self.Nj))
        self.dWk = x_k[1:,:] - x_k[0:-1,:]
        self.dWj = x_j[1:,:] - x_j[0:-1,:]
        self.z0 = np.random.uniform(self.z0_low, self.z0_high, size=(self.Nj, 1))