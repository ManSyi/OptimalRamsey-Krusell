import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from ..classes import Calibration as cal
from ..classes import Simulation as sml
from typing import Callable
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, cal):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(cal.num_inputs, cal.num_neurons),
            nn.ReLU(),
            nn.Linear(cal.num_neurons, cal.num_neurons),
            nn.ReLU(),
            nn.Linear(cal.num_neurons, cal.num_outputs),
        )

    def forward(self, x):
        x = self.flatten(x)
        value = self.linear_relu_stack(x)
        return value

def xi_fun(model: NeuralNetwork, z,a,g,A):
    with torch.no_grad():
        output = model(sml.toSample_x(z,a,g,A)).squeeze()
    return sml.toOutput_y(output,len(z.shape), len(A.shape), z,A)

def xi_fun_with_grad(model: NeuralNetwork, z,a,g,A, sample_index):
    return model(sml.toSample_x(z,a,g,A).squeeze().index_select(0, sample_index))