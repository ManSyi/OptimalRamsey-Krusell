"""

PyTorch implementation of a permutation-invariant approximator y = V(z, a, g, A).

"""



# deep_set_pytorch.py
import torch
import torch.nn as nn
from sympy import Identity
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import random
from typing import Callable
import math

# ------------------------
# Utilities
# ------------------------
def make_mlp(input_dim: int,
             hidden_dim: int,
             output_dim: int,
             n_hidden_layers: int,
             hidden_activation: Callable[..., nn.Module] = nn.ReLU,
             final_activation: Callable[..., nn.Module] = nn.Softplus) -> nn.Sequential:
    """
    Create an MLP with:
      - n_hidden_layers hidden layers (each: Linear -> hidden_activation)
      - final linear layer -> final_activation
    If n_hidden_layers == 0, it is simply Linear(input_dim -> output_dim) + final_activation.
    """
    layers = []
    if n_hidden_layers <= 0:
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(final_activation())
    else:
        # first hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(hidden_activation())
        # additional hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(hidden_activation())
        # final linear
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(final_activation())
    return nn.Sequential(*layers)

# ------------------------
# DeepSet model
# ------------------------
class DeepSet(nn.Module):
    """
    DeepSet architecture:
      - phi: shared network applied to each element of g (element-wise)
      - aggregation: sum over element embeddings
      - rho: network that takes [aggregated_phi, z, a, A] and outputs scalar y
    """
    def __init__(self,
                 g_dim: int,
                 phi_hidden_dim: int,
                 phi_out_dim: int,
                 rho_hidden_dim: int,
                 n_phi_layers: int,
                 n_rho_layers: int):
        """
        g_dim: dimension of each element of g (usually 1 if g is real-valued vector)
        phi_out_dim: embedding size for each element after phi
        """
        super().__init__()
        # phi processes each scalar element of g independently (so input dim = g_dim)
        self.phi = make_mlp(input_dim=g_dim,
                            hidden_dim=phi_hidden_dim,
                            output_dim=phi_out_dim,
                            n_hidden_layers=n_phi_layers,
                            hidden_activation=nn.ReLU,
                            final_activation=nn.Softplus)

        # rho takes aggregated phi (phi_out_dim) plus (z,a,A) -> concatenate -> map to scalar nn.Identity
        rho_input_dim = phi_out_dim + 3  # z, a, A
        self.rho = make_mlp(input_dim=rho_input_dim,
                            hidden_dim=rho_hidden_dim,
                            output_dim=1,
                            n_hidden_layers=n_rho_layers,
                            hidden_activation=nn.ReLU,
                            final_activation=nn.Softplus)

    def forward(self, z: torch.Tensor, a: torch.Tensor, g: torch.Tensor, A: torch.Tensor):
        """
        Inputs:
          z: (batch,)
          a: (batch,)
          g: (batch, N, g_dim)
          A: (batch,)
        Output:
          y: (batch, 1)
        """
        # shape bookkeeping
        batch_size = g.shape[0]
        N = g.shape[1]
        if len(g.shape) == 2:
            g_dim = 1
        else:
            g_dim = g.shape[2]


        # Flatten elements to process with phi: (batch*N, g_dim)
        g_flat = g.reshape(batch_size * N, g_dim)
        phi_out = self.phi(g_flat)                          # (batch*N, phi_out_dim)
        phi_out = phi_out.view(batch_size, N, -1)           # (batch, N, phi_out_dim)

        # aggregate (sum) over N to ensure permutation invariance
        agg = phi_out.sum(dim=1)                            # (batch, phi_out_dim)

        # concatenate scalars (z,a,A) after unsqueezing and then rho
        if len(a.shape) == 1:
            scalars = torch.stack([z, a, A], dim=1)  # (batch, 3)
            rho_in = torch.cat([agg, scalars], dim=1)
            # (batch, phi_out_dim + 3)
            y = self.rho(rho_in).squeeze(-1)
        elif len(a.shape) == 2:
            cols = a.shape[1]
            scalars = torch.stack([z.repeat(cols), a.T.flatten(), A.repeat(cols)], dim=1)  # (batch, 3)
            rho_in = torch.cat([agg.repeat(cols,1), scalars], dim=1)
                             # (batch, 1) due to rho output dim = 1
            y = reshape_fortran(self.rho(rho_in), (batch_size,cols))
        return y


# Wrapper that outputs f0 + residual
class InitializedModel(nn.Module):
    def __init__(self, base_residual_net, f0_callable):
        super().__init__()
        self.residual = base_residual_net
        self.f0 = f0_callable

        # ensure residual initial output = 0 by zeroing final linear
        if hasattr(self.residual, "out") and isinstance(self.residual.out, nn.Linear):
            nn.init.zeros_(self.residual.out.weight)
            if self.residual.out.bias is not None:
                nn.init.zeros_(self.residual.out.bias)
        else:
            # If final layer has different name, find last linear
            last_linear = None
            for m in reversed(list(self.residual.modules())):
                if isinstance(m, nn.Linear):
                    last_linear = m
                    break
            if last_linear is not None:
                nn.init.zeros_(last_linear.weight)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)
            else:
                raise RuntimeError("Couldn't find final Linear in residual net; pass a net with final nn.Linear")

    def forward(self, z, a, g, A):
        f0_val = self.f0(z, a, g, A)          # shape (batch,)
        res = self.residual(z, a, g, A)       # shape (batch,) -> initially zero
        return (f0_val + res).squeeze(-1)

# -------------------------
# Utility: permute each sample's set independently
# -------------------------
def permute_batch_sets(g):
    """
    g: (batch, N, d)
    return: g_perm, perms
    """
    b, N, d = g.shape
    g_perm = torch.empty_like(g)
    perms = []
    for i in range(b):
        p = torch.randperm(N)
        perms.append(p)
        g_perm[i] = g[i, p]
    return g_perm, perms


class IndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        item = self.base[idx]
        # item may be tensor or tuple (x,y); return index as final element
        if isinstance(item, tuple):
            return (*item, idx)
        else:
            return item, idx

def xi_fun(model, z,a,g,A):
    with torch.no_grad():
        output = model(z,a,g,A)
    return output

def xi_fun_with_grad(model, z,a,g,A):
    return model(z,a,g,A)

def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

def adjust_learning_rate(optimizer, epoch, err1, err2, lr_old):
    lr = lr_old
    if abs(err1-err2) / err1 < 0.5 and err1 > err2:
        lr = lr_old / 10
        print(f"New learning rate = {lr}\n")
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr
    # if epoch % 15 == 0:
    #     lr = lr_old * (0.1 ** (epoch // 15))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    # else:
    #     lr = lr_old
    # return lr


