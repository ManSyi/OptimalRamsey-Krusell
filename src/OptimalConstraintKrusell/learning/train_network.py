import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset

# -------------------------
# Small MLP helper
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, act=nn.SiLU, use_layernorm=True, dropout=0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------
# Set encoder for fixed N
# DeepSets: phi -> sum -> rho
# Attention pooling: phi -> attention weights -> weighted sum -> rho
# -------------------------
class DeepSetsEncoder(nn.Module):
    def __init__(self, elem_dim=1, out_dim=128, phi_hidden=(64,64)):
        super().__init__()
        self.phi = MLP(elem_dim, list(phi_hidden) + [out_dim])
    def forward(self, g):  # g: (batch, N, elem_dim)
        b, N, d = g.shape
        x = g.view(b * N, d)
        phi_x = self.phi(x).view(b, N, -1)
        # sum pooling
        pooled = phi_x.sum(dim=1)  # (b, out_dim)
        return pooled

class AttentionSetEncoder(nn.Module):
    def __init__(self, elem_dim=1, out_dim=128, phi_hidden=(64,64), att_hidden=64):
        super().__init__()
        self.phi = MLP(elem_dim, list(phi_hidden) + [out_dim])
        # attention scorer maps phi(elem) -> scalar
        self.att_mlp = MLP(out_dim, [att_hidden])
        self.att_proj = nn.Linear(att_hidden, 1)
    def forward(self, g):  # g: (batch, N, elem_dim)
        b, N, d = g.shape
        x = g.view(b * N, d)
        phi_x = self.phi(x).view(b, N, -1)  # (b, N, out_dim)
        h = self.att_mlp(phi_x.view(b * N, -1))  # (b*N, att_hidden)
        logits = self.att_proj(h).view(b, N)     # (b, N)
        weights = F.softmax(logits, dim=1).unsqueeze(-1)  # (b, N, 1)
        pooled = (phi_x * weights).sum(dim=1)  # (b, out_dim)
        return pooled

# -------------------------
# Full Value net: concatenates scalar proj with set repr
# -------------------------
class ValueNet(nn.Module):
    def __init__(self, encoder, scalar_hidden=128, head_hidden=(128,64)):
        """
        encoder: module mapping (batch,N,elem_dim) -> (batch, set_dim)
        """
        super().__init__()
        self.encoder = encoder
        set_dim = None
        # infer set_dim by a small forward pass (lazy)
        dummy = torch.randn(1, 2, 1)  # (batch, N_dummy, elem_dim_dummy)
        with torch.no_grad():
            set_dim = encoder(dummy).shape[-1]
        self.scalar_proj = MLP(3, [scalar_hidden])
        head_in = set_dim + scalar_hidden
        self.head = MLP(head_in, list(head_hidden))
        self.out = nn.Linear(head_hidden[-1], 1)

    def forward(self, z, a, g, A):
        """
        z,a,A: shape (batch,) or (batch,1)
        g: (batch, N) or (batch, N, elem_dim)
        returns y: (batch,)
        """
        # canonicalize
        z = torch.as_tensor(z)
        a = torch.as_tensor(a)
        A = torch.as_tensor(A)
        g = torch.as_tensor(g)

        if g.dim() == 2:  # (batch,N) -> (batch,N,1)
            g = g.unsqueeze(-1)

        def ensure_col(x):
            if x.dim() == 0:
                return x.unsqueeze(0).unsqueeze(-1)
            if x.dim() == 1:
                return x.unsqueeze(-1)
            return x

        zc = ensure_col(z).float()
        ac = ensure_col(a).float()
        Ac = ensure_col(A).float()
        scalars = torch.cat([zc, ac, Ac], dim=-1)  # (batch,3)

        set_repr = self.encoder(g.float())  # (batch, set_dim)
        srepr = self.scalar_proj(scalars)
        h = torch.cat([set_repr, srepr], dim=-1)
        h = self.head(h)
        y = self.out(h).squeeze(-1)
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
        return f0_val + res

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

def xi_fun(model:ValueNet, z,a,g,A):
    with torch.no_grad():
        output = model(z,a,g,A)
    return output

def xi_fun_with_grad(model:ValueNet, z,a,g,A):
    return model(z,a,g,A)