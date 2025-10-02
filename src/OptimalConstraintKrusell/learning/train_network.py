"""
sets_model.py

PyTorch implementation of a permutation-invariant approximator y = V(z, a, g, A).

Components:
 - ElementEncoder: encodes each element of g
 - ScalarEncoder: encodes scalar inputs z, a, A
 - DeepSetPooling: sum/mean pooling after phi (Deep Sets)
 - AttentionPooling: learned query + scaled dot-product attention pooling
 - SixLayerPredictor: the final 6-layer MLP (ReLU hidden, identity final)
 - SetNet: composes encoders, pooling, and predictor; selectable pooling type
 - Example synthetic dataset + training loop
"""

import math
import random
from typing import Literal, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------
# Utility / Small MLP module
# ---------------------------
class MLP(nn.Module):
    """Small configurable MLP.
    Uses ReLU for hidden activations and identity at the end (no activation).
    """
    def __init__(self, in_dim: int, hidden_dims: list[int], activate_final: bool = False):
        super().__init__()
        layers = []
        dims = [in_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            # final activation optionally excluded by caller
            if i < len(hidden_dims) - 1 or activate_final:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------
# Encoders for scalars and set elems
# -----------------------------------
class ElementEncoder(nn.Module):
    """Encodes each scalar element of g into a d-dim vector."""
    def __init__(self, d: int):
        super().__init__()
        # two-layer encoder: scalar -> hidden -> d
        self.net = nn.Sequential(
            nn.Linear(1, max(8, d)),
            nn.ReLU(),
            nn.Linear(max(8, d), d)
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        # g: (batch, N) or (batch, N, 1)
        if g.dim() == 2:
            g = g.unsqueeze(-1)  # (B, N, 1)
        B, N, _ = g.shape
        g_flat = g.view(B * N, 1)
        encoded = self.net(g_flat)            # (B*N, d)
        encoded = encoded.view(B, N, -1)     # (B, N, d)
        return encoded


class ScalarEncoder(nn.Module):
    """Encodes the scalars z, a, A into a single vector of dimension d."""
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, max(8, d)),
            nn.ReLU(),
            nn.Linear(max(8, d), d)
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # z,a,A: (batch,) or (batch,1)
        x = torch.stack([z.squeeze(-1), a.squeeze(-1), A.squeeze(-1)], dim=-1)  # (B,3)
        return self.net(x)  # (B,d)


# -----------------------------------
# Pooling modules (permutation invariant)
# -----------------------------------
class DeepSetPooling(nn.Module):
    """DeepSets pooling: apply phi (elementwise encoder) then sum/mean over set dimension."""
    def __init__(self, reduction: Literal["sum", "mean"] = "sum"):
        super().__init__()
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.reduction = reduction

    def forward(self, encoded_elems: torch.Tensor) -> torch.Tensor:
        # encoded_elems: (B, N, d)
        if self.reduction == "sum":
            return encoded_elems.sum(dim=1)   # (B, d)
        else:
            return encoded_elems.mean(dim=1)  # (B, d)


class AttentionPooling(nn.Module):
    """Attention pooling: learnable query vector Q (1 x d) attends to set elements (scaled dot-product).
    Produces a single pooled vector per batch element. Permutation invariant.
    """
    def __init__(self, d_model: int, temperature: Optional[float] = None):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_model))  # (1,d)
        # small projection for keys/values — optional but often helpful
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.temperature = temperature or math.sqrt(d_model)

    def forward(self, encoded_elems: torch.Tensor) -> torch.Tensor:
        # encoded_elems: (B, N, d)
        B, N, d = encoded_elems.shape
        K = self.key(encoded_elems)    # (B,N,d)
        V = self.value(encoded_elems)  # (B,N,d)
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # (B,1,d)
        # compute scaled dot-product: (B,1,N)
        scores = torch.matmul(q, K.transpose(-2, -1)) / self.temperature
        attn = torch.softmax(scores, dim=-1)  # (B,1,N)
        pooled = torch.matmul(attn, V).squeeze(1)  # (B,d)
        return pooled


# -----------------------------------
# Predictor: six-layer MLP
# -----------------------------------
class SixLayerPredictor(nn.Module):
    """Six linear layers in sequence. ReLU for hidden; identity at final layer."""
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int = 1):
        """
        hidden_dims should be length 5 (so total linear layers = 6 including final).
        Example: hidden_dims = [128, 128, 64, 64, 32] -> linear layers:
                 in_dim->128, 128->128, 128->64, 64->64, 64->32, 32->out_dim (six linears)
        """
        super().__init__()
        if len(hidden_dims) != 5:
            raise ValueError("hidden_dims length must be 5 to form exactly 6 Linear layers total.")
        layers = []
        dims = [in_dim] + hidden_dims + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            # Add ReLU for all but the final layer
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,) for scalar output


# -----------------------------------
# High-level model that composes pieces
# -----------------------------------
class SetNet(nn.Module):
    """Composed model: encodes scalars and set, pools, concatenates and predicts."""
    def __init__(self,
                 d_model: int = 64,
                 pooling: Literal["deepset_sum", "deepset_mean", "attention"] = "deepset_sum",
                 predictor_hidden_dims: Optional[list[int]] = None):
        super().__init__()
        self.element_encoder = ElementEncoder(d=d_model)
        self.scalar_encoder = ScalarEncoder(d=d_model)

        if pooling.startswith("deepset"):
            reduction = "mean" if pooling.endswith("mean") else "sum"
            self.pool = DeepSetPooling(reduction=reduction)
        elif pooling == "attention":
            self.pool = AttentionPooling(d_model=d_model)
        else:
            raise ValueError("pooling must be one of 'deepset_sum', 'deepset_mean', 'attention'")

        if predictor_hidden_dims is None:
            predictor_hidden_dims = [128, 128, 64, 32, 16]  # length 5 -> total 6 linear layers
        if len(predictor_hidden_dims) != 5:
            raise ValueError("predictor_hidden_dims length must be 5")

        # input to predictor is scalar-encoding (d) concatenated with pooled set (d) => 2d
        self.predictor = SixLayerPredictor(in_dim=2 * d_model,
                                           hidden_dims=predictor_hidden_dims,
                                           out_dim=1)

    def forward(self, z: torch.Tensor, a: torch.Tensor, g: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        z,a,A: (B,) or (B,1)
        g: (B,N)
        returns: (B,) predicted y
        """
        z_enc = self.scalar_encoder(z, a, A)   # (B, d)
        g_enc = self.element_encoder(g)        # (B, N, d)
        pooled = self.pool(g_enc)              # (B, d)
        feat = torch.cat([z_enc, pooled], dim=-1)  # (B, 2d)
        y_hat = self.predictor(feat)           # (B,)
        return y_hat


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

def xi_fun(model, z,a,g,A):
    with torch.no_grad():
        output = model(z,a,g,A)
    return output

def xi_fun_with_grad(model, z,a,g,A):
    return model(z,a,g,A)