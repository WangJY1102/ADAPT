"""Shared-direction cosine initialization used by ADAPT."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SharedCosineBasis(nn.Module):
    """Task-local classifier rows share one initial direction, then adapt separately.

    basis_raw is an outer parameter. Do not wrap initial_weight in nn.Parameter:
    that would disconnect task heads from the shared initialization.
    """
    def __init__(self, dim=224, temperature=10.0, seed=20261024):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        anchor = torch.randn(dim, generator=generator)
        self.basis_raw = nn.Parameter(F.normalize(anchor, dim=0))
        self.temperature = float(temperature)  # fixed, not another learned scale
        self.dim = dim

    def initial_weight(self, n_way):
        return F.normalize(self.basis_raw, dim=0, eps=1e-8)[None, :].repeat(n_way, 1)

    def forward(self, features, task_weight):
        return self.temperature * F.linear(
            F.normalize(features, dim=-1, eps=1e-8),
            F.normalize(task_weight, dim=-1, eps=1e-8))


class CosineClassifier(nn.Module):
    """Drop-in classifier for functional_call; no task bias or learned scale."""
    def __init__(self, dim=224, n_way=5, temperature=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_way, dim) / math.sqrt(dim), requires_grad=False)
        self.temperature = float(temperature)

    def forward(self, features):
        return self.temperature * F.linear(
            F.normalize(features, dim=-1, eps=1e-8),
            F.normalize(self.weight, dim=-1, eps=1e-8))
