"""Reconstruction-constrained prediction objectives."""

from dataclasses import dataclass

import torch


def relative_reconstruction_error(signal, residual, eps=1e-12):
    if signal.ndim == 3:
        rec = residual.square().sum(dim=1) / signal.square().sum(dim=1).clamp_min(eps)
        return rec.mean(dim=-1)
    return residual.square().sum(dim=-1) / signal.square().sum(dim=-1).clamp_min(eps)


@dataclass
class ReconstructionConstraint:
    tolerance: float
    dual_lr: float
    value: float = 0.0
    max_value: float = 1_000.0

    def update(self, mean_reconstruction):
        violation = mean_reconstruction - self.tolerance
        self.value = min(self.max_value, max(0.0, self.value + self.dual_lr * violation))
        return violation
