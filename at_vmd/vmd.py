"""Convergent & differentiable VMD solver."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ADMMState:
    positive_spectrum: torch.Tensor
    frequencies: torch.Tensor
    modes_hat: torch.Tensor
    omega: torch.Tensor
    dual: torch.Tensor
    iterations: int = 0
    mode_change: float = float("inf")


@dataclass(frozen=True)
class VMDResult:
    modes: torch.Tensor
    residual: torch.Tensor
    omega: torch.Tensor
    iterations: int
    converged: bool
    mode_change: float


@dataclass
class DifferentiableADMMState:
    positive_spectrum: torch.Tensor
    frequencies: torch.Tensor
    modes_hat: torch.Tensor
    omega: torch.Tensor
    dual: torch.Tensor
    iterations: int = 0


class ConvergentVMD(nn.Module):
    """Convergent & differentiable VMD in one module."""

    def __init__(self, modes: int = 4, max_iterations: int = 500,
                 tolerance: float = 1e-7, tau: float = 0.0, eps: float = 1e-12):
        super().__init__()
        self.modes = modes
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.tau = tau
        self.eps = eps

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expand_alpha(self, alpha, batch_size):
        if alpha.shape == (batch_size,):
            return alpha.unsqueeze(1).expand(-1, self.modes)
        return alpha

    def _mirror(self, x):
        _, L = x.shape
        h = L // 2
        mirrored = torch.cat([torch.flip(x[:, :h], dims=[-1]), x,
                              torch.flip(x[:, h:], dims=[-1])], dim=-1)
        return mirrored, mirrored.size(-1)

    def _spectrum_to_time(self, modes_hat):
        ml = modes_hat.size(-1)
        pos = modes_hat[:, :, ml // 2:]
        full = torch.zeros_like(modes_hat)
        full[:, :, ml // 2:] = pos
        idx = torch.arange(1, ml // 2 + 1, device=modes_hat.device).flip(0)
        full[:, :, idx] = pos.conj()
        full[:, :, 0] = full[:, :, -1].conj()
        mirrored = torch.fft.ifft(
            torch.roll(full, shifts=-(ml // 2), dims=-1), dim=-1).real
        s, e = ml // 4, 3 * ml // 4
        return mirrored[:, :, s:e]

    def _one_iteration(self, pos, freqs, modes_hat, omega, dual, a):
        """Shared Gauss-Seidel iteration for both convergent & differentiable."""
        prev = modes_hat
        mode_sum = prev.sum(dim=1)
        new_modes, new_omega = [], []
        for k in range(self.modes):
            mode_sum = mode_sum - prev[:, k]
            denom = 1.0 + a[:, k:k+1] * (freqs.unsqueeze(0) - omega[:, k:k+1]).pow(2)
            m_hat = (pos - mode_sum - dual / 2.0) / denom
            new_modes.append(m_hat)
            mode_sum = mode_sum + m_hat
            ph = m_hat[:, m_hat.size(-1) // 2:]
            power = ph.abs().pow(2)
            psum = power.sum(dim=-1)
            centre = (freqs[freqs.numel() // 2:].unsqueeze(0) * power).sum(dim=-1) / psum.clamp_min(self.eps)
            new_omega.append(torch.where(psum > self.eps, centre, torch.zeros_like(centre)))
        modes_hat = torch.stack(new_modes, dim=1)
        omega = torch.stack(new_omega, dim=1)
        dual = dual + self.tau * (modes_hat.sum(dim=1) - pos)
        return modes_hat, omega, dual

    # ------------------------------------------------------------------
    # Convergent (no-grad) path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def solve(self, x, alpha, omega_init):
        B, L = x.shape
        mirrored, ml = self._mirror(x)
        freqs = torch.arange(1, ml + 1, device=x.device, dtype=x.dtype) / ml - 0.5 - 1.0 / ml
        spectrum = torch.roll(torch.fft.fft(mirrored, dim=-1), shifts=ml // 2, dims=-1)
        pos = spectrum.clone(); pos[:, :ml // 2] = 0
        modes_hat = torch.zeros(B, self.modes, ml, device=x.device, dtype=spectrum.dtype)
        omega = omega_init.detach().clone()
        dual = torch.zeros_like(pos)
        a = self._expand_alpha(alpha, B)
        for iteration in range(1, self.max_iterations + 1):
            prev = modes_hat
            modes_hat, omega, dual = self._one_iteration(pos, freqs, modes_hat, omega, dual, a)
            change = float((modes_hat - prev).abs().pow(2).sum(dim=(1,2)).div_(ml).mean().item())
            if change <= self.tolerance:
                break
        modes = self._spectrum_to_time(modes_hat)
        return VMDResult(modes=modes, residual=x - modes.sum(dim=1), omega=omega,
                         iterations=iteration, converged=change <= self.tolerance,
                         mode_change=change)

    # ------------------------------------------------------------------
    # Differentiable path
    # ------------------------------------------------------------------

    def _init_diff(self, x, omega):
        B, L = x.shape
        mirrored, ml = self._mirror(x)
        freqs = torch.arange(1, ml + 1, device=x.device, dtype=x.dtype) / ml - 0.5 - 1.0 / ml
        spectrum = torch.roll(torch.fft.fft(mirrored, dim=-1), shifts=ml // 2, dims=-1)
        pos = spectrum.clone(); pos[:, :ml // 2] = 0
        return DifferentiableADMMState(
            positive_spectrum=pos, frequencies=freqs,
            modes_hat=torch.zeros(B, self.modes, ml, device=x.device, dtype=spectrum.dtype),
            omega=omega, dual=torch.zeros_like(pos))

    def differentiable_step(self, state, alpha):
        B = state.positive_spectrum.size(0)
        a = self._expand_alpha(alpha, B)
        prev = state.modes_hat
        modes_hat, omega, dual = self._one_iteration(
            state.positive_spectrum, state.frequencies, prev, state.omega, state.dual, a)
        change = (modes_hat - prev).abs().pow(2).sum(dim=(1, 2)) / modes_hat.size(-1)
        return DifferentiableADMMState(
            positive_spectrum=state.positive_spectrum, frequencies=state.frequencies,
            modes_hat=modes_hat, omega=omega, dual=dual,
            iterations=state.iterations + 1), change

    def differentiable_decompose(self, x, alpha, omega, steps=1):
        if steps < 1:
            raise ValueError("steps must be at least 1")
        state = self._init_diff(x, omega)
        for _ in range(steps):
            state, _ = self.differentiable_step(state, alpha)
        modes = self._spectrum_to_time(state.modes_hat)
        return modes, x - modes.sum(dim=1), state.omega
