"""Amortized and global VMD parameter encoders."""

import torch
import torch.nn as nn


class AmortizedVMDEncoder(nn.Module):
    """Predict per-window bandwidths α and centre frequencies ω from a window.

    α is parameterised as ``α_base · exp(α_log_range · tanh(latent))``,
    mirroring the original AT-VMD implementation.
    """

    def __init__(self, modes: int, alpha_base: float = 2000.0,
                 alpha_log_range: float = 3.0,
                 hidden_dim: int = 32, predict_alpha: bool = False):
        super().__init__()
        self.modes = modes
        self.alpha_base = alpha_base
        self.alpha_log_range = alpha_log_range
        self.predict_alpha = predict_alpha
        omega_base = torch.arange(modes, dtype=torch.float32) * (0.5 / modes)
        self.register_buffer("omega_base", omega_base)
        self.omega_half_bin = 0.25 / modes
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.omega_head = nn.Linear(hidden_dim, modes)
        nn.init.zeros_(self.omega_head.weight)
        nn.init.zeros_(self.omega_head.bias)
        if predict_alpha:
            self.alpha_head = nn.Linear(hidden_dim, modes)
            nn.init.zeros_(self.alpha_head.weight)
            nn.init.zeros_(self.alpha_head.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B = x.size(0)
        features = self.encoder(x.unsqueeze(1))
        omega_latent = self.omega_head(features)
        omega = (self.omega_base.to(x).unsqueeze(0)
                 + self.omega_half_bin * torch.tanh(omega_latent)
                 ).clamp(min=0.0, max=0.5 - 1e-6)
        if self.predict_alpha:
            alpha = self.alpha_base * torch.exp(
                self.alpha_log_range * torch.tanh(self.alpha_head(features)))
        else:
            alpha = torch.full((B, self.modes), self.alpha_base,
                               device=x.device, dtype=x.dtype)
        return {"alpha": alpha, "omega": omega}


class GlobalOmegaEncoder(nn.Module):
    """One learned set of VMD controls shared by all input windows."""

    def __init__(self, modes: int, alpha_base: float = 2000.0,
                 alpha_log_range: float = 3.0,
                 predict_alpha: bool = False):
        super().__init__()
        self.modes = modes
        self.alpha_base = alpha_base
        self.alpha_log_range = alpha_log_range
        self.predict_alpha = predict_alpha
        self.omega_latent = nn.Parameter(torch.zeros(modes))
        omega_base = torch.arange(modes, dtype=torch.float32) * (0.5 / modes)
        self.register_buffer("omega_base", omega_base)
        self.omega_half_bin = 0.25 / modes
        if predict_alpha:
            self.alpha_latent = nn.Parameter(torch.zeros(modes))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = x.size(0)
        if self.predict_alpha:
            alpha = (
                self.alpha_base
                * torch.exp(self.alpha_log_range * torch.tanh(self.alpha_latent))
            ).unsqueeze(0).expand(batch, -1)
        else:
            alpha = torch.full(
                (batch, self.modes), self.alpha_base, device=x.device, dtype=x.dtype)
        omega = (
            self.omega_base.to(x)
            + self.omega_half_bin * torch.tanh(self.omega_latent).to(x)
        ).clamp(min=0.0, max=0.5 - 1e-6)
        return {
            "alpha": alpha,
            "omega": omega.unsqueeze(0).expand(batch, -1),
        }
