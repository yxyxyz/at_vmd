"""DLinear baseline: trend-seasonal decomposition + linear projection."""

import torch
import torch.nn as nn


class DLinear(nn.Module):
    """DLinear (AAAI 2023): moving-average trend/seasonal + per-channel linear."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.lookback = int(config["lookback"])
        self.horizon = int(config["horizon"])
        kernel = int(config.get("dlinear_kernel", 25))
        self.pad = (kernel - 1) // 2
        self.pool = nn.AvgPool1d(kernel_size=kernel, stride=1)
        self.seasonal = nn.Linear(self.lookback, self.horizon)
        self.trend = nn.Linear(self.lookback, self.horizon)
        init_w = torch.full((self.horizon, self.lookback), 1.0 / self.lookback)
        self.seasonal.weight = nn.Parameter(init_w.clone())
        self.trend.weight = nn.Parameter(init_w.clone())

    def _decomp(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        front = x[:, :1, :].repeat(1, self.pad, 1)
        end = x[:, -1:, :].repeat(1, self.pad, 1)
        padded = torch.cat([front, x, end], dim=1)
        trend = self.pool(padded.permute(0, 2, 1)).permute(0, 2, 1)
        return x - trend, trend

    def forward(self, x: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        seasonal, trend = self._decomp(x)
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)
        out = self.seasonal(seasonal) + self.trend(trend)
        return out.permute(0, 2, 1)  # [B, H, C]
