"""Raw TCN: VMDNetForecaster applied directly to raw input."""

import torch
import torch.nn as nn

from ..forecasters import VMDNetForecaster


class TCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = VMDNetForecaster(1, int(config["lookback"]), int(config["horizon"]),
                                   int(config["d_model"]), tuple(config["tcn_hidden"]),
                                   float(config["embedding_dropout"]), float(config["tcn_dropout"]),
                                   int(config["fusion_hidden"]))

    def forward(self, x, marks):
        if x.ndim == 2: x = x.unsqueeze(-1)
        B = x.size(0)
        return torch.stack([self.fc(x[:, :, c].unsqueeze(1), marks,
                                    torch.zeros(B, 1, device=x.device))
                            for c in range(x.size(-1))], dim=-1)
