"""VMDNet: convergent VMD → TCN forecaster."""

import torch
import torch.nn as nn

from ..vmd import ConvergentVMD
from ..forecasters import VMDNetForecaster


class VMDNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        K = int(config["modes"]); self.K = K
        self.alpha = float(config["alpha_base"])
        self.vmd = ConvergentVMD(K, int(config.get("vmdnet_max_iterations", 500)),
                                 float(config.get("vmdnet_tolerance", 1e-7)))
        self.fc = VMDNetForecaster(K, int(config["lookback"]), int(config["horizon"]),
                                   int(config["d_model"]), tuple(config["tcn_hidden"]),
                                   float(config["embedding_dropout"]), float(config["tcn_dropout"]),
                                   int(config["fusion_hidden"]))
        self.register_buffer("w0", torch.arange(K, dtype=torch.float32) * (0.5 / K))

    @torch.no_grad()
    def decompose(self, x):
        B, _, C = x.shape; ml, mo = [], []
        for c in range(C):
            a = torch.full((B, self.K), self.alpha, device=x.device)
            w = self.w0.unsqueeze(0).expand(B, -1).to(x)
            r = self.vmd.solve(x[:, :, c], a, w)
            ml.append(r.modes); mo.append(r.omega)
        return torch.stack(ml, dim=-1), torch.stack(mo, dim=-1)

    def forward(self, x, marks):
        if x.ndim == 2: x = x.unsqueeze(-1)
        modes, omega = self.decompose(x)
        out = []
        for c in range(modes.size(-1)):
            out.append(self.fc(modes[:, :, :, c], marks, omega[:, :, c]))
        return torch.stack(out, dim=-1)

    def forecast(self, modes, marks, omega):
        out = []
        for c in range(modes.size(-1)):
            out.append(self.fc(modes[:, :, :, c], marks, omega[:, :, c]))
        return torch.stack(out, dim=-1)
