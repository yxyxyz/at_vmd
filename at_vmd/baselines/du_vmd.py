"""DU-VMD: deep-unrolled VMD → TCN forecaster."""

import torch
import torch.nn as nn

from ..forecasters import VMDNetForecaster


class _DeepUnrolledVMD(nn.Module):
    def __init__(self, modes, iterations, eps=1e-12):
        super().__init__(); self.K = modes; self.iters = iterations; self.eps = eps

    def forward(self, x, alpha, omega):
        B, L = x.shape; h = L // 2
        mirrored = torch.cat([torch.flip(x[:, :h], dims=[-1]), x,
                              torch.flip(x[:, h:], dims=[-1])], dim=-1)
        ml = mirrored.size(-1)
        freqs = torch.arange(1, ml+1, device=x.device, dtype=x.dtype) / ml - 0.5 - 1.0 / ml
        spec = torch.fft.fftshift(torch.fft.fft(mirrored, dim=-1), dim=-1)
        pos = spec.clone(); pos[:, :ml//2] = 0
        a = alpha.unsqueeze(1).expand(-1, self.K) if alpha.ndim == 1 else alpha
        mh = torch.zeros(B, self.K, ml, device=x.device, dtype=spec.dtype)
        for _ in range(self.iters):
            prev, nm, nw = mh, [], []
            for k in range(self.K):
                lo = torch.stack(nm, dim=1).sum(dim=1) if nm else torch.zeros_like(pos)
                up = prev[:, k+1:].sum(dim=1) if k+1 < self.K else torch.zeros_like(pos)
                m = (pos-lo-up) / (1.0 + a[:, k:k+1]*(freqs.unsqueeze(0)-omega[:, k:k+1]).pow(2))
                nm.append(m)
                ph = m[:, ml//2:]; power = ph.abs().pow(2); ps = power.sum(dim=-1)
                ctr = (freqs[ml//2:].unsqueeze(0)*power).sum(dim=-1)/ps.clamp_min(self.eps)
                nw.append(torch.where(ps > self.eps, ctr, torch.zeros_like(ctr)))
            mh = torch.stack(nm, dim=1); omega = torch.stack(nw, dim=1)
        ph = mh[:, :, ml//2:]; full = torch.zeros_like(mh); full[:, :, ml//2:] = ph
        idx = torch.arange(1, ml//2+1, device=x.device).flip(0)
        full[:, :, idx] = ph.conj(); full[:, :, 0] = full[:, :, -1].conj()
        modes = torch.fft.ifft(torch.fft.ifftshift(full, dim=-1), dim=-1).real[:, :, ml//4:3*ml//4]
        return modes, x - modes.sum(dim=1), omega


class DUVMD(nn.Module):
    def __init__(self, config):
        super().__init__()
        K = int(config["modes"]); self.K = K; self.alpha_base = float(config["alpha_base"])
        self.alpha_delta = nn.Parameter(torch.zeros(K))
        self.vmd = _DeepUnrolledVMD(K, int(config["du_iterations"]))
        self.fc = VMDNetForecaster(K, int(config["lookback"]), int(config["horizon"]),
                                   int(config["d_model"]), tuple(config["tcn_hidden"]),
                                   float(config["embedding_dropout"]), float(config["tcn_dropout"]),
                                   int(config["fusion_hidden"]))
        self.register_buffer("w0", torch.arange(K, dtype=torch.float32) * (0.5 / K))

    def forward(self, x, marks):
        if x.ndim == 2: x = x.unsqueeze(-1)
        out = []
        for c in range(x.size(-1)):
            a = self.alpha_base * torch.exp(torch.tanh(self.alpha_delta))
            a = a.unsqueeze(0).expand(x.size(0), -1).to(x)
            w = self.w0.unsqueeze(0).expand(x.size(0), -1).to(x)
            modes, _, omega = self.vmd(x[:, :, c], a, w)
            out.append(self.fc(modes, marks, omega))
        return torch.stack(out, dim=-1)
