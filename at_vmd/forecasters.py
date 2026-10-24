"""Forecaster heads for decomposed VMD modes."""

import math

import torch
import torch.nn as nn


# ===========================================================================
# VMDNet-style TCN forecaster
# ===========================================================================

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        emb = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        scale = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        emb[:, 0::2] = torch.sin(pos * scale)
        emb[:, 1::2] = torch.cos(pos * scale)
        self.register_buffer("pe", emb.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)].to(x)


class TokenEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(1, d_model, 3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class MultiModeEmbedding(nn.Module):
    def __init__(self, n_modes, d_model=64, time_dim=4, dropout=0.1):
        super().__init__()
        self.n_modes = n_modes
        self.tokens = nn.ModuleList([TokenEmbedding(d_model) for _ in range(n_modes)])
        self.freqs = nn.ModuleList([nn.Linear(1, d_model, bias=False) for _ in range(n_modes)])
        self.pos = PositionalEmbedding(d_model)
        self.temp = nn.Linear(time_dim, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, modes, marks, omega):
        t = self.temp(marks)
        out = []
        for i in range(self.n_modes):
            v = self.tokens[i](modes[:, i].unsqueeze(-1))
            f = self.freqs[i](omega[:, i].unsqueeze(-1)).unsqueeze(1)
            out.append(v + self.pos(v) + t + f)
        return self.drop(torch.stack(out, dim=1))


class Chomp1d(nn.Module):
    def __init__(self, size): super().__init__(); self.size = size
    def forward(self, x): return x[:, :, :-self.size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, dropout):
        super().__init__()
        p = 2 * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=p, dilation=dilation)
        self.chomp1 = Chomp1d(p)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=p, dilation=dilation)
        self.chomp2 = Chomp1d(p)
        self.drop1 = nn.Dropout(dropout); self.drop2 = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        o = self.drop1(self.act(self.chomp1(self.conv1(x))))
        o = self.drop2(self.act(self.chomp2(self.conv2(o))))
        r = x if self.down is None else self.down(x)
        return self.norm((o + r).transpose(1, 2)).transpose(1, 2)


class TCNDecoder(nn.Module):
    def __init__(self, d_model, horizon, hidden_dims=(64, 64), dropout=0.2):
        super().__init__()
        layers = []
        for i, ch in enumerate(hidden_dims):
            layers.append(TemporalBlock(d_model if i == 0 else hidden_dims[i-1], ch, 2**i, dropout))
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_dims[-1], 1)
        self.H = horizon

    def forward(self, x):
        e = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.proj(e[:, -self.H:])


class VMDNetForecaster(nn.Module):
    def __init__(self, n_modes=4, lookback=336, horizon=96, d_model=64,
                 hidden_dims=(64, 64), emb_dropout=0.1, tcn_dropout=0.2, fusion_hidden=128):
        super().__init__()
        self.n_modes = n_modes; self.lookback = lookback; self.H = horizon
        self.emb = MultiModeEmbedding(n_modes, d_model, dropout=emb_dropout)
        self.decoders = nn.ModuleList([TCNDecoder(d_model, horizon, hidden_dims, tcn_dropout) for _ in range(n_modes)])
        self.fusion = nn.Sequential(nn.Linear(n_modes * horizon, fusion_hidden), nn.GELU(), nn.Linear(fusion_hidden, horizon))

    def forward(self, modes, marks, omega):
        e = self.emb(modes, marks, omega)
        d = torch.stack([self.decoders[i](e[:, i]) for i in range(self.n_modes)], dim=1)
        return self.fusion(d.reshape(modes.size(0), -1))


# ===========================================================================
# Alternative heads for backbone ablation
# ===========================================================================

class ModeLSTMForecaster(nn.Module):
    def __init__(self, n_modes, lookback, horizon, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.n_modes = n_modes; self.lookback = lookback
        self.lstm = nn.LSTM(n_modes, hidden, layers, dropout=dropout if layers > 1 else 0, batch_first=True)
        self.proj = nn.Linear(hidden, horizon)

    def forward(self, modes, marks, omega):
        _, h = self.lstm(modes.transpose(1, 2))
        return self.proj(h[0][-1])


class ModeSumDLinearForecaster(nn.Module):
    def __init__(self, n_modes, lookback, horizon, kernel=25):
        super().__init__()
        self.n_modes = n_modes; self.lookback = lookback; self.H = horizon
        self.pad = (kernel - 1) // 2
        self.pool = nn.AvgPool1d(kernel, stride=1)
        self.seas = nn.Linear(lookback, horizon); self.trnd = nn.Linear(lookback, horizon)
        w = torch.full((horizon, lookback), 1.0 / lookback)
        self.seas.weight = nn.Parameter(w.clone()); self.trnd.weight = nn.Parameter(w.clone())

    def forward(self, modes, marks, omega):
        f = modes[:, :, :1].expand(-1, -1, self.pad)
        e = modes[:, :, -1:].expand(-1, -1, self.pad)
        p = torch.cat([f, modes, e], dim=-1)
        t = self.pool(p)
        s = modes - t
        return (self.seas(s) + self.trnd(t)).sum(dim=1)


# ===========================================================================
# V2: mode-wise forecasters (omega-aware, fair comparison)
# ===========================================================================

class ModeLSTMForecasterV2(nn.Module):
    """Per-mode LSTM with omega injection + FFN fusion.

    Each VMD mode k gets its own LSTM decoder.
    Omega (centre frequency) is concatenated to the mode value at each timestep.
    Final prediction = FusionMLP(concat(per-mode LSTM outputs)).
    """

    def __init__(self, n_modes, lookback, horizon, hidden=32, layers=1,
                 dropout=0.1, fusion_hidden=128):
        super().__init__()
        self.n_modes = n_modes
        self.lstms = nn.ModuleList([
            nn.LSTM(2, hidden, layers, dropout=dropout if layers > 1 else 0,
                    batch_first=True)
            for _ in range(n_modes)
        ])
        self.projs = nn.ModuleList([
            nn.Linear(hidden, horizon) for _ in range(n_modes)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(n_modes * horizon, fusion_hidden),
            nn.GELU(),
            nn.Linear(fusion_hidden, horizon),
        )

    def forward(self, modes, marks, omega):
        # modes: (B, K, L)   omega: (B, K)
        B, K, L = modes.shape
        omega_exp = omega.unsqueeze(-1).expand(-1, -1, L)  # (B, K, L)
        per_mode = []
        for k in range(K):
            # Concatenate mode value + omega frequency: (B, L, 2)
            inp = torch.stack([modes[:, k], omega_exp[:, k]], dim=-1)
            _, h = self.lstms[k](inp)
            per_mode.append(self.projs[k](h[0][-1]))  # (B, H)
        return self.fusion(torch.cat(per_mode, dim=-1))  # (B, H)


class ModeDLinearForecasterV2(nn.Module):
    """Per-mode DLinear (trend-seasonal decomposition) + sum fusion.

    Each mode is independently decomposed via moving-average;
    per-mode linear projections produce horizon-length forecasts;
    final prediction = sum over modes (no omega needed for DLinear).
    """

    def __init__(self, n_modes, lookback, horizon, kernel=25):
        super().__init__()
        self.n_modes = n_modes
        self.pad = (kernel - 1) // 2
        self.pool = nn.AvgPool1d(kernel, stride=1)
        self.seas = nn.ModuleList([nn.Linear(lookback, horizon) for _ in range(n_modes)])
        self.trnd = nn.ModuleList([nn.Linear(lookback, horizon) for _ in range(n_modes)])
        w = torch.full((horizon, lookback), 1.0 / lookback)
        for i in range(n_modes):
            self.seas[i].weight = nn.Parameter(w.clone())
            self.trnd[i].weight = nn.Parameter(w.clone())

    def forward(self, modes, marks, omega):
        # modes: (B, K, L)
        f = modes[:, :, :1].expand(-1, -1, self.pad)
        e = modes[:, :, -1:].expand(-1, -1, self.pad)
        p = torch.cat([f, modes, e], dim=-1)          # (B, K, L+2*pad)
        t = self.pool(p)                               # (B, K, L)
        s = modes - t
        out = 0.0
        for k in range(self.n_modes):
            out = out + self.seas[k](s[:, k]) + self.trnd[k](t[:, k])
        return out  # (B, H)
