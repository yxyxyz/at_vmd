"""AT-VMD: omega-only amortized VMD + pluggable forecaster head."""

import torch
import torch.nn as nn

from .vmd import ConvergentVMD
from .encoder import AmortizedVMDEncoder, GlobalOmegaEncoder
from .forecasters import (VMDNetForecaster, ModeLSTMForecaster,
                           ModeSumDLinearForecaster,
                           ModeLSTMForecasterV2, ModeDLinearForecasterV2)


class ATVMD(nn.Module):
    """Learns per‑window centre frequencies ω(x) from forecasting loss.

    Bandwidth α is fixed at ``alpha_base`` (default 2000) unless
    ``adaptive_alpha`` is enabled.  Ablation switches are passed via the
    config dict, not constructor args.
    """

    def __init__(self, modes=4, lookback=336, horizon=96, alpha_base=2000.0,
                 alpha_log_range=3.0,
                 encoder_hidden=32, d_model=64, hidden_dims=(64, 64),
                 emb_dropout=0.1, tcn_dropout=0.2, fusion_hidden=128,
                 vmd_tau=0.0, forecaster_cls=None, forecaster_kwargs=None,
                 adaptive_omega=True, adaptive_alpha=False,
                 control_type="per_sample",
                 task_gradient=True, vmd_steps=1):
        super().__init__()
        self.lookback = lookback; self.modes = modes
        self.alpha_base = alpha_base
        self.alpha_log_range = alpha_log_range
        self.adaptive_omega = adaptive_omega
        self.adaptive_alpha = adaptive_alpha
        self.control_type = control_type
        self.task_gradient = task_gradient
        self.vmd_steps = vmd_steps
        if control_type not in {"per_sample", "global"}:
            raise ValueError(f"Unknown control_type {control_type!r}")
        if vmd_steps < 1:
            raise ValueError("vmd_steps must be at least 1")

        if forecaster_cls is None:
            forecaster_cls = VMDNetForecaster
        if forecaster_kwargs is None:
            forecaster_kwargs = dict(n_modes=modes, lookback=lookback, horizon=horizon,
                                     d_model=d_model, hidden_dims=hidden_dims,
                                     emb_dropout=emb_dropout, tcn_dropout=tcn_dropout,
                                     fusion_hidden=fusion_hidden)
        self.forecaster = forecaster_cls(**forecaster_kwargs)
        if control_type == "global":
            self.encoder = GlobalOmegaEncoder(
                modes=modes, alpha_base=alpha_base,
                alpha_log_range=self.alpha_log_range,
                predict_alpha=adaptive_alpha)
        else:
            self.encoder = AmortizedVMDEncoder(
                modes=modes, alpha_base=alpha_base,
                alpha_log_range=self.alpha_log_range,
                hidden_dim=encoder_hidden,
                predict_alpha=adaptive_alpha)
        self.vmd = ConvergentVMD(modes=modes, tau=vmd_tau)

    def _encode(self, x_c):
        B = x_c.size(0)
        ctrl = self.encoder(x_c)
        if not self.adaptive_omega:
            ctrl["omega"] = (torch.arange(self.modes, device=x_c.device, dtype=x_c.dtype)
                             * (0.5 / self.modes)).unsqueeze(0).expand(B, -1)
        if not self.adaptive_alpha:
            ctrl["alpha"] = torch.full((B, self.modes), self.alpha_base,
                                       device=x_c.device, dtype=x_c.dtype)
        return ctrl

    def forward(self, x, marks):
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        C = x.size(-1)
        preds, resids, alphas, omega_controls, omegas = [], [], [], [], []
        for c in range(C):
            ctrl = self._encode(x[:, :, c])
            modes, residual, omega = self.vmd.differentiable_decompose(
                x[:, :, c], ctrl["alpha"], ctrl["omega"], self.vmd_steps)
            forecast_modes = modes if self.task_gradient else modes.detach()
            forecast_omega = omega if self.task_gradient else omega.detach()
            preds.append(self.forecaster(forecast_modes, marks, forecast_omega))
            resids.append(residual)
            alphas.append(ctrl["alpha"])
            omega_controls.append(ctrl["omega"])
            omegas.append(omega)
        return {"prediction": torch.stack(preds, dim=-1),
                "residual": torch.stack(resids, dim=-1),
                "alpha": torch.stack(alphas, dim=-1),
                "omega_control": torch.stack(omega_controls, dim=-1),
                "omega": torch.stack(omegas, dim=-1)}


def build_atvmd(config):
    modes, L, H = int(config["modes"]), int(config["lookback"]), int(config["horizon"])
    fc = config.get("forecaster", "tcn")
    if fc == "tcn":
        fc_cls = VMDNetForecaster
        fc_kw = dict(n_modes=modes, lookback=L, horizon=H, d_model=int(config["d_model"]),
                     hidden_dims=tuple(config["tcn_hidden"]),
                     emb_dropout=float(config["embedding_dropout"]),
                     tcn_dropout=float(config["tcn_dropout"]),
                     fusion_hidden=int(config["fusion_hidden"]))
    elif fc == "lstm":
        fc_cls = ModeLSTMForecasterV2
        fc_kw = dict(n_modes=modes, lookback=L, horizon=H,
                     hidden=int(config.get("lstm_hidden", 32)),
                     layers=int(config.get("lstm_layers", 1)),
                     dropout=float(config.get("lstm_dropout", 0.1)),
                     fusion_hidden=int(config.get("fusion_hidden", 128)))
    elif fc == "dlinear":
        fc_cls = ModeDLinearForecasterV2
        fc_kw = dict(n_modes=modes, lookback=L, horizon=H,
                     kernel=int(config.get("dlinear_kernel", 25)))
    else:
        raise ValueError(f"Unknown forecaster {fc!r}")
    return ATVMD(
        modes=modes, lookback=L, horizon=H,
        alpha_base=float(config["alpha_base"]),
        alpha_log_range=float(config.get("alpha_log_range", 3.0)),
        encoder_hidden=int(config["encoder_hidden"]),
        d_model=int(config["d_model"]),
        hidden_dims=tuple(config["tcn_hidden"]),
        emb_dropout=float(config["embedding_dropout"]),
        tcn_dropout=float(config["tcn_dropout"]),
        fusion_hidden=int(config["fusion_hidden"]),
        vmd_tau=float(config["vmd_tau"]),
        forecaster_cls=fc_cls, forecaster_kwargs=fc_kw,
        adaptive_omega=bool(config.get("adaptive_omega", True)),
        adaptive_alpha=bool(config.get("adaptive_alpha", False)),
        control_type=str(config.get("control_type", "per_sample")),
        task_gradient=bool(config.get("task_gradient", True)),
        vmd_steps=int(config.get("vmd_steps", 1)),
    )
