"""PatchTST wrapper — thin adapter over vendored TSL code."""

from types import SimpleNamespace
import torch
import torch.nn as nn

from .third_party.tsl.models.PatchTST import Model as _PatchTST


class PatchTST(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        enc_in = int(config.get("enc_in", 1))
        args = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=int(config["lookback"]), label_len=0,
            pred_len=int(config["horizon"]),
            enc_in=enc_in, c_out=enc_in,
            d_model=int(config.get("baseline_d_model", 64)),
            d_ff=int(config.get("baseline_d_ff", 2048)),
            n_heads=int(config.get("baseline_heads", 8)),
            e_layers=int(config.get("baseline_layers", 2)),
            factor=1,
            dropout=float(config.get("baseline_dropout", 0.1)),
            activation="gelu", embed="timeF", freq="h",
            top_k=int(config.get("timesnet_top_k", 5)),
            num_kernels=int(config.get("timesnet_num_kernels", 6)),
            num_class=1,
        )
        self.model = _PatchTST(
            args,
            patch_len=int(config.get("patch_len", 16)),
            stride=int(config.get("patch_stride", 8)),
        )

    def forward(self, x: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        return self.model(x, marks, None, None)  # [B, H, C]
