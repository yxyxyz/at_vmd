"""Data loading — 60/20/20 chronological split, train-only normalisation."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DatasetSpec:
    filename: str
    target: str
    clock: str
    frequency: str = "1h"


DATASETS = {
    "entsoe_be":    DatasetSpec("entsoe_belgium_hourly.csv", "electricity_demand_belgium", "UTC"),
    "entsoe_de_lu": DatasetSpec("entsoe_germany_luxembourg_hourly.csv", "electricity_demand_germany_luxembourg", "UTC"),
    "entsoe":       DatasetSpec("entsoe_poland_hourly.csv", "electricity_demand_poland", "UTC"),
    "aemo":         DatasetSpec("aemo_nsw_hourly.csv", "electricity_demand_aemo_nsw", "AEST"),
    "pjm":          DatasetSpec("pjm_hourly.csv", "electricity_demand_pjm", "UTC"),
}


def sha256_file(path):
    d = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            d.update(b)
    return d.hexdigest()


class WindowDataset(Dataset):
    def __init__(self, series, marks, first, last, L, H):
        self.series = series; self.marks = marks
        self.starts = torch.arange(first, last + 1)
        self.L = L; self.H = H

    def __len__(self): return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i]); t = s + self.L
        return self.series[s:t], self.marks[s:t], self.series[t:t + self.H]


class PowerDataModule:
    def __init__(self, dataset, data_dir, lookback=336, horizon=96, freq=None, column=None):
        self.dataset = dataset
        self.spec = DATASETS[dataset]
        self.data_dir = Path(data_dir)
        self.L = lookback; self.H = horizon
        self.freq = freq or self.spec.frequency
        self.column = column
        self._load()

    def _load(self):
        path = self.data_dir / self.spec.filename
        frame = pd.read_csv(path)
        target_cols = [self.column if self.column else self.spec.target]
        dates = pd.to_datetime(frame["date"], format="%Y-%m-%d %H:%M:%S")
        self.dates = dates.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
        raw = frame[target_cols].to_numpy(np.float32)
        if raw.ndim == 1: raw = raw[:, None]

        N = len(raw)
        self.train_end = int(N * 0.6)
        self.val_end = int(N * 0.8)
        train_mean = raw[:self.train_end].mean(axis=0, dtype=np.float64)
        train_std = raw[:self.train_end].std(axis=0, dtype=np.float64, ddof=0)
        self.train_mean = float(train_mean[0])
        self.train_std = float(train_std[0])

        norm = ((raw.astype(np.float64) - train_mean) / train_std).astype(np.float32)
        marks = np.column_stack([
            dates.dt.hour.to_numpy(np.float32) / 23.0 - 0.5,
            dates.dt.dayofweek.to_numpy(np.float32) / 6.0 - 0.5,
            (dates.dt.day.to_numpy(np.float32) - 1.0) / 30.0 - 0.5,
            (dates.dt.dayofyear.to_numpy(np.float32) - 1.0) / 365.0 - 0.5,
        ]).astype(np.float32)

        self.series = torch.from_numpy(norm)
        self.marks = torch.from_numpy(marks)
        self.rows = N
        self.sha256 = sha256_file(path)

    def split(self, name):
        L = len(self.series)
        r = {"train": (0, self.train_end - self.L - self.H),
             "val": (self.train_end - self.L, self.val_end - self.L - self.H),
             "test": (self.val_end - self.L, L - self.L - self.H)}
        return WindowDataset(self.series, self.marks, *r[name], self.L, self.H)

    def loader(self, name, batch_size, seed, num_workers=0):
        shuffle = name == "train"
        gen = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(self.split(name), batch_size=batch_size, shuffle=shuffle,
                          drop_last=False, num_workers=num_workers, generator=gen)

    def metadata(self):
        return {"dataset": self.dataset, "rows": self.rows, "sha256": self.sha256,
                "train_end": self.train_end, "val_end": self.val_end}
