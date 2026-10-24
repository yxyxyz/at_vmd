"""Benchmark model registry."""

from .vmdnet import VMDNet
from .du_vmd import DUVMD
from .dlinear import DLinear
from .tcn import TCN
from .lstm import LSTM
from .itransformer import ITransformer
from .patchtst import PatchTST

BENCHMARKS = {
    "vmdnet": VMDNet,
    "du-vmd": DUVMD,
    "dlinear": DLinear,
    "tcn": TCN,
    "lstm": LSTM,
    "itransformer": ITransformer,
    "patchtst": PatchTST,
}

def build_benchmark(name, config):
    return BENCHMARKS[name](config)
