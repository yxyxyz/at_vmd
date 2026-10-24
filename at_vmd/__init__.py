"""AT-VMD: Amortized Variational Mode Decomposition — minimal academic release."""

from .model import ATVMD, build_atvmd
from .data import DATASETS, PowerDataModule
from .engine import PAPER_SEEDS, train
from .objectives import ReconstructionConstraint, relative_reconstruction_error
from .baselines import BENCHMARKS, build_benchmark

__all__ = [
    "ATVMD", "BENCHMARKS", "DATASETS", "PAPER_SEEDS", "PowerDataModule",
    "ReconstructionConstraint", "build_atvmd", "build_benchmark",
    "relative_reconstruction_error", "train",
]
