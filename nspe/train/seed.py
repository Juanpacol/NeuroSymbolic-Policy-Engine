"""Seeding helper for reproducible training runs.

With 831 validation cases, differences of ~0.01 in accuracy are inside
seed noise, so every reported number needs several seeds and a stated
spread. That is only meaningful if a seed actually pins the run down.
"""

from __future__ import annotations

import os
import random

import torch


def set_seed(seed: int, deterministic: bool = False) -> torch.Generator:
    """Seeds Python, NumPy, and torch, and returns a seeded generator.

    Args:
        seed: the seed to apply.
        deterministic: if ``True``, also request deterministic cuDNN
            kernels. This costs throughput and is off by default; turn
            it on when chasing a reproducibility discrepancy rather than
            for routine runs.

    Returns:
        A ``torch.Generator`` seeded with ``seed``, suitable for passing
        to a ``DataLoader`` so shuffling is reproducible too.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
