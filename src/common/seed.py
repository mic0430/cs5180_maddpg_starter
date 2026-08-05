from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(
    seed: int,
    deterministic: bool = True,
) -> None:
    """Seed Python, NumPy, and PyTorch.

    Args:
        seed:
            Non-negative integer seed.
        deterministic:
            Request deterministic PyTorch behavior where
            supported. PyTorch is configured with warn-only
            behavior because some operations do not have a
            deterministic implementation on every backend.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    if seed < 0:
        raise ValueError("seed must be non-negative.")

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG",
            ":4096:8",
        )

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        deterministic,
        warn_only=True,
    )

    cudnn_backend = getattr(
        torch.backends,
        "cudnn",
        None,
    )
    if cudnn_backend is not None:
        cudnn_backend.deterministic = deterministic
        cudnn_backend.benchmark = False