from __future__ import annotations

import random
from typing import Optional

import numpy as np

try:  # Torch might not be available for light-weight commands (e.g. load, clean).
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


def resolve_device(requested: Optional[str]) -> str:
    """
    Return "cuda" only when it was requested and is available; otherwise "cpu".
    """
    if (requested or "cpu").lower() != "cuda":
        return "cpu"
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    print("[runtime] CUDA requested but not available - falling back to CPU.")
    return "cpu"


def configure_runtime(seed: int, device: Optional[str] = None) -> None:
    """
    Seed the Python, NumPy and PyTorch random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
