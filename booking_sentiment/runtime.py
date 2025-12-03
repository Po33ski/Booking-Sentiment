from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:  # Torch might not be available for light-weight commands (e.g. load, clean).
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


def configure_runtime(seed: int, device: Optional[str] = None) -> None:
    """
    Apply deterministic settings (hash seed + PRNG seeds) once per process.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    normalized_device = (device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
    if normalized_device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

