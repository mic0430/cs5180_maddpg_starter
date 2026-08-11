
from __future__ import annotations

import torch


_ALLOWED_DEVICE_NAMES = {"auto", "cpu", "cuda", "mps"}


def _mps_available() -> bool:
    """Return whether PyTorch's Apple MPS backend is available."""
    backend = getattr(torch.backends, "mps", None)
    return bool(
        backend is not None
        and backend.is_available()
    )


def resolve_device(
    requested: str | None = "auto",
) -> torch.device:
    """Resolve a requested PyTorch device.

    Automatic selection order:
        1. CUDA
        2. Apple MPS
        3. CPU

    Args:
        requested:
            One of ``auto``, ``cuda``, ``mps``, or ``cpu``.
            ``None`` is treated as ``auto``.

    Returns:
        A resolved ``torch.device``.

    Raises:
        TypeError:
            If requested is not a string or None.
        ValueError:
            If the requested device name is unsupported.
        RuntimeError:
            If CUDA or MPS is requested but unavailable.
    """
    if requested is None:
        device_name = "auto"
    elif isinstance(requested, str):
        device_name = requested.strip().lower()
    else:
        raise TypeError(
            "requested device must be a string or None."
        )

    if device_name not in _ALLOWED_DEVICE_NAMES:
        allowed = ", ".join(
            sorted(_ALLOWED_DEVICE_NAMES)
        )
        raise ValueError(
            f"Unsupported device '{device_name}'. "
            f"Expected one of: {allowed}."
        )

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if _mps_available():
            return torch.device("mps")

        return torch.device("cpu")

    if (
        device_name == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested, but CUDA is not available "
            "in this PyTorch environment."
        )

    if (
        device_name == "mps"
        and not _mps_available()
    ):
        raise RuntimeError(
            "MPS was requested, but Apple MPS is not available "
            "in this PyTorch environment."
        )

    return torch.device(device_name)
