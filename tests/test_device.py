
from __future__ import annotations

import pytest
import torch

import src.common.device as device_module
from src.common.device import resolve_device


def test_explicit_cpu_is_always_available() -> None:
    device = resolve_device("cpu")
    assert device == torch.device("cpu")


def test_auto_prefers_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        device_module,
        "_mps_available",
        lambda: True,
    )

    assert resolve_device("auto").type == "cuda"


def test_auto_uses_mps_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        device_module,
        "_mps_available",
        lambda: True,
    )

    assert resolve_device("auto").type == "mps"


def test_auto_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        device_module,
        "_mps_available",
        lambda: False,
    )

    assert resolve_device("auto").type == "cpu"


@pytest.mark.parametrize(
    "requested",
    [
        "cuda",
        "mps",
    ],
)
def test_unavailable_requested_device_raises_error(
    requested: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        device_module,
        "_mps_available",
        lambda: False,
    )

    with pytest.raises(
        RuntimeError,
        match=requested.upper(),
    ):
        resolve_device(requested)


def test_invalid_device_name_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported device",
    ):
        resolve_device("quantum-gpu")