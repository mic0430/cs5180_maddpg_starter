
from __future__ import annotations

import torch
from torch import nn


def _validate_matching_structure(
    target: nn.Module,
    source: nn.Module,
) -> None:
    """Verify that two modules have matching state structures."""
    target_state = target.state_dict()
    source_state = source.state_dict()

    if target_state.keys() != source_state.keys():
        raise ValueError(
            "Target and source modules have different "
            "state-dictionary keys."
        )

    for name in target_state:
        if (
            target_state[name].shape
            != source_state[name].shape
        ):
            raise ValueError(
                f"State tensor '{name}' has different shapes: "
                f"target={tuple(target_state[name].shape)}, "
                f"source={tuple(source_state[name].shape)}."
            )


def hard_update(
    target: nn.Module,
    source: nn.Module,
) -> None:
    """Copy all source parameters and buffers into target."""
    _validate_matching_structure(
        target=target,
        source=source,
    )
    target.load_state_dict(source.state_dict())


def soft_update(
    target: nn.Module,
    source: nn.Module,
    tau: float,
) -> None:
    """Move target parameters toward source parameters.

    The update is:

        target = tau * source + (1 - tau) * target
    """
    if not isinstance(tau, (int, float)):
        raise TypeError("tau must be a numeric value.")

    tau = float(tau)

    if not 0.0 <= tau <= 1.0:
        raise ValueError(
            "tau must be between 0 and 1 inclusive."
        )

    _validate_matching_structure(
        target=target,
        source=source,
    )

    source_parameters = dict(
        source.named_parameters()
    )
    source_buffers = dict(
        source.named_buffers()
    )

    with torch.no_grad():
        for name, target_parameter in (
            target.named_parameters()
        ):
            source_parameter = source_parameters[name]
            source_value = source_parameter.detach().to(
                device=target_parameter.device,
                dtype=target_parameter.dtype,
            )

            target_parameter.mul_(1.0 - tau)
            target_parameter.add_(
                source_value,
                alpha=tau,
            )

        # Non-floating state, such as BatchNorm counters,
        # should be copied rather than interpolated.
        for name, target_buffer in target.named_buffers():
            source_buffer = source_buffers[name]
            target_buffer.copy_(
                source_buffer.detach().to(
                    device=target_buffer.device,
                    dtype=target_buffer.dtype,
                )
            )
