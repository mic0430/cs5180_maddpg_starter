
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from src.common.target_updates import hard_update, soft_update


def make_network() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )


def test_hard_update_copies_parameters_exactly() -> None:
    source = make_network()
    target = make_network()

    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(2.0)

        for parameter in target.parameters():
            parameter.zero_()

    hard_update(
        target=target,
        source=source,
    )

    for target_parameter, source_parameter in zip(
        target.parameters(),
        source.parameters(),
    ):
        torch.testing.assert_close(
            target_parameter,
            source_parameter,
        )

        assert (
            target_parameter.data_ptr()
            != source_parameter.data_ptr()
        )


def test_soft_update_interpolates_parameters() -> None:
    source = make_network()
    target = copy.deepcopy(source)

    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(1.0)

        for parameter in target.parameters():
            parameter.zero_()

    soft_update(
        target=target,
        source=source,
        tau=0.25,
    )

    for parameter in target.parameters():
        expected = torch.full_like(
            parameter,
            0.25,
        )
        torch.testing.assert_close(
            parameter,
            expected,
        )


@pytest.mark.parametrize(
    "tau",
    [
        -0.01,
        1.01,
    ],
)
def test_invalid_tau_raises_error(
    tau: float,
) -> None:
    source = make_network()
    target = make_network()

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        soft_update(
            target=target,
            source=source,
            tau=tau,
        )


def test_mismatched_networks_raise_error() -> None:
    source = nn.Linear(3, 2)
    target = nn.Linear(4, 2)

    with pytest.raises(
        ValueError,
        match="different shapes",
    ):
        hard_update(
            target=target,
            source=source,
        )
