
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.common.seed import seed_everything


def test_reseeding_reproduces_all_random_values() -> None:
    seed_everything(123)

    python_value_1 = random.random()
    numpy_value_1 = np.random.random(5)
    torch_value_1 = torch.rand(5)

    seed_everything(123)

    python_value_2 = random.random()
    numpy_value_2 = np.random.random(5)
    torch_value_2 = torch.rand(5)

    assert python_value_1 == python_value_2
    np.testing.assert_array_equal(
        numpy_value_1,
        numpy_value_2,
    )
    torch.testing.assert_close(
        torch_value_1,
        torch_value_2,
    )


def test_different_seeds_produce_different_values() -> None:
    seed_everything(1)
    first = torch.rand(5)

    seed_everything(2)
    second = torch.rand(5)

    assert not torch.equal(first, second)


def test_negative_seed_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        seed_everything(-1)


@pytest.mark.parametrize(
    "invalid_seed",
    [
        1.5,
        "42",
        True,
    ],
)
def test_non_integer_seed_raises_error(
    invalid_seed: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        seed_everything(invalid_seed)  # type: ignore[arg-type]
