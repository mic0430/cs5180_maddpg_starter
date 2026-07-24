def test_core_imports() -> None:
    import gymnasium  # noqa: F401
    import mpe2  # noqa: F401
    import numpy  # noqa: F401
    import pettingzoo  # noqa: F401
    import torch  # noqa: F401


def test_simple_spread_reset() -> None:
    from mpe2 import simple_spread_v3

    env = simple_spread_v3.parallel_env(
        N=3,
        max_cycles=5,
        continuous_actions=True,
    )
    observations, infos = env.reset(seed=0)

    assert len(env.agents) == 3
    assert set(observations) == set(env.agents)

    env.close()
