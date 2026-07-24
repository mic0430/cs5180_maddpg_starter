from __future__ import annotations

from mpe2 import simple_spread_v3


def main() -> None:
    env = simple_spread_v3.parallel_env(
        N=3,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=True,
        render_mode=None,
    )

    observations, infos = env.reset(seed=42)
    print("Agents:", env.agents)

    for agent in env.agents:
        print(
            f"{agent}: observation shape={observations[agent].shape}, "
            f"action space={env.action_space(agent)}"
        )

    actions = {
        agent: env.action_space(agent).sample()
        for agent in env.agents
    }

    next_observations, rewards, terminations, truncations, infos = env.step(actions)

    print("Rewards:", rewards)
    print("Terminations:", terminations)
    print("Truncations:", truncations)

    assert set(actions) == set(rewards)
    assert set(next_observations).issubset(set(actions))

    env.close()
    print("MPE2 Parallel API smoke test: PASS")


if __name__ == "__main__":
    main()
