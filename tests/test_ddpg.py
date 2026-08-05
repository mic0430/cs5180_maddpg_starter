import torch
from torch import nn

from src.algorithms.ddpg import DDPGAgent, DDPGBatch


class TinyActor(nn.Module):
    """Small actor used only for unit testing."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_dim, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(observations)


class TinyCritic(nn.Module):
    """Small critic used only for unit testing."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_dim + action_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat(
            (observations, actions),
            dim=-1,
        )

        return self.network(inputs)


def make_agent(
    tau: float = 0.01,
) -> DDPGAgent:
    torch.manual_seed(0)

    actor = TinyActor(
        observation_dim=4,
        action_dim=2,
    )

    critic = TinyCritic(
        observation_dim=4,
        action_dim=2,
    )

    return DDPGAgent(
        actor=actor,
        critic=critic,
        gamma=0.95,
        tau=tau,
        action_low=-1.0,
        action_high=1.0,
    )


def make_batch(
    batch_size: int = 32,
) -> DDPGBatch:
    torch.manual_seed(1)

    return DDPGBatch(
        observations=torch.randn(batch_size, 4),
        actions=torch.empty(
            batch_size,
            2,
        ).uniform_(-1.0, 1.0),
        rewards=torch.randn(batch_size, 1),
        next_observations=torch.randn(batch_size, 4),
        dones=torch.zeros(batch_size, 1),
    )


def parameters_match(
    first: nn.Module,
    second: nn.Module,
) -> bool:
    return all(
        torch.allclose(
            first_parameter,
            second_parameter,
        )
        for first_parameter, second_parameter in zip(
            first.parameters(),
            second.parameters(),
            strict=True,
        )
    )


def parameters_changed(
    before: list[torch.Tensor],
    module: nn.Module,
) -> bool:
    return any(
        not torch.allclose(old, new)
        for old, new in zip(
            before,
            module.parameters(),
            strict=True,
        )
    )


def test_target_networks_start_as_copies() -> None:
    agent = make_agent()

    assert agent.target_actor is not agent.actor
    assert agent.target_critic is not agent.critic

    assert parameters_match(
        agent.actor,
        agent.target_actor,
    )

    assert parameters_match(
        agent.critic,
        agent.target_critic,
    )


def test_act_returns_bounded_action_with_correct_shape() -> None:
    agent = make_agent()
    observation = torch.randn(4)

    # Large noise ensures the clamp is actually tested.
    action = agent.act(
        observation,
        noise_std=10.0,
    )

    assert action.shape == (2,)
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)
    assert not action.requires_grad


def test_terminal_transition_does_not_bootstrap() -> None:
    agent = make_agent()

    rewards = torch.tensor([
        [2.0],
        [-3.0],
    ])

    batch = DDPGBatch(
        observations=torch.randn(2, 4),
        actions=torch.randn(2, 2).clamp(-1.0, 1.0),
        rewards=rewards,
        next_observations=torch.randn(2, 4),

        # Both transitions are terminal.
        dones=torch.ones(2, 1),
    )

    targets = agent.compute_critic_targets(batch)

    # When done is 1, target should equal reward.
    assert torch.allclose(targets, rewards)
    assert not targets.requires_grad


def test_update_returns_finite_metrics_and_changes_networks() -> None:
    agent = make_agent()
    batch = make_batch()

    actor_before = [
        parameter.detach().clone()
        for parameter in agent.actor.parameters()
    ]

    critic_before = [
        parameter.detach().clone()
        for parameter in agent.critic.parameters()
    ]

    metrics = agent.update(batch)

    assert set(metrics) == {
        "actor_loss",
        "critic_loss",
        "mean_q",
        "mean_target_q",
    }

    assert all(
        torch.isfinite(torch.tensor(value))
        for value in metrics.values()
    )

    assert parameters_changed(
        actor_before,
        agent.actor,
    )

    assert parameters_changed(
        critic_before,
        agent.critic,
    )


def test_soft_update_moves_target_toward_online_network() -> None:
    agent = make_agent(tau=0.25)

    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.fill_(1.0)

        for parameter in agent.target_actor.parameters():
            parameter.zero_()

    agent.update_target_networks()

    # target = 0.75(0) + 0.25(1) = 0.25
    for parameter in agent.target_actor.parameters():
        expected = torch.full_like(
            parameter,
            0.25,
        )

        assert torch.allclose(
            parameter,
            expected,
        )