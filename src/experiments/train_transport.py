from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.algorithms.independent_ddpg import IndependentDDPG
from src.algorithms.maddpg import MADDPG
from src.common.replay_buffer import MultiAgentReplayBuffer
from src.common.seed import seed_everything
from src.envs.transport_env import CooperativeTransportEnv


Learner = MADDPG | IndependentDDPG


@dataclass
class TrainingConfig:
    """Configuration for cooperative-transport training."""

    algorithm: str = "maddpg"
    seed: int = 42
    device: str = "auto"

    environment_config: Path | str = Path(
        "configs/cooperative_transport_narrow_passage.yaml"
    )

    episodes: int = 100
    batch_size: int = 128
    replay_capacity: int = 100_000
    learning_starts: int = 1_024
    update_every: int = 1

    gamma: float = 0.95
    tau: float = 0.01
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    hidden_sizes: tuple[int, ...] = (64, 64)

    exploration_noise_std: float = 0.20
    exploration_noise_final: float = 0.05
    exploration_decay_episodes: int = 0

    evaluation_interval: int = 0
    evaluation_episodes: int = 0

    log_dir: Path | str = Path("runs/transport")
    checkpoint_path: Path | str | None = None

    def __post_init__(self) -> None:
        self.algorithm = self.algorithm.strip().lower()
        self.environment_config = Path(
            self.environment_config
        )
        self.log_dir = Path(self.log_dir)

        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(
                self.checkpoint_path
            )

        if self.algorithm not in {
            "maddpg",
            "independent_ddpg",
        }:
            raise ValueError(
                "algorithm must be 'maddpg' or "
                "'independent_ddpg'."
            )

        for name in (
            "episodes",
            "batch_size",
            "replay_capacity",
            "update_every",
        ):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be at least 1."
                )

        if self.learning_starts < 0:
            raise ValueError(
                "learning_starts must be non-negative."
            )

        for name in (
            "exploration_decay_episodes",
            "evaluation_interval",
            "evaluation_episodes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be non-negative."
                )

        if self.replay_capacity < self.batch_size:
            raise ValueError(
                "replay_capacity must be at least batch_size."
            )

        if self.exploration_noise_std < 0.0:
            raise ValueError(
                "exploration_noise_std must be non-negative."
            )

        if self.exploration_noise_final < 0.0:
            raise ValueError(
                "exploration_noise_final must be non-negative."
            )

        if (
            self.exploration_noise_final
            > self.exploration_noise_std
        ):
            raise ValueError(
                "exploration_noise_final must not exceed "
                "exploration_noise_std."
            )

        partial_evaluation = (
            self.evaluation_interval == 0
        ) != (
            self.evaluation_episodes == 0
        )

        if partial_evaluation:
            raise ValueError(
                "evaluation_interval and evaluation_episodes "
                "must either both be zero or both be positive."
            )

        self.hidden_sizes = tuple(
            int(size)
            for size in self.hidden_sizes
        )

        if not self.hidden_sizes:
            raise ValueError(
                "hidden_sizes must not be empty."
            )

        if any(size < 1 for size in self.hidden_sizes):
            raise ValueError(
                "All hidden sizes must be positive."
            )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> TrainingConfig:
        config_path = Path(path)

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data: dict[str, Any] = (
                yaml.safe_load(file) or {}
            )

        project = data.get("project", {})
        environment = data.get("environment", {})
        training = data.get("training", {})
        logging = data.get("logging", {})

        initial_noise = training.get(
            "exploration_noise_std",
            0.20,
        )

        return cls(
            algorithm=data.get(
                "algorithm",
                "maddpg",
            ),
            seed=project.get("seed", 42),
            device=project.get("device", "auto"),
            environment_config=environment.get(
                "config",
                "configs/"
                "cooperative_transport_narrow_passage.yaml",
            ),
            episodes=training.get(
                "episodes",
                100,
            ),
            batch_size=training.get(
                "batch_size",
                128,
            ),
            replay_capacity=training.get(
                "replay_capacity",
                100_000,
            ),
            learning_starts=training.get(
                "learning_starts",
                1_024,
            ),
            update_every=training.get(
                "update_every",
                1,
            ),
            gamma=training.get(
                "gamma",
                0.95,
            ),
            tau=training.get(
                "tau",
                0.01,
            ),
            actor_lr=training.get(
                "actor_lr",
                1e-3,
            ),
            critic_lr=training.get(
                "critic_lr",
                1e-3,
            ),
            hidden_sizes=tuple(
                training.get(
                    "hidden_sizes",
                    [64, 64],
                )
            ),
            exploration_noise_std=initial_noise,
            exploration_noise_final=training.get(
                "exploration_noise_final",
                initial_noise,
            ),
            exploration_decay_episodes=training.get(
                "exploration_decay_episodes",
                0,
            ),
            evaluation_interval=training.get(
                "evaluation_interval",
                0,
            ),
            evaluation_episodes=training.get(
                "evaluation_episodes",
                0,
            ),
            log_dir=logging.get(
                "log_dir",
                "runs/transport",
            ),
            checkpoint_path=logging.get(
                "checkpoint_path",
            ),
        )


@dataclass(frozen=True)
class EvaluationResult:
    mean_return: float
    success_rate: float
    mean_steps: float
    mean_final_distance: float
    mean_collisions: float
    mean_force_disagreement: float
    mean_control_effort: float
    mean_max_abs_orientation: float


@dataclass(frozen=True)
class TrainingResult:
    algorithm: str
    episodes_completed: int
    environment_steps: int
    update_count: int
    mean_episode_return: float
    training_seconds: float
    final_evaluation: EvaluationResult | None
    training_episode_returns: tuple[float, ...]
    evaluation_episode_indices: tuple[int, ...]
    evaluation_results: tuple[EvaluationResult, ...]
    checkpoint_path: Path | None


def load_environment_config(
    path: str | Path,
) -> dict[str, Any]:
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file) or {}

    return config


def reward_vector(
    rewards: Any,
    num_agents: int,
) -> np.ndarray:
    array = np.asarray(
        rewards,
        dtype=np.float32,
    )

    if array.ndim == 0:
        return np.full(
            num_agents,
            float(array),
            dtype=np.float32,
        )

    if array.shape != (num_agents,):
        raise ValueError(
            f"Expected rewards shape ({num_agents},), "
            f"got {array.shape}."
        )

    return array


def exploration_noise_for_episode(
    config: TrainingConfig,
    episode: int,
) -> float:
    if episode < 0:
        raise ValueError(
            "episode must be non-negative."
        )

    if config.exploration_decay_episodes == 0:
        return config.exploration_noise_std

    progress = min(
        episode / config.exploration_decay_episodes,
        1.0,
    )

    return float(
        config.exploration_noise_std
        + progress
        * (
            config.exploration_noise_final
            - config.exploration_noise_std
        )
    )


def build_learner(
    config: TrainingConfig,
    env: CooperativeTransportEnv,
) -> Learner:
    common_arguments = {
        "num_agents": env.num_agents,
        "observation_dim": env.observation_dim,
        "action_dim": env.action_dim,
        "hidden_sizes": config.hidden_sizes,
        "actor_lr": config.actor_lr,
        "critic_lr": config.critic_lr,
        "gamma": config.gamma,
        "tau": config.tau,
        "exploration_noise_std": (
            config.exploration_noise_std
        ),
        "device": config.device,
    }

    if config.algorithm == "maddpg":
        return MADDPG(**common_arguments)

    if config.algorithm == "independent_ddpg":
        return IndependentDDPG(**common_arguments)

    raise ValueError(
        f"Unsupported algorithm: {config.algorithm}"
    )


def _info_float(
    info: dict[str, Any],
    key: str,
) -> float:
    value = info.get(key, np.nan)

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def evaluate_transport(
    learner: Learner,
    config: TrainingConfig,
) -> EvaluationResult:
    if config.evaluation_episodes < 1:
        raise ValueError(
            "evaluation_episodes must be positive "
            "when evaluation is performed."
        )

    env_config = load_environment_config(
        config.environment_config
    )

    env = CooperativeTransportEnv(env_config)

    returns: list[float] = []
    successes: list[float] = []
    steps_list: list[float] = []
    final_distances: list[float] = []
    collisions: list[float] = []
    force_disagreements: list[float] = []
    control_efforts: list[float] = []
    max_abs_orientations: list[float] = []

    evaluation_seed_base = config.seed + 100_000

    for evaluation_episode in range(
        config.evaluation_episodes
    ):
        observations, info = env.reset(
            seed=(
                evaluation_seed_base
                + evaluation_episode
            )
        )

        episode_return = 0.0
        steps = 0
        terminated = False
        truncated = False

        max_abs_orientation = 0.0

        initial_orientation = _info_float(
            info,
            "payload_orientation",
        )

        if np.isfinite(initial_orientation):
            max_abs_orientation = abs(
                initial_orientation
            )

        while not (terminated or truncated):
            actions = learner.select_actions(
                observations,
                explore=False,
            )

            (
                observations,
                rewards,
                terminated,
                truncated,
                info,
            ) = env.step(actions)

            rewards = reward_vector(
                rewards,
                env.num_agents,
            )

            episode_return += float(
                np.mean(rewards)
            )
            steps += 1

            orientation = _info_float(
                info,
                "payload_orientation",
            )

            if np.isfinite(orientation):
                max_abs_orientation = max(
                    max_abs_orientation,
                    abs(orientation),
                )

        returns.append(episode_return)
        successes.append(
            float(bool(info.get("success", False)))
        )
        steps_list.append(float(steps))

        final_distances.append(
            _info_float(
                info,
                "payload_distance_to_goal",
            )
        )
        collisions.append(
            _info_float(
                info,
                "collision_count",
            )
        )
        force_disagreements.append(
            _info_float(
                info,
                "mean_force_disagreement",
            )
        )
        control_efforts.append(
            _info_float(
                info,
                "total_control_effort",
            )
        )
        max_abs_orientations.append(
            max_abs_orientation
        )

    def safe_mean(values: list[float]) -> float:
        array = np.asarray(
            values,
            dtype=np.float64,
        )

        if np.all(np.isnan(array)):
            return float("nan")

        return float(np.nanmean(array))

    return EvaluationResult(
        mean_return=safe_mean(returns),
        success_rate=safe_mean(successes),
        mean_steps=safe_mean(steps_list),
        mean_final_distance=safe_mean(
            final_distances
        ),
        mean_collisions=safe_mean(
            collisions
        ),
        mean_force_disagreement=safe_mean(
            force_disagreements
        ),
        mean_control_effort=safe_mean(
            control_efforts
        ),
        mean_max_abs_orientation=safe_mean(
            max_abs_orientations
        ),
    )


def train_transport(
    config: TrainingConfig,
) -> TrainingResult:
    seed_everything(config.seed)

    rng = np.random.default_rng(
        config.seed
    )

    env_config = load_environment_config(
        config.environment_config
    )

    env = CooperativeTransportEnv(
        env_config
    )

    learner = build_learner(
        config,
        env,
    )

    replay_buffer = MultiAgentReplayBuffer(
        capacity=config.replay_capacity,
        num_agents=env.num_agents,
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        seed=config.seed,
    )

    config.log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = SummaryWriter(
        log_dir=str(config.log_dir)
    )

    global_step = 0
    update_count = 0

    episode_returns: list[float] = []
    evaluation_episode_indices: list[int] = []
    evaluation_results: list[EvaluationResult] = []
    final_evaluation: EvaluationResult | None = None

    start_time = time.perf_counter()

    try:
        for episode in range(config.episodes):
            observations, info = env.reset(
                seed=config.seed + episode
            )

            terminated = False
            truncated = False
            episode_return = 0.0
            episode_steps = 0

            current_noise_std = (
                exploration_noise_for_episode(
                    config,
                    episode,
                )
            )

            while not (terminated or truncated):
                if global_step < config.learning_starts:
                    actions = rng.uniform(
                        env.action_low,
                        env.action_high,
                        size=(
                            env.num_agents,
                            env.action_dim,
                        ),
                    ).astype(np.float32)
                else:
                    actions = learner.select_actions(
                        observations,
                        explore=True,
                        noise_std=current_noise_std,
                    )

                (
                    next_observations,
                    rewards,
                    terminated,
                    truncated,
                    info,
                ) = env.step(actions)

                rewards = reward_vector(
                    rewards,
                    env.num_agents,
                )

                terminations = np.full(
                    env.num_agents,
                    bool(terminated),
                    dtype=np.bool_,
                )

                truncations = np.full(
                    env.num_agents,
                    bool(truncated),
                    dtype=np.bool_,
                )

                replay_buffer.add(
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    terminations=terminations,
                    truncations=truncations,
                )

                observations = next_observations

                episode_return += float(
                    np.mean(rewards)
                )

                episode_steps += 1
                global_step += 1

                can_update = (
                    global_step >= config.learning_starts
                    and len(replay_buffer)
                    >= config.batch_size
                    and global_step
                    % config.update_every
                    == 0
                )

                if can_update:
                    batch = replay_buffer.sample(
                        config.batch_size,
                        replace=False,
                    )

                    statistics = learner.update(
                        batch
                    )

                    update_count += 1

                    for name, value in (
                        statistics.items()
                    ):
                        if not np.isfinite(value):
                            raise RuntimeError(
                                "Learner update produced "
                                f"non-finite {name}."
                            )

                        writer.add_scalar(
                            f"losses/{name}",
                            value,
                            global_step,
                        )

            episode_returns.append(
                episode_return
            )

            writer.add_scalar(
                "train/episode_return",
                episode_return,
                episode + 1,
            )
            writer.add_scalar(
                "train/episode_steps",
                episode_steps,
                episode + 1,
            )
            writer.add_scalar(
                "train/success",
                float(
                    bool(
                        info.get(
                            "success",
                            False,
                        )
                    )
                ),
                episode + 1,
            )
            writer.add_scalar(
                "train/exploration_noise_std",
                current_noise_std,
                episode + 1,
            )
            writer.add_scalar(
                "train/collision_count",
                _info_float(
                    info,
                    "collision_count",
                ),
                episode + 1,
            )
            writer.add_scalar(
                "train/mean_force_disagreement",
                _info_float(
                    info,
                    "mean_force_disagreement",
                ),
                episode + 1,
            )
            writer.add_scalar(
                "train/control_effort",
                _info_float(
                    info,
                    "total_control_effort",
                ),
                episode + 1,
            )
            writer.add_scalar(
                "train/final_abs_orientation",
                abs(
                    _info_float(
                        info,
                        "payload_orientation",
                    )
                ),
                episode + 1,
            )

            if (
                config.evaluation_interval > 0
                and (episode + 1)
                % config.evaluation_interval
                == 0
            ):
                evaluation = evaluate_transport(
                    learner,
                    config,
                )

                final_evaluation = evaluation

                evaluation_episode_indices.append(
                    episode + 1
                )
                evaluation_results.append(
                    evaluation
                )

                writer.add_scalar(
                    "eval/mean_return",
                    evaluation.mean_return,
                    episode + 1,
                )
                writer.add_scalar(
                    "eval/success_rate",
                    evaluation.success_rate,
                    episode + 1,
                )
                writer.add_scalar(
                    "eval/mean_steps",
                    evaluation.mean_steps,
                    episode + 1,
                )
                writer.add_scalar(
                    "eval/mean_final_distance",
                    evaluation.mean_final_distance,
                    episode + 1,
                )
                writer.add_scalar(
                    "eval/mean_collisions",
                    evaluation.mean_collisions,
                    episode + 1,
                )
                writer.add_scalar(
                    "eval/mean_force_disagreement",
                    evaluation.mean_force_disagreement,
                    episode + 1,
                )
                writer.add_scalar(
                    "eval/mean_control_effort",
                    evaluation.mean_control_effort,
                    episode + 1,
                )
                writer.add_scalar(
                    "eval/mean_max_abs_orientation",
                    evaluation.mean_max_abs_orientation,
                    episode + 1,
                )

            if (
                episode == 0
                or (episode + 1) % 10 == 0
                or episode + 1 == config.episodes
            ):
                print(
                    f"{config.algorithm} "
                    f"episode={episode + 1}/"
                    f"{config.episodes} "
                    f"return={episode_return:.3f} "
                    f"success="
                    f"{info.get('success', False)} "
                    f"steps={episode_steps} "
                    f"noise={current_noise_std:.3f}"
                )

        training_seconds = (
            time.perf_counter()
            - start_time
        )

        checkpoint_path: Path | None = None

        if config.checkpoint_path is not None:
            checkpoint_path = Path(
                config.checkpoint_path
            )
            learner.save(checkpoint_path)

        writer.flush()

        return TrainingResult(
            algorithm=config.algorithm,
            episodes_completed=config.episodes,
            environment_steps=global_step,
            update_count=update_count,
            mean_episode_return=float(
                np.mean(episode_returns)
            ),
            training_seconds=float(
                training_seconds
            ),
            final_evaluation=final_evaluation,
            training_episode_returns=tuple(
                episode_returns
            ),
            evaluation_episode_indices=tuple(
                evaluation_episode_indices
            ),
            evaluation_results=tuple(
                evaluation_results
            ),
            checkpoint_path=checkpoint_path,
        )

    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train MADDPG or Independent DDPG "
            "on cooperative transport."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to transport training YAML.",
    )

    args = parser.parse_args()

    config = TrainingConfig.from_yaml(
        args.config
    )

    result = train_transport(config)

    print()
    print("Training complete")
    print(result)


if __name__ == "__main__":
    main()