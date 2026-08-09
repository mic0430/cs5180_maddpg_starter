import numpy as np
import pytest

from src.envs.transport_env import CooperativeTransportEnv


# ============================================================
# Base config
# ============================================================


def make_config():
    return {
        "env": {
            "name": "cooperative_transport",
            "seed": 3,
            "num_agents": 2,
            "dt": 0.05,
            "max_steps": 300,
        },

        "world": {
            "x_min": -2.0,
            "x_max": 2.0,
            "y_min": -2.0,
            "y_max": 2.0,
            "boundary_mode": "clip",
        },

        "agent": {
            "mass": 1.0,
            "max_control_force": 1.0,
            "max_speed": 2.0,
            "linear_damping": 0.20,
            "max_attachment_distance": 0.35,
        },

        "payload": {
            "shape": "rectangle",
            "dimensions": [0.60, 0.30],
            "mass": 3.0,
            "linear_damping": 0.30,
            "max_speed": 2.0,
            "angular_damping": 0.20,
            "max_angular_speed": 4.0,

            "attachments": {
                "offsets": [
                    [-0.70, 0.00],
                    [0.70, 0.00],
                ],
                "enforce_clearance": True,
                "clearance_margin": 0.02,
            },
        },

        "coupling": {
            "spring_constant": 8.0,
            "spring_damping": 0.50,
            "rest_length": 0.20,
            "max_force": 5.0,
            "epsilon": 1e-8,
        },

        "target": {
            "position": [1.50, 0.00],
            "radius": 0.20,
            "success_mode": "center_inside",
            "require_full_payload_inside": False,
        },

        "obstacles": {
            "enabled": False,
            "items": [],
            "collision_mode": "reject_motion",
        },

        "initial_state": {
            "payload_position": [0.0, 0.0],
            "payload_velocity": [0.0, 0.0],
            "payload_orientation": 0.0,
            "payload_angular_velocity": 0.0,

            # Attachments = +/-0.70.
            # rest_length = 0.20.
            # Therefore agents begin exactly at rest length.
            "agent_positions": [
                [-0.90, 0.00],
                [0.90, 0.00],
            ],

            "agent_velocities": [
                [0.0, 0.0],
                [0.0, 0.0],
            ],

            "payload_position_noise": 0.0,
            "agent_position_noise": 0.0,
            "payload_velocity_noise": 0.0,
            "agent_velocity_noise": 0.0,
            "orientation_noise": 0.0,
            "angular_velocity_noise": 0.0,
        },

        "action": {
            "dimension": 2,
            "low": -1.0,
            "high": 1.0,
        },

        "observation": {
            "include_agent_velocity": True,
            "include_attachment_relative_position": True,
            "include_payload_relative_velocity": True,
            "include_goal_relative_to_payload": True,
            "include_coupling_force": True,
            "include_payload_orientation": True,
            "include_payload_angular_velocity": True,
        },

        "reward": {
            "shared": True,
            "progress_weight": 1.0,
            "success_bonus": 10.0,
            "action_penalty_weight": 0.001,
        },

        "termination": {
            "terminate_on_success": True,
            "terminate_on_nonfinite_state": True,
            "terminate_on_collision": False,
        },

        "numerical_safety": {
            "clip_agent_speed": True,
            "clip_payload_speed": True,
            "clip_angular_speed": True,
            "reject_nonfinite_values": True,
        },

        "evaluation": {
            "track_success": True,
            "track_episode_return": True,
            "track_completion_time": True,
            "track_control_effort": True,
            "track_coupling_forces": True,
            "track_force_disagreement": True,
            "track_payload_distance_to_goal": True,
            "track_payload_orientation": True,
            "track_payload_angular_velocity": True,
            "track_obstacle_collisions": True,
        },
    }


def make_env():
    return CooperativeTransportEnv(make_config())


# ============================================================
# Initialization
# ============================================================


def test_env_initialization():
    env = make_env()

    assert env.num_agents == 2
    assert env.dt == pytest.approx(0.05)
    assert env.max_steps == 300

    assert env.action_dim == 2

    assert env.payload.geometry.shape_name == "rectangle"

    assert env.agent_positions.shape == (2, 2)
    assert env.agent_velocities.shape == (2, 2)
    assert env.coupling_forces.shape == (2, 2)

    assert env.last_actions.shape == (2, 2)


def test_initial_payload_state():
    env = make_env()

    np.testing.assert_allclose(
        env.payload.position,
        [0.0, 0.0],
        atol=1e-8,
    )

    np.testing.assert_allclose(
        env.payload.velocity,
        [0.0, 0.0],
        atol=1e-8,
    )

    assert env.payload.orientation == pytest.approx(0.0)
    assert env.payload.angular_velocity == pytest.approx(0.0)


# ============================================================
# Reset
# ============================================================


def test_reset_returns_observation_and_info():
    env = make_env()

    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)

    assert obs.shape == (2, 12)


def test_reset_restores_episode_counters():
    env = make_env()

    env.step(
        np.array([
            [1.0, 0.0],
            [1.0, 0.0],
        ])
    )

    env.reset()

    assert env.step_count == 0
    assert env.episode_return == pytest.approx(0.0)

    assert env.collision_count == 0

    assert env.total_control_effort == pytest.approx(0.0)
    assert env.total_force_disagreement == pytest.approx(0.0)

    assert not env.success_reward_given


def test_reset_restores_agent_positions():
    env = make_env()

    env.agent_positions[:] += 0.5

    env.reset()

    np.testing.assert_allclose(
        env.agent_positions,
        [
            [-0.90, 0.00],
            [0.90, 0.00],
        ],
        atol=1e-8,
    )


def test_reset_seed_is_reproducible():
    config = make_config()

    config["initial_state"]["agent_position_noise"] = 0.1

    env = CooperativeTransportEnv(config)

    obs1, _ = env.reset(seed=123)
    positions1 = env.agent_positions.copy()

    obs2, _ = env.reset(seed=123)
    positions2 = env.agent_positions.copy()

    np.testing.assert_allclose(
        positions1,
        positions2,
    )

    np.testing.assert_allclose(
        obs1,
        obs2,
    )


# ============================================================
# Observation
# ============================================================


def test_observation_dimension():
    env = make_env()

    obs, _ = env.reset()

    # velocity                     2
    # attachment relative pos      2
    # payload relative velocity    2
    # goal relative payload        2
    # coupling force               2
    # orientation                  1
    # angular velocity             1
    #
    # total = 12
    assert obs.shape == (2, 12)

    assert env.observation_dim == 12


def test_point_payload_has_same_observation_dimension():
    config = make_config()

    config["payload"]["shape"] = "point"
    config["payload"]["dimensions"] = []

    config["payload"]["attachments"][
        "enforce_clearance"
    ] = False

    env = CooperativeTransportEnv(config)

    obs, _ = env.reset()

    assert obs.shape == (2, 12)

    # Point payload has zero orientation and angular velocity.
    np.testing.assert_allclose(
        obs[:, -2:],
        0.0,
        atol=1e-8,
    )


def test_local_observations_are_agent_specific():
    env = make_env()

    obs, _ = env.reset()

    # Their attachment-relative positions point in
    # opposite directions.
    assert not np.allclose(
        obs[0],
        obs[1],
    )


# ============================================================
# Action handling
# ============================================================


def test_invalid_action_shape_raises():
    env = make_env()

    with pytest.raises(ValueError):
        env.step(
            np.array([1.0, 0.0])
        )


def test_actions_are_clipped():
    env = make_env()

    env.step(
        np.array([
            [100.0, 0.0],
            [-100.0, 0.0],
        ])
    )

    np.testing.assert_allclose(
        env.last_actions,
        [
            [1.0, 0.0],
            [-1.0, 0.0],
        ],
    )


# ============================================================
# Initial spring equilibrium
# ============================================================


def test_initial_spring_forces_are_zero():
    env = make_env()

    forces = env._compute_coupling_forces()

    np.testing.assert_allclose(
        forces,
        np.zeros((2, 2)),
        atol=1e-8,
    )


def test_zero_action_equilibrium():
    env = make_env()

    old_payload_pos = env.payload.position.copy()
    old_payload_orientation = env.payload.orientation

    old_agent_positions = env.agent_positions.copy()

    env.step(
        np.zeros((2, 2))
    )

    np.testing.assert_allclose(
        env.payload.position,
        old_payload_pos,
        atol=1e-8,
    )

    assert env.payload.orientation == pytest.approx(
        old_payload_orientation
    )

    np.testing.assert_allclose(
        env.agent_positions,
        old_agent_positions,
        atol=1e-8,
    )


# ============================================================
# Coupling force
# ============================================================


def test_stretched_right_agent_pulled_left():
    env = make_env()

    env.agent_positions[1] = np.array(
        [1.0, 0.0]
    )

    force = env._compute_coupling_forces()[1]

    assert force[0] < 0.0
    assert force[1] == pytest.approx(0.0)


def test_stretched_left_agent_pulled_right():
    env = make_env()

    env.agent_positions[0] = np.array(
        [-1.0, 0.0]
    )

    force = env._compute_coupling_forces()[0]

    assert force[0] > 0.0


def test_coupling_force_is_clipped():
    config = make_config()

    config["coupling"]["max_force"] = 0.1

    env = CooperativeTransportEnv(config)

    env.agent_positions[1] = np.array(
        [1.05, 0.0]
    )

    force = env._compute_coupling_forces()[1]

    assert (
        np.linalg.norm(force)
        <= 0.1 + 1e-8
    )


# ============================================================
# Agent dynamics
# ============================================================


def test_agent_control_moves_agent():
    env = make_env()

    old_position = env.agent_positions[0].copy()

    env.step(
        np.array([
            [1.0, 0.0],
            [0.0, 0.0],
        ])
    )

    assert (
        env.agent_positions[0, 0]
        > old_position[0]
    )


def test_agent_speed_clipping():
    config = make_config()

    config["agent"]["max_speed"] = 0.1

    env = CooperativeTransportEnv(config)

    for _ in range(50):
        env.step(
            np.array([
                [1.0, 0.0],
                [1.0, 0.0],
            ])
        )

    speeds = np.linalg.norm(
        env.agent_velocities,
        axis=1,
    )

    assert np.all(
        speeds <= 0.1 + 1e-8
    )


# ============================================================
# Attachment constraint
# ============================================================


def test_attachment_distance_constraint():
    env = make_env()

    env.agent_positions[1] = np.array(
        [2.0, 0.0]
    )

    env._enforce_all_attachment_constraints()

    attachment = (
        env.payload.get_attachment_position(1)
    )

    distance = np.linalg.norm(
        env.agent_positions[1]
        - attachment
    )

    assert distance <= (
        env.max_attachment_distance + 1e-8
    )


def test_outward_radial_velocity_removed():
    env = make_env()

    attachment = (
        env.payload.get_attachment_position(1)
    )

    env.agent_positions[1] = (
        attachment
        + np.array([1.0, 0.0])
    )

    env.agent_velocities[1] = np.array(
        [1.0, 0.0]
    )

    env._enforce_all_attachment_constraints()

    assert env.agent_velocities[1, 0] <= 1e-8


# ============================================================
# Payload translation / rotation
# ============================================================


def test_symmetric_forces_translate_without_rotation():
    env = make_env()

    left = env.payload.get_attachment_position(0)
    right = env.payload.get_attachment_position(1)

    env.payload.apply_force_at_point(
        [1.0, 0.0],
        left,
    )

    env.payload.apply_force_at_point(
        [1.0, 0.0],
        right,
    )

    env.payload.integrate(
        dt=0.1,
        clip_linear_speed=False,
        clip_angular_speed=False,
    )

    assert env.payload.position[0] > 0.0

    assert env.payload.orientation == pytest.approx(
        0.0,
        abs=1e-8,
    )


def test_asymmetric_force_generates_rotation():
    env = make_env()

    right = env.payload.get_attachment_position(1)

    env.payload.apply_force_at_point(
        force=[0.0, 1.0],
        world_point=right,
    )

    env.payload.integrate(
        dt=0.1,
        clip_linear_speed=False,
        clip_angular_speed=False,
    )

    assert env.payload.angular_velocity > 0.0
    assert env.payload.orientation > 0.0


def test_equal_opposite_torques_cancel():
    env = make_env()

    left = env.payload.get_attachment_position(0)
    right = env.payload.get_attachment_position(1)

    env.payload.apply_force_at_point(
        [0.0, 1.0],
        left,
    )

    env.payload.apply_force_at_point(
        [0.0, 1.0],
        right,
    )

    assert env.payload.torque_accumulator == pytest.approx(
        0.0,
        abs=1e-8,
    )


# ============================================================
# Reward
# ============================================================


def test_reward_is_shared():
    env = make_env()

    _, rewards, _, _, _ = env.step(
        np.zeros((2, 2))
    )

    assert rewards.shape == (2,)

    assert rewards[0] == pytest.approx(
        rewards[1]
    )


def test_zero_action_zero_progress_reward():
    env = make_env()

    _, rewards, _, _, _ = env.step(
        np.zeros((2, 2))
    )

    assert rewards[0] == pytest.approx(
        0.0,
        abs=1e-8,
    )


def test_action_penalty():
    config = make_config()

    config["reward"]["progress_weight"] = 0.0
    config["reward"]["success_bonus"] = 0.0
    config["reward"]["action_penalty_weight"] = 0.1

    env = CooperativeTransportEnv(config)

    actions = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    _, rewards, _, _, _ = env.step(actions)

    # Sum squared action = 2.
    # penalty = 0.1 * 2 = 0.2
    assert rewards[0] == pytest.approx(
        -0.2
    )


def test_success_bonus_given_once():
    config = make_config()

    config["target"]["position"] = [0.0, 0.0]

    # Keep episode alive after success so we can check
    # bonus is not repeated.
    config["termination"][
        "terminate_on_success"
    ] = False

    env = CooperativeTransportEnv(config)

    _, reward1, terminated1, _, _ = env.step(
        np.zeros((2, 2))
    )

    _, reward2, terminated2, _, _ = env.step(
        np.zeros((2, 2))
    )

    assert reward1[0] == pytest.approx(
        10.0
    )

    assert reward2[0] == pytest.approx(
        0.0
    )

    assert not terminated1
    assert not terminated2


# ============================================================
# Success / termination
# ============================================================


def test_success_detection():
    config = make_config()

    config["target"]["position"] = [0.0, 0.0]

    env = CooperativeTransportEnv(config)

    assert env._check_success()


def test_success_terminates_episode():
    config = make_config()

    config["target"]["position"] = [0.0, 0.0]

    env = CooperativeTransportEnv(config)

    _, _, terminated, truncated, _ = env.step(
        np.zeros((2, 2))
    )

    assert terminated
    assert not truncated


def test_max_steps_truncates_episode():
    config = make_config()

    config["env"]["max_steps"] = 2

    env = CooperativeTransportEnv(config)

    _, _, terminated1, truncated1, _ = env.step(
        np.zeros((2, 2))
    )

    assert not terminated1
    assert not truncated1

    _, _, terminated2, truncated2, _ = env.step(
        np.zeros((2, 2))
    )

    assert not terminated2
    assert truncated2


# ============================================================
# World boundaries
# ============================================================


def test_payload_boundary_clipping():
    env = make_env()

    env.payload.position = np.array(
        [10.0, -10.0]
    )

    env._handle_world_boundaries()

    np.testing.assert_allclose(
        env.payload.position,
        [2.0, -2.0],
    )


def test_agent_boundary_clipping():
    env = make_env()

    env.agent_positions[0] = np.array(
        [100.0, -100.0]
    )

    env._handle_world_boundaries()

    np.testing.assert_allclose(
        env.agent_positions[0],
        [2.0, -2.0],
    )


# ============================================================
# Obstacles
# ============================================================


def test_payload_obstacle_collision():
    config = make_config()

    config["obstacles"] = {
        "enabled": True,
        "collision_mode": "reject_motion",
        "items": [
            {
                "shape": "rectangle",
                "position": [0.0, 0.0],
                "dimensions": [0.5, 0.5],
                "orientation": 0.0,
            },
        ],
    }

    env = CooperativeTransportEnv(config)

    assert env._payload_collides_with_obstacle()


def test_distant_obstacle_has_no_collision():
    config = make_config()

    config["obstacles"] = {
        "enabled": True,
        "collision_mode": "reject_motion",
        "items": [
            {
                "shape": "rectangle",
                "position": [1.8, 1.8],
                "dimensions": [0.2, 0.2],
                "orientation": 0.0,
            },
        ],
    }

    env = CooperativeTransportEnv(config)

    assert not env._payload_collides_with_obstacle()


def test_collision_snapshot_restore():
    env = make_env()

    snapshot = env._payload_snapshot()

    env.payload.position[:] = [1.0, 1.0]
    env.payload.velocity[:] = [1.0, 1.0]
    env.payload.orientation = 0.5
    env.payload.angular_velocity = 2.0

    env._restore_payload_snapshot(snapshot)

    np.testing.assert_allclose(
        env.payload.position,
        snapshot["position"],
    )

    np.testing.assert_allclose(
        env.payload.velocity,
        snapshot["velocity"],
    )

    assert env.payload.orientation == pytest.approx(
        snapshot["orientation"]
    )

    assert env.payload.angular_velocity == pytest.approx(
        snapshot["angular_velocity"]
    )


def test_collision_rollback_preserves_attachment_constraint():
    config = make_config()

    config["obstacles"] = {
        "enabled": True,
        "collision_mode": "reject_motion",
        "items": [
            {
                "shape": "rectangle",
                "position": [0.35, 0.0],
                "dimensions": [0.20, 1.0],
                "orientation": 0.0,
            },
        ],
    }

    env = CooperativeTransportEnv(config)

    env.payload.velocity = np.array(
        [2.0, 0.0]
    )

    env.step(
        np.zeros((2, 2))
    )

    attachments = (
        env.payload.get_attachment_positions()
    )

    distances = np.linalg.norm(
        env.agent_positions
        - attachments,
        axis=1,
    )

    assert np.all(
        distances
        <= env.max_attachment_distance + 1e-8
    )


# ============================================================
# Force disagreement
# ============================================================


def test_identical_forces_have_zero_disagreement():
    env = make_env()

    env.coupling_forces[:] = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
    ])

    assert env._force_disagreement() == pytest.approx(
        0.0
    )


def test_conflicting_forces_have_positive_disagreement():
    env = make_env()

    env.coupling_forces[:] = np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
    ])

    assert env._force_disagreement() > 0.0


def test_force_disagreement_accumulates():
    env = make_env()

    env.agent_positions[0] = np.array(
        [-1.0, 0.0]
    )

    env.step(
        np.zeros((2, 2))
    )

    assert env.total_force_disagreement >= 0.0


# ============================================================
# Control effort
# ============================================================


def test_control_effort_accumulates():
    env = make_env()

    actions = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    _, _, _, _, info = env.step(actions)

    # 1^2 + 1^2 = 2
    assert info[
        "total_control_effort"
    ] == pytest.approx(2.0)


def test_zero_action_has_zero_control_effort():
    env = make_env()

    _, _, _, _, info = env.step(
        np.zeros((2, 2))
    )

    assert info[
        "total_control_effort"
    ] == pytest.approx(0.0)


# ============================================================
# Global centralized state
# ============================================================


def test_global_state_vector_shape():
    env = make_env()

    state = env.global_state_vector()

    # Two agents:
    #
    # positions           4
    # velocities          4
    # payload position    2
    # payload velocity    2
    # orientation         1
    # angular velocity    1
    # relative target     2
    #
    # total              16
    assert state.shape == (16,)


def test_global_state_dim():
    env = make_env()

    assert env.global_state_dim == 16


def test_joint_action_dim():
    env = make_env()

    assert env.joint_action_dim == 4


def test_global_state_changes_when_payload_moves():
    env = make_env()

    state1 = env.global_state_vector().copy()

    env.payload.position += np.array(
        [0.2, 0.0]
    )

    state2 = env.global_state_vector()

    assert not np.allclose(
        state1,
        state2,
    )


# ============================================================
# Full state dictionary
# ============================================================


def test_state_dictionary():
    env = make_env()

    state = env.state()

    expected = {
        "agent_positions",
        "agent_velocities",
        "payload",
        "target_position",
        "coupling_forces",
    }

    assert expected.issubset(
        state.keys()
    )

    assert state[
        "agent_positions"
    ].shape == (2, 2)

    assert state[
        "agent_velocities"
    ].shape == (2, 2)


# ============================================================
# Info / metrics
# ============================================================


def test_info_contains_stage2_metrics():
    env = make_env()

    _, _, _, _, info = env.step(
        np.zeros((2, 2))
    )

    expected_keys = {
        "success",
        "step_count",
        "episode_return",
        "payload_distance_to_goal",
        "payload_position",
        "payload_velocity",
        "payload_orientation",
        "payload_angular_velocity",
        "coupling_forces",
        "force_disagreement",
        "mean_force_disagreement",
        "total_control_effort",
        "mean_attachment_distance",
        "collision_count",
        "collision",
        "finite_state",
    }

    assert expected_keys.issubset(
        info.keys()
    )


def test_mean_attachment_distance_at_initial_state():
    env = make_env()

    info = env._build_info()

    # Initial agent-attachment distance = 0.20.
    assert info[
        "mean_attachment_distance"
    ] == pytest.approx(
        0.20
    )


# ============================================================
# Numerical validity
# ============================================================


def test_normal_state_is_finite():
    env = make_env()

    assert env._state_is_finite()


def test_nan_agent_position_invalid():
    env = make_env()

    env.agent_positions[0, 0] = np.nan

    assert not env._state_is_finite()


def test_inf_agent_velocity_invalid():
    env = make_env()

    env.agent_velocities[0, 1] = np.inf

    assert not env._state_is_finite()


def test_nan_payload_invalid():
    env = make_env()

    env.payload.position[0] = np.nan

    assert not env._state_is_finite()


def test_nan_coupling_force_invalid():
    env = make_env()

    env.coupling_forces[0, 0] = np.nan

    assert not env._state_is_finite()


# ============================================================
# Config validation
# ============================================================


def test_invalid_num_agents():
    config = make_config()

    config["env"]["num_agents"] = 0

    with pytest.raises(ValueError):
        CooperativeTransportEnv(config)


def test_invalid_dt():
    config = make_config()

    config["env"]["dt"] = 0.0

    with pytest.raises(ValueError):
        CooperativeTransportEnv(config)


def test_invalid_max_steps():
    config = make_config()

    config["env"]["max_steps"] = 0

    with pytest.raises(ValueError):
        CooperativeTransportEnv(config)


def test_invalid_world_bounds():
    config = make_config()

    config["world"]["x_min"] = 2.0
    config["world"]["x_max"] = -2.0

    with pytest.raises(ValueError):
        CooperativeTransportEnv(config)


def test_invalid_agent_mass():
    config = make_config()

    config["agent"]["mass"] = 0.0

    with pytest.raises(ValueError):
        CooperativeTransportEnv(config)


def test_invalid_rest_length():
    config = make_config()

    config["coupling"]["rest_length"] = 0.35

    with pytest.raises(ValueError):
        CooperativeTransportEnv(config)


def test_attachment_count_must_match_agents():
    config = make_config()

    config["payload"][
        "attachments"
    ]["offsets"] = [
        [-0.7, 0.0]
    ]

    with pytest.raises(ValueError):
        CooperativeTransportEnv(config)


def test_initial_position_count_must_match_agents():
    config = make_config()

    config["initial_state"][
        "agent_positions"
    ] = [
        [-0.9, 0.0]
    ]

    with pytest.raises(ValueError):
        CooperativeTransportEnv(config)


def test_initial_velocity_count_must_match_agents():
    config = make_config()

    config["initial_state"][
        "agent_velocities"
    ] = [
        [0.0, 0.0]
    ]

    with pytest.raises(ValueError):
        CooperativeTransportEnv(config)


def test_invalid_action_dimension():
    config = make_config()

    config["action"]["dimension"] = 3

    with pytest.raises(ValueError):
        CooperativeTransportEnv(config)


def test_invalid_target_radius():
    config = make_config()

    config["target"]["radius"] = 0.0

    with pytest.raises(ValueError):
        CooperativeTransportEnv(config)


# ============================================================
# Variable agent count
# ============================================================


def test_three_agent_environment():
    config = make_config()

    config["env"]["num_agents"] = 3

    config["payload"]["attachments"]["offsets"] = [
        [-0.70, 0.00],
        [0.70, 0.00],
        [0.00, 0.70],
    ]

    config["initial_state"]["agent_positions"] = [
        [-0.90, 0.00],
        [0.90, 0.00],
        [0.00, 0.90],
    ]

    config["initial_state"]["agent_velocities"] = [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]

    env = CooperativeTransportEnv(config)

    obs, _ = env.reset()

    assert env.num_agents == 3

    assert obs.shape == (3, 12)

    assert env.joint_action_dim == 6

    # Global state:
    #
    # agent positions      6
    # agent velocities     6
    # payload position     2
    # payload velocity     2
    # orientation          1
    # angular velocity     1
    # target relative      2
    #
    # = 20
    assert env.global_state_dim == 20


# ============================================================
# Basic multi-agent cooperation sanity checks
# ============================================================


def test_coordinated_actions_move_both_agents_same_direction():
    env = make_env()

    old_positions = env.agent_positions.copy()

    env.step(
        np.array([
            [1.0, 0.0],
            [1.0, 0.0],
        ])
    )

    assert (
        env.agent_positions[0, 0]
        > old_positions[0, 0]
    )

    assert (
        env.agent_positions[1, 0]
        > old_positions[1, 0]
    )


def test_conflicting_actions_create_different_agent_velocities():
    env = make_env()

    env.step(
        np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
        ])
    )

    assert (
        env.agent_velocities[0, 0]
        > 0
    )

    assert (
        env.agent_velocities[1, 0]
        < 0
    )
    