# Cooperative Transport Environment

## Introduction

`CooperativeTransportEnv` is a configurable multi-agent cooperative transport environment designed for evaluating cooperative reinforcement learning algorithms, including MADDPG, Independent DDPG, and hand-crafted controllers.

The task requires multiple agents to transport a shared payload from an initial position to a target region through spring-damper couplings. The environment is designed to be extensible while keeping a unified interface for all algorithms.

### Current Features

- Configurable number of cooperative agents.
- Multiple payload geometries:
  - Point
  - Circle
  - Line Segment
  - Rectangle
- Configurable payload dimensions.
- Configurable payload mass and damping.
- Optional payload rotation (enabled for non-point payloads).
- Configurable attachment locations on the payload.
- Spring-damper coupling between each agent and its assigned attachment.
- Agent-to-attachment distance constraints.
- Configurable target region.
- Optional obstacle configuration with collision checking.
- Shared cooperative reward.
- Local observations for decentralized actors.
- Continuous 2-D action space.
- Unified configuration through `config.yaml`.

The environment is intentionally independent of any specific reinforcement learning algorithm. It defines only the task dynamics, physics, observations, rewards, and termination conditions. Learning algorithms interact with the environment through the standard `reset()` and `step()` interfaces.

The current design also supports future extensions, including more complex payload geometries, obstacle-rich navigation tasks, and cooperative manipulation requiring payload rotation.

---

## Environment API

### Reset

```python
observations, info = env.reset()
```

Returns:

- `observations`: `(num_agents, observation_dim)`
- `info`: environment information

---

### Step

```python
next_observations, rewards, terminated, truncated, info = env.step(actions)
```

Input:

```python
actions.shape == (num_agents, action_dim)
```

Returns:

```python
next_observations.shape == (num_agents, observation_dim)

rewards.shape == (num_agents,)
```

where

- `terminated` indicates task termination.
- `truncated` indicates the episode reached the maximum number of steps.

---

## Observation

Each agent receives its own local observation.

The default observation contains:

- agent velocity
- attachment relative position
- payload relative velocity
- target relative position
- coupling force
- payload orientation
- payload angular velocity

All agents have the same observation dimension.

---

## Action

Each agent outputs a continuous 2-D action:

```text
[a_x, a_y]
```

with

```text
a_x, a_y ∈ [-1, 1]
```

The environment converts actions into physical control forces internally.

---

## Reward

A shared team reward is used.

The reward consists of:

- progress toward the target
- one-time success bonus
- small action penalty

All agents receive the same reward.

---

## Termination

The environment returns

```python
terminated: bool
truncated: bool
```

The cooperative task ends simultaneously for all agents.

---

## Integration with MADDPG

The current MADDPG implementation already accepts observations with shape

```text
(num_agents, observation_dim)
```

and outputs actions with shape

```text
(num_agents, action_dim)
```

No PettingZoo wrapper is required.

The centralized critic constructs joint observations internally.

---

## Replay Buffer

The ReplayBuffer stores one termination value per agent.

Before inserting a transition, convert

```python
terminated
```

into

```python
termination_vector = np.full(
    env.num_agents,
    terminated,
    dtype=np.bool_,
)
```

This adaptation should be performed in the training loop rather than inside the environment.

---

## Design Principle

`CooperativeTransportEnv` should remain algorithm-independent.

If future integration issues occur, the preferred modification order is:

1. Training loop
2. ReplayBuffer
3. Wrapper / Adapter

The environment should only be modified when the **task definition**, **physics model**, or **observation/reward design** changes.

---

## Data Flow

```text
Configuration
      │
      ▼
CooperativeTransportEnv
      │
      ▼
Observations
      │
      ▼
MADDPG / IDDPG / Controller
      │
      ▼
Actions
      │
      ▼
Environment Step
      │
      ▼
Replay Buffer
      │
      ▼
Training
```