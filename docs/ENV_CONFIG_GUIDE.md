# Configuration Guide

## Overview

The cooperative transport environment is fully configurable through `config.yaml`.

Most experiments can be created by modifying configuration parameters without changing the environment source code. This allows different tasks to share the same physics implementation while evaluating different algorithms under consistent conditions.

---

# Configuration Structure

```text
config.yaml
│
├── env
├── world
├── agent
├── payload
├── coupling
├── target
├── obstacles
├── initial_state
├── action
├── observation
├── reward
├── termination
├── numerical_safety
└── evaluation
```

Each section controls one aspect of the environment.

---

# Configuration Sections

## env

General environment settings.

Typical parameters:

- `num_agents`
- `dt`
- `max_steps`
- `seed`

Used to control:

- number of cooperative agents
- physics simulation time step
- episode length
- random seed

---

## world

Workspace definition.

Typical parameters:

- world boundaries
- boundary handling mode

Example:

```yaml
world:
  x_min: -2
  x_max: 2
  y_min: -2
  y_max: 2
```

---

## agent

Agent physical properties.

Typical parameters:

- mass
- maximum control force
- maximum speed
- damping
- maximum attachment distance

These parameters determine how easily agents can move and how much force they can apply.

---

## payload

Payload geometry and dynamics.

Configurable properties include:

- shape
- dimensions
- mass
- damping
- rotation

Supported payload shapes:

| Shape | Dimensions | Rotation |
|--------|------------|----------|
| Point | None | No |
| Circle | Radius | Yes |
| Segment | Length | Yes |
| Rectangle | Width × Height | Yes |

Example:

```yaml
payload:
  shape: rectangle
  dimensions: [0.6, 0.3]
```

---

## attachments

Defines how agents connect to the payload.

Configurable parameters:

- attachment locations
- number of attachment points
- clearance checking
- clearance margin

The number of attachment points must match `env.num_agents`.

Example:

```yaml
attachments:
  offsets:
    - [-0.7, 0.0]
    - [ 0.7, 0.0]
```

---

## coupling

Defines the spring-damper model between each agent and its attachment.

Typical parameters:

- spring constant
- damping coefficient
- spring rest length
- maximum coupling force

Changing these values alters the stiffness of the connection.

---

## target

Defines the goal.

Typical parameters:

- target position
- goal radius
- success condition

Current implementation uses a circular goal region.

---

## obstacles

Defines obstacles in the workspace.

Configurable properties:

- enable / disable
- obstacle shape
- obstacle position
- obstacle size

Example:

```yaml
obstacles:
  enabled: true
```

---

## initial_state

Defines the initial environment state.

Typical parameters:

- payload position
- payload velocity
- payload orientation
- agent positions
- agent velocities
- random initialization noise

Noise can be added to improve policy robustness.

---

## observation

Controls which information is included in each agent's observation.

Typical options include:

- agent velocity
- attachment relative position
- payload velocity
- target relative position
- coupling force
- payload orientation
- payload angular velocity

Observation components can be enabled or disabled independently.

---

## reward

Defines the cooperative reward function.

Current reward consists of:

- progress reward
- success bonus
- action penalty

Additional reward terms can be added in future experiments if needed.

---

## termination

Defines episode termination conditions.

Current options include:

- terminate on success
- terminate on collision
- terminate on invalid numerical state

---

## numerical_safety

Provides numerical stability during physics simulation.

Examples:

- speed clipping
- angular speed clipping
- NaN / Inf detection

---

## evaluation

Controls which metrics are recorded during evaluation.

Examples include:

- success rate
- episode return
- completion time
- control effort
- force disagreement
- payload orientation
- collision count

These metrics are used for analysis only and do not affect training.

---

# Common Configuration Changes

## Increase the number of agents

```yaml
env:
  num_agents: 3
```

Update the attachment locations and initial agent states accordingly.

---

## Change the payload shape

```yaml
payload:
  shape: rectangle
  dimensions: [0.8, 0.4]
```

---

## Increase task difficulty

Possible approaches include:

- increasing payload mass
- reducing target radius
- decreasing maximum control force
- adding obstacles

---

## Enable obstacle navigation

```yaml
obstacles:
  enabled: true
```

Then define obstacle positions and sizes.

---

## Modify spring behavior

Adjust:

```yaml
coupling:
  spring_constant:
  spring_damping:
  rest_length:
```

to create softer or stiffer couplings.

---

## Randomize initial states

Increase initialization noise:

```yaml
initial_state:
  payload_position_noise: 0.05
  agent_position_noise: 0.05
```

This improves robustness and prevents overfitting to a single initial configuration.

---

# Design Principle

Whenever possible, new experiments should be created by modifying `config.yaml` rather than changing the environment source code.

The environment should only be modified when introducing new task mechanics, physics models, observation definitions, or reward structures.

This separation keeps all algorithms operating on the same task and ensures fair comparisons across different controllers and reinforcement learning methods.
