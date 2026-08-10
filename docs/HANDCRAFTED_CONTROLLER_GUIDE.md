# Hand-Crafted Force-Feedback Controller

## Overview

This controller is a deterministic decentralized baseline for cooperative transport.

Each agent uses only its own local observation. It does not access teammate state, global environment state, or obstacle locations during execution.

Implementation:

`src/controllers/force_feedback.py`

## Controller

The controller combines:

- goal-directed motion
- attachment-position correction
- payload-relative velocity feedback
- local coupling-force feedback
- agent velocity damping
- bounded 2-D actions

Default gains:

- goal: 2.00
- attachment: 0.35
- relative velocity: 0.20
- coupling force: 0.15
- agent damping: 0.10
- action limit: 1.00

The goal gain was selected using a separate 20-seed development set.

## Narrow-Passage Task

Configuration:

`configs/cooperative_transport_narrow_passage.yaml`

The rectangular payload must travel through a narrow straight passage. Agents are not given obstacle locations, so the task evaluates transport precision and decentralized coordination rather than obstacle-aware path planning.

Held-out 10-seed evaluation with the selected controller achieved 8/10 successful deliveries.

## Tests

Controller tests:

`tests/test_force_feedback_controller.py`

Run:

`python -m pytest tests/test_force_feedback_controller.py -q`

The tests include an end-to-end narrow-passage environment smoke test.