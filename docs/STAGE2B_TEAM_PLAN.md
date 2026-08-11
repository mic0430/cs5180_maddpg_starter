# Stage 2B — Team Plan

## Goal

Stage 2B builds the custom cooperative-transport system needed for
Experiments 2 and 3.

All three tasks are developed independently and then integrated.

---

## Individual Tasks

### Michael — Issue #8
**Hand-Crafted Force-Feedback Controller**

Branch:

`michael-handcrafted-controller`

Responsible for:

- hand-crafted force-feedback controller
- decentralized controller logic
- controller configuration
- controller tests

---

### Alex — Issue #9
**Cooperative Transport Environment**

Branch:

`transport-environment`

Responsible for:

- 2-D cooperative transport environment
- shared payload
- spring/coupling dynamics
- observations and rewards
- success/termination conditions
- environment tests

---

### Aaron — Issue #10
**Robustness and Evaluation**

Branch:

`robustness-evaluation`

Responsible for:

- observation noise
- action deadzone
- transport metrics
- result aggregation
- robustness plots
- evaluation tests

---

# How the Three Tasks Fit Together

```text
Michael
Hand-Crafted Controller
        │
        │
Alex ───┼──→ Cooperative Transport System
Environment
        │
        │
Aaron ──┘
Robustness + Evaluation


After we integrate we can begin Experiment 2 and 3 (stage 3), the project will be roughly 60% done overall.