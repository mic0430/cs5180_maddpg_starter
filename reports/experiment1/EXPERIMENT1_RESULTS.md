# Experiment 1: Simple Spread Validation

## Objective

Validate the MADDPG and independent DDPG implementations on the
PettingZoo/MPE2 Simple Spread cooperative-navigation task.

The comparison used identical environment, network, optimization, evaluation,
and random-seed settings. The only algorithmic difference was the critic:

- MADDPG: centralized critic with decentralized actors
- Independent DDPG: local critic and local actor for each agent

## Compute-aware experimental design

The original plan used five seeds and 2,000 training episodes per seed.
Initial runtime testing showed that this scale was unnecessarily expensive
when considering the later transport and robustness experiments.

The final configuration was fixed before collecting the reported results:

- Seeds: 3, 7, and 11
- Training episodes per seed: 1,000
- Maximum steps per episode: 25
- Batch size: 128
- Learning starts: 512 environment steps
- Optimizer updates: every 2 environment steps
- Deterministic evaluation: every 100 episodes
- Evaluation episodes per checkpoint: 5

The same reduced budget was applied to both learned algorithms.

## Runtime

| Algorithm | Runtime |
|---|---:|
| MADDPG | 20 minutes 8.9 seconds |
| Independent DDPG | 18 minutes 49.5 seconds |
| Total training time | Approximately 39 minutes |

This confirmed that the revised experiment fit comfortably within the
project's five-to-six-hour computation limit per experiment.

## Performance across all evaluation checkpoints

| Algorithm | Mean evaluation return | Standard deviation |
|---|---:|---:|
| MADDPG | -20.498 | 1.090 |
| Independent DDPG | -20.269 | 1.286 |

These values average all deterministic evaluations from episodes 100 through
1,000. Independent DDPG obtained a marginally better full-trajectory mean,
although the difference was small relative to across-seed variability.

## Final deterministic evaluation at episode 1,000

| Seed | MADDPG | Independent DDPG | Better |
|---:|---:|---:|---|
| 3 | -18.274 | -18.744 | MADDPG |
| 7 | -18.802 | -20.381 | MADDPG |
| 11 | -18.604 | -19.582 | MADDPG |

| Algorithm | Final mean return | Final standard deviation |
|---|---:|---:|
| MADDPG | -18.560 | 0.267 |
| Independent DDPG | -19.569 | 0.819 |

Higher, or less-negative, returns are better. At the final checkpoint,
MADDPG outperformed independent DDPG on all three matched seeds. Its final
mean advantage was approximately 1.009 return points, and its final results
were more consistent across seeds.

## Interpretation

Both algorithms learned rapidly during the early portion of training and then
reached similar performance ranges. Independent DDPG had slightly stronger
average performance over the complete learning trajectory, while MADDPG
finished with the stronger deterministic policy at episode 1,000.

The expected directional result was therefore observed at the final
checkpoint, but the experiment should not be interpreted as definitive proof
of general MADDPG superiority. Only three seeds were used, the uncertainty
bands overlapped during much of training, and the training budget was reduced
to satisfy the available compute constraint.

Experiment 1 primarily serves as a correctness and reproducibility gate before
the custom cooperative-transport experiments.

## Preserved artifacts

- `comparison_summary.csv`
- `final_checkpoint_by_seed.csv`
- `final_checkpoint_summary.csv`
- `training_curves.png`
- `evaluation_curves.png`
- `final_checkpoint_by_seed.png`
