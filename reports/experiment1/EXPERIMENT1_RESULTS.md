# Experiment 1 — MADDPG vs Independent DDPG on Simple Spread

## Purpose

Experiment 1 serves as a correctness and reproducibility benchmark before evaluating the algorithms on the custom cooperative-transport environment.

The comparison tests whether centralized-critic training with MADDPG provides an advantage over Independent DDPG under matched execution conditions.

At execution time, both methods use decentralized actors. The primary difference is the critic information available during training:

- MADDPG uses centralized critics conditioned on joint multi-agent information.
- Independent DDPG uses a separate local critic for each agent.

Less-negative return indicates better performance.

---

## Experimental Setup

Both algorithms used the same training settings:

- Environment: PettingZoo/MPE2 Simple Spread
- Number of agents: 3
- Training episodes: 1,000
- Episode horizon: 25 steps
- Batch size: 128
- Replay capacity: 100,000
- Learning starts: 512 environment steps
- Update frequency: every 2 environment steps
- Discount factor: 0.95
- Target-update coefficient: 0.01
- Actor learning rate: 0.001
- Critic learning rate: 0.001
- Hidden layers: [64, 64]
- Exploration noise: 0.20 → 0.05
- Exploration decay: 800 episodes
- Deterministic evaluation interval: every 100 episodes
- Evaluation episodes per checkpoint: 5
- Evaluation uses exploration disabled

Ten unique matched seeds were used for both algorithms:

`3, 7, 11, 19, 23, 29, 31, 37, 41, 47`

This produced:

- 10 MADDPG runs
- 10 Independent DDPG runs
- 20 total training runs

---

## Evaluation Performance Across Training

The average evaluation statistics across the ten seed runs were:

| Algorithm | Mean Evaluation Return | Standard Deviation |
|---|---:|---:|
| MADDPG | -20.346 | 1.351 |
| Independent DDPG | -20.631 | 1.756 |

MADDPG therefore had a slightly better mean deterministic evaluation return across the evaluation trajectory and lower across-seed variability.

The deterministic evaluation curve was also aggregated separately at every 100-episode checkpoint using the mean and standard deviation across all ten seeds.

At Episode 1,000:

| Algorithm | Mean Return | Standard Deviation |
|---|---:|---:|
| MADDPG | -19.396 | 1.046 |
| Independent DDPG | -19.963 | 1.879 |

---

## Final Deterministic Evaluation at Episode 1,000

Final deterministic returns for each matched seed were:

| Seed | MADDPG | Independent DDPG | MADDPG - DDPG | Winner |
|---:|---:|---:|---:|---|
| 3 | -18.274 | -18.744 | +0.470 | MADDPG |
| 7 | -18.802 | -20.381 | +1.579 | MADDPG |
| 11 | -18.604 | -19.582 | +0.978 | MADDPG |
| 19 | -19.712 | -18.369 | -1.343 | Independent DDPG |
| 23 | -18.687 | -20.210 | +1.522 | MADDPG |
| 29 | -18.876 | -20.992 | +2.117 | MADDPG |
| 31 | -20.623 | -20.489 | -0.134 | Independent DDPG |
| 37 | -20.360 | -21.468 | +1.108 | MADDPG |
| 41 | -18.681 | -16.258 | -2.423 | Independent DDPG |
| 47 | -21.342 | -23.134 | +1.792 | MADDPG |

Matched-seed outcome:

- MADDPG wins: 7
- Independent DDPG wins: 3
- Ties: 0

The mean paired difference was:

`MADDPG - Independent DDPG = +0.566`

with a paired-difference standard deviation of:

`1.468`

A positive difference favors MADDPG.

---

## Statistical Significance

Because both algorithms were evaluated using the same 10 random seeds, the final deterministic returns form matched pairs.

Two two-sided paired statistical tests were applied to the Episode 1,000 returns:

| Test | Statistic | p-value |
|---|---:|---:|
| Paired t-test | 1.2203 | 0.253369 |
| Wilcoxon signed-rank | 16.0000 | 0.275391 |

The mean paired difference was `+0.5664`, with a 95% confidence interval of `[-0.4836, 1.6165]`.

At `alpha = 0.05`, neither test found a statistically significant difference between MADDPG and Independent DDPG. The confidence interval also includes zero.

Therefore, although MADDPG achieved a slightly higher final mean return and won 7 of 10 matched-seed comparisons, Experiment 1 does not establish a clear performance winner between the two algorithms.

---

## Runtime

Runtime was measured separately for every complete seed run.

### MADDPG

- Mean runtime per seed: 9.89 minutes
- Runtime standard deviation: 1.25 minutes
- Minimum runtime: 6.57 minutes
- Maximum runtime: 11.33 minutes
- Total runtime across 10 seeds: 98.91 minutes

### Independent DDPG

- Mean runtime per seed: 8.21 minutes
- Runtime standard deviation: 0.70 minutes
- Minimum runtime: 7.29 minutes
- Maximum runtime: 9.16 minutes
- Total runtime across 10 seeds: 82.08 minutes

### Entire Experiment

- Total training runs: 20
- Sum of individual seed runtimes: 180.98 minutes
- Overall measured wall-clock time: 181.15 minutes
- Overall measured wall-clock time: approximately 3.02 hours

Independent DDPG was therefore faster per seed under this implementation and hardware configuration.

---

## Interpretation

Under the matched 10-seed, 1,000-episode compute budget, MADDPG achieved a slightly higher final deterministic mean return than Independent DDPG.

MADDPG:

- achieved a better final mean return,
- won 7 of 10 matched-seed comparisons,
- had lower final across-seed variability,
- and had a slightly better mean evaluation return across training.

Independent DDPG:

- won 3 of 10 matched seeds,
- occasionally achieved very strong individual-seed results,
- and trained faster on average.

The descriptive results slightly favor MADDPG, but the paired t-test and Wilcoxon signed-rank test were both non-significant at `alpha = 0.05`. Therefore, we do not conclude that MADDPG clearly outperformed Independent DDPG on Simple Spread under this training budget.

Experiment 1 is therefore treated as a successful implementation and reproducibility gate before moving to the custom cooperative-transport experiments.

---

## Generated Artifacts

The curated Experiment 1 outputs include:

- `comparison_summary.csv`
- `training_curve_summary.csv`
- `evaluation_curve_summary.csv`
- `final_checkpoint_by_seed.csv`
- `final_checkpoint_summary.csv`
- `paired_final_comparison.csv`
- `paired_final_summary.csv`
- `exp1_statistical_tests.csv`
- `overall_wall_clock.txt`
- `training_curves.png`
- `evaluation_curves.png`
- `final_checkpoint_by_seed.png`

The earlier three-seed results are superseded by this official ten-seed experiment.