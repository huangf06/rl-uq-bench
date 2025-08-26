# Project: UQ Benchmark in Deep Reinforcement Learning

This project aims to benchmark multiple uncertainty quantification (UQ) methods in deep RL agents (e.g., DQN, Bootstrapped DQN, QR-DQN, MC Dropout) across multiple noise settings and random seeds.

The goal is to evaluate each method's calibration, robustness, and distributional quality using a consistent experimental framework.

## Claude Behavior Guidelines

- You may infer the folder and file structure from the project directly; it is not listed here.
- Assist with code editing, analysis, and report writing as needed.
- When writing code, focus on minimal, clean diffs or full replacements as appropriate.
- When writing evaluation summaries, use scientific writing appropriate for a master's thesis.

## Metric Focus

The main evaluation metrics are:
- ECE (Expected Calibration Error) / ACE
- CRPS (Continuous Ranked Probability Score)
- Coverage
- WIS (Weighted Interval Score)

## Known Constraints

- Experiments involve 4 UQ methods × 5 noise levels × 10 seeds.
- Evaluation and plotting are conducted post-hoc from saved models.
- Metrics may be affected by reward clipping or noise injection.
