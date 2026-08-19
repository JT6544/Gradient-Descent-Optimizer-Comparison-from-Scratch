# Source notes

This repository rebuilds concepts demonstrated in three earlier educational scripts:

- `py-01-gradient-descent.py`
- `py-04-ex3.14-normalised.py`
- `py-06-adam_rmsprop_improved.py`

The original uploaded files are not modified or copied into this repository.

## Preserved ideas

- comparison of learning dynamics on one-dimensional polynomial objectives;
- standard and normalised gradient updates;
- RMSProp and Adam-style adaptive updates;
- plots of weight and objective histories.

## Rebuild changes

- separates objectives, optimizers, experiment orchestration, plotting, and tests;
- supports scalar and multidimensional objectives through one array-based interface;
- implements global L2-normalised gradient descent explicitly;
- implements standard RMSProp with an exponential second-moment accumulator;
- adds Adam first- and second-moment bias correction;
- adds numerical guards and configuration validation;
- replaces top-level notebook-style execution with a command-line entry point;
- records machine-readable summaries and complete convergence histories;
- adds analytical-gradient finite-difference tests and exact update tests;
- adds deterministic two-dimensional quadratic and Rosenbrock benchmarks;
- removes the Autograd and external LaTeX runtime dependencies.

## Uploaded-source checksums

```text
59fee304f1f181a0d5c18b80b05e239a616c26fee70981019b908a3d5c376cee  py-06-adam_rmsprop_improved.py
0821e9da7274616cb6b8c78f9cab1e2b24330bc1740836ab1fc79dc5d577f0a7  py-01-gradient-descent.py
b8772dc926d713abd29be5ac4e8dd170357dfae20655347a80c215a020df3048  py-04-ex3.14-normalised.py
```
