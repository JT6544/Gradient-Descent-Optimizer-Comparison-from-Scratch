# Gradient-Descent Optimizer Comparison — Gate 3 Improvement Record

This document records the changes made when rebuilding the three original optimizer demonstrations into the repository distributed as `Gradient-Descent-Optimizer-Comparison-from-Scratch-Gate3.zip`. The three Python files beside this document are untouched copies of the starting sources.

## Project objective

The original scripts illustrated gradient descent, gradient normalisation, RMSProp, and an Adam-like update on one-dimensional quartic functions. The rebuild preserves those examples while creating a reusable, from-scratch optimization laboratory that compares standard algorithms across objectives with different geometry.

## Summary of improvements

| Area | Original implementation | Rebuilt repository | Why the change matters | Impact |
|---|---|---|---|---|
| Structure | Three independent, top-level plotting scripts | Importable `optimization_lab` package plus one CLI | Removes duplicated experiment logic and makes the algorithms reusable | Optimizers and objectives can be tested or imported independently |
| Derivatives | Relied on Autograd in demonstration scripts | Defines analytical gradients for every benchmark | Keeps the core comparison genuinely from scratch | No automatic-differentiation framework is required |
| Adam | Used first-gradient moment initialisation and omitted bias correction | Uses zero-initialised moments and standard bias correction | Early Adam steps otherwise do not implement the published algorithm | Adam results now correspond to the conventional update rule |
| RMSProp | Initialised the accumulator from the first gradient | Starts the squared-gradient accumulator at zero | Makes initialization explicit and consistent with the standard algorithm | Early updates are predictable and independently testable |
| Normalised descent | Applied an elementwise sign-like rule in the scalar example | Defines global L2-normalised gradient descent | Extends the method unambiguously to vectors | Step direction has unit Euclidean norm when the gradient is nonzero |
| Benchmark scope | Used only a quartic and a tilted quartic | Adds an ill-conditioned quadratic and Rosenbrock's valley | One-dimensional examples cannot reveal conditioning or curved-valley behaviour | The comparison exposes materially different optimizer strengths and failures |
| Fairness | Informal per-script settings and plotting windows | Uses deterministic starts, declared per-objective learning rates, common stopping criteria, and a 5,000-step budget | A comparison needs explicit controls and reproducible limits | Each result can be traced to a complete experiment specification |
| Safety | Minimal parameter checking | Validates shapes, finite values, hyperparameter ranges, and objective/gradient compatibility | Invalid experiments should fail clearly rather than produce misleading traces | API errors are detected early with actionable messages |
| Evidence | Interactive figures and printed values only | Saves trajectories, losses, gradients, convergence flags, JSON/CSV summaries, and publication figures | Numerical claims should survive beyond one process | The complete result set is inspectable and regenerable |
| Verification | No tests or automation | Adds eight unit tests and continuous integration | Optimizer equations and experiment bookkeeping are regression-sensitive | Updates, gradients, stopping rules, and deterministic outputs are checked |

## Algorithms implemented

Let $f:\mathbb{R}^d\rightarrow\mathbb{R}$ and $g_t=\nabla f(x_t)$.

### Gradient descent

$$
x_{t+1}=x_t-\eta g_t.
$$

### L2-normalised gradient descent

$$
x_{t+1}=
\begin{cases}
x_t-\eta\dfrac{g_t}{\lVert g_t\rVert_2}, & \lVert g_t\rVert_2>0,\\[6pt]
x_t, & \lVert g_t\rVert_2=0.
\end{cases}
$$

Unlike an elementwise sign update, this gives the whole vector step a controlled Euclidean length $\eta$.

### RMSProp

$$
v_t=\rho v_{t-1}+(1-\rho)g_t\odot g_t,
$$

$$
x_{t+1}=x_t-\eta\frac{g_t}{\sqrt{v_t}+\varepsilon}.
$$

All operations in the denominator are elementwise.

### Adam

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t\odot g_t,
$$

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t},
$$

$$
x_{t+1}=x_t-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\varepsilon}.
$$

The bias-corrected moments are the most important mathematical change from the original Adam-like script.

## Benchmark expansion

The rebuild evaluates four objectives:

1. Quartic:

   $$
   f(x)=x^4.
   $$

2. Tilted quartic:

   $$
   f(x)=x^4+0.5x.
   $$

3. Ill-conditioned quadratic:

   $$
   f(x)=\frac{1}{2}x^\mathsf{T}Ax,
   $$

   where the eigenvalues of $A$ differ substantially.

4. Rosenbrock function:

   $$
   f(x,y)=(1-x)^2+100\left(y-x^2\right)^2.
   $$

The added multidimensional cases show behaviour hidden by the original scalar demonstrations: slow progress along poorly conditioned axes and difficulty following a narrow curved valley.

## Convergence definition

Each run has a maximum budget of 5,000 update steps and a declared gradient-norm tolerance. A run is marked converged when

$$
\lVert\nabla f(x_t)\rVert_2\leq 10^{-6}.
$$

The repository also records objective gap, distance to a known minimizer, final gradient norm, and the full trajectory. Per-objective learning rates are stated in the generated summary; consequently, the tables compare documented configurations and must not be interpreted as a universal ranking under one common learning rate.

## Measured impact

The default deterministic experiment produced the following compact summary. “No” means the common 5,000-step budget ended before the gradient tolerance was met.

| Objective | Optimizer | Steps | Final objective gap | Converged |
|---|---|---:|---:|:---:|
| Quartic | Gradient descent | 5,000 | $6.223\times10^{-6}$ | No |
| Quartic | Normalised GD | 150 | $2.0\times10^{-60}$ | Yes |
| Quartic | RMSProp | 318 | $1.561\times10^{-9}$ | Yes |
| Quartic | Adam | 31 | $5.212\times10^{-11}$ | Yes |
| Tilted quartic | Gradient descent | 34 | $1.144\times10^{-12}$ | Yes |
| Tilted quartic | Normalised GD | 5,000 | $5.56\times10^{-6}$ | No |
| Tilted quartic | RMSProp | 205 | $3.46\times10^{-13}$ | Yes |
| Tilted quartic | Adam | 266 | $1.142\times10^{-12}$ | Yes |
| Ill-conditioned quadratic | Gradient descent | 1,513 | $4.957\times10^{-13}$ | Yes |
| Ill-conditioned quadratic | Normalised GD | 5,000 | $3.125\times10^{-2}$ | No |
| Ill-conditioned quadratic | RMSProp | 125 | $8.139\times10^{-14}$ | Yes |
| Ill-conditioned quadratic | Adam | 330 | $8.987\times10^{-15}$ | Yes |
| Rosenbrock | Gradient descent | 5,000 | $4.951\times10^{-3}$ | No |
| Rosenbrock | Normalised GD | 5,000 | $1.532\times10^{-3}$ | No |
| Rosenbrock | RMSProp | 5,000 | $5.648\times10^{-4}$ | No |
| Rosenbrock | Adam | 4,276 | $9.695\times10^{-13}$ | Yes |

These results demonstrate impact rather than a universal winner:

- Adam is the only default configuration that reaches the stated tolerance on all four objectives.
- Normalised descent reaches the quartic minimum rapidly but stalls or oscillates near some minima because its step length does not naturally shrink with the gradient.
- Its exact-looking quartic result is configuration-specific: the initial position and step size align unusually well, so it should not be generalized.
- The ill-conditioned and Rosenbrock cases reveal problems that the original one-dimensional figures could not expose.

## Repository and publication improvements

The rebuilt repository adds:

- reusable optimizer, objective, experiment, and plotting modules;
- `optimizer_comparison.py` as a deterministic command-line entry point;
- source-provenance notes linking the three demonstrations to the rebuild;
- JSON and CSV result summaries plus three generated figures;
- tests for objective gradients, optimizer updates, validation, convergence, and reproducibility;
- dependency metadata, ignore rules, and a GitHub Actions workflow.

## Gate 3 documentation improvements

Gate 3 retains the complete verified Gate 2 implementation and adds a publication-quality `README.md`. The README defines all four update rules with rendered equations, describes every objective and stopping rule, states the per-objective hyperparameters, explains how to regenerate the tables and figures, and interprets the default results without claiming a universal winner. It also calls out configuration-specific behaviour such as the unusually exact normalised-descent path on the quartic and the difficulty of the Rosenbrock valley.

This makes the repository independently understandable, reproducible, and ready for review on GitHub. Gate 3 changes documentation only; optimizer code, tests, experiment settings, and recorded numerical results are identical to Gate 2.

## Files represented by this folder

- `py-01-gradient-descent.py` — untouched original gradient-descent demonstration.
- `py-04-ex3.14-normalised.py` — untouched original normalised-gradient demonstration.
- `py-06-adam_rmsprop_improved.py` — untouched original RMSProp/Adam demonstration.
- `IMPROVEMENTS.md` — this improvement and impact record.

The original scripts are included for comparison and provenance. Run the rebuilt repository from its own archive; this folder is documentation support, not a replacement distribution.
