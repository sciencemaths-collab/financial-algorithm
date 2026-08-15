# Reproducible benchmark results

Synthetic validation only. These experiments establish numerical behavior and controlled-signal recovery; they do not establish live-market profitability.

## Portfolio optimization

| Method | Mean utility | 95% bootstrap CI |
|---|---:|---:|
| Equal | 0.07347 | [0.07204, 0.07489] |
| SLSQP | 0.11601 | [0.11379, 0.11834] |
| SA | 0.11910 | [0.11702, 0.12107] |
| Hybrid SA→SLSQP | **0.11915** | **[0.11710, 0.12110]** |

Interpretation: once turnover/discrete penalties are introduced, global search improves the true objective relative to solving only the smooth convex subproblem. The hybrid refinement is slightly better than SA alone in this benchmark, but the difference is small and should not be overstated.

## Monte Carlo DCF

50,000-path valuation distribution: **P10 = 1054.17, P50 = 1376.21, P90 = 1837.07**.

The benchmark also measures convergence from 250 to 20,000 simulation paths across 12 independent seeds.

## Market-trend model

Controlled persistent-process mean out-of-sample skill (accuracy minus majority baseline): **+0.0170**, 95% bootstrap CI **[+0.0077, +0.0259]**.

Null random-walk control mean skill: **−0.0107**, 95% bootstrap CI **[−0.0178, −0.0031]**.

Interpretation: after replacing the earlier uncalibrated trend equation with a regularized probabilistic model, the controlled experiment shows modest recovery of an injected momentum/persistence signal while the null control does not produce positive skill. This is a synthetic signal-recovery result, not evidence of stock-market predictability.

## Local verification

The academic core test suite was executed locally on 2026-08-15: **10 passed**. The benchmark was then executed end-to-end to produce the numbers above.