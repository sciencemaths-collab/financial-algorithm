# Academic Validation Protocol

## Scope
This repository contains three distinct but connected quantitative-finance components: (1) portfolio optimization, (2) financial valuation and risk modeling, and (3) probabilistic market-trend modeling. The architecture figure is a system-level map: data and preprocessing feed the models; constraints govern optimization; backtesting and validation evaluate outputs. Not every box is a separate algorithm.

## Scientific claims
The package is an experimental/research implementation, not investment advice. Synthetic experiments establish numerical correctness, optimization behavior, convergence, and controlled-signal recovery. They do **not** establish live-market profitability. Real-market claims require frozen datasets, transaction-cost assumptions, walk-forward evaluation, and comparison with investable baselines.

## Portfolio experiment
The convex benchmark compares equal weighting, simulated annealing (SA), SLSQP, and SA→SLSQP. The non-convex benchmark adds turnover costs, minimum-position effects, cardinality penalties, and sparse support changes. Thirty independently seeded markets are evaluated. The primary endpoint is true utility after all penalties. Bootstrap 95% confidence intervals are reported.

## Valuation experiment
DCF is tested for discount-rate monotonicity and terminal-growth validity. Monte Carlo DCF perturbs growth and discount-rate assumptions. Convergence is measured from 250 to 20,000 paths across 12 seeds, with a 50,000-path reference distribution. Sensitivity analysis varies discount rate and terminal growth.

## Trend experiment
The predictor uses normalized log-price slope, short and medium momentum, integrated log-price displacement, and volatility. A ridge-regularized probabilistic classifier is trained only on the training window and evaluated out of sample. Controlled persistent processes are compared with a null random-walk control. This design tests whether the model recovers a known signal without treating random noise as evidence.

## Reproducibility
Run `pytest -q` and `MPLBACKEND=Agg python scripts/academic_benchmark.py`. Fixed seeds are used. Figures are vector SVG for academic slides/posters. Numerical results are written to `results/ACADEMIC_RESULTS.md`.

## Recommended real-data extension
Use adjusted daily prices from a frozen, redistributable dataset; define the universe before evaluation; use rolling/expanding windows; include delisting/survivorship controls where possible; compare against equal-weight, buy-and-hold, minimum-variance, and simple momentum baselines; model fees/slippage; report Sharpe, drawdown, turnover, VaR/CVaR, calibration/Brier score, and bootstrap confidence intervals. Keep the final test interval untouched until model choices are frozen.