# Financial Algorithms: Optimization, Valuation & Market Prediction

A reproducible quantitative-finance research suite containing **three distinct algorithms inside one system architecture**.

1. **Quantitative Finance & Portfolio Optimization**: constrained allocation using Simulated Annealing (global search), SLSQP (local constrained refinement), and a hybrid SA→SLSQP method.
2. **Financial Valuation & Risk Modeling**: DCF valuation, Monte Carlo uncertainty, sensitivity analysis, and P10/P50/P90 value ranges.
3. **Market Trend Prediction**: a regularized probabilistic model using normalized log-price slope, short/medium momentum, integrated trajectory displacement, and volatility.

The presentation architecture is therefore a system map, not a claim that every box is a different algorithm. Data ingestion/cleaning and validation feed the models; constraints define feasible portfolios; backtesting and statistical validation evaluate outputs.

## Mathematical core

Portfolio objective:

`U(w) = μᵀw − (γ/2) wᵀΣw − transaction/discrete penalties`

subject to full investment and long-only constraints. The academic benchmark adds turnover, cardinality, minimum-position and sparse-support effects, making the objective genuinely non-convex.

DCF:

`V = Σ CF_t/(1+r)^t + TV/(1+r)^n`, with `TV = CF_n(1+g)/(r-g)`.

Trend model:

`x_t = [normalized slope, short momentum, medium momentum, integrated displacement]`, followed by regularized probabilistic classification for `P(P_{t+h} > P_t)`.

## Verified benchmark

The current academic benchmark was executed locally with **30 independent synthetic markets**, bootstrap confidence intervals, an explicit null control for trend prediction, DCF convergence testing, and unit tests.

| Experiment | Reproduced result |
|---|---|
| Equal-weight non-convex utility | 0.07347 [0.07204, 0.07489] |
| SLSQP utility evaluated on true non-convex objective | 0.11601 [0.11379, 0.11834] |
| Simulated Annealing | 0.11910 [0.11702, 0.12107] |
| Hybrid SA→SLSQP | **0.11915 [0.11710, 0.12110]** |
| Monte Carlo DCF, 50,000 paths | **P10 1054.17 / P50 1376.21 / P90 1837.07** |
| Trend controlled-signal skill | **+0.0170 [0.0077, 0.0259]** |
| Trend null random-walk skill | **−0.0107 [−0.0178, −0.0031]** |
| Local academic test suite | **10/10 passed** |

These results are intentionally conservative. The hybrid optimizer shows only a small advantage over SA alone. The trend result demonstrates recovery of an injected synthetic persistence signal, **not evidence of live-market predictability**.

## Reproduce

```bash
python -m pip install -e '.[test]'
pytest -q
MPLBACKEND=Agg python scripts/academic_benchmark.py
```

The benchmark generates vector SVG figures suitable for academic slides/posters:

- `results/portfolio_academic.svg`
- `results/dcf_academic.svg`
- `results/trend_academic.svg`

See `ACADEMIC_VALIDATION.md` for experimental design and `results/ACADEMIC_RESULTS.md` for the reproduced evidence table.

## Academic use

For a paper or formal conference claim, the next evidence layer should use a frozen historical dataset with an untouched final test interval, survivorship-aware universe construction, realistic fees/slippage, rolling or expanding training windows, and comparisons against equal-weight, buy-and-hold, minimum-variance and simple momentum baselines.

## Scope

Research and educational software. Not investment advice. No guarantee of financial performance.