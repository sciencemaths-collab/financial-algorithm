# Financial Algorithms

A reproducible research implementation of three distinct finance algorithms:

1. **Portfolio Optimization** — Simulated Annealing (global search), SLSQP (constrained local optimization), and a hybrid SA→SLSQP method.
2. **Financial Valuation & Risk Modeling** — discounted cash flow (DCF) with Monte Carlo uncertainty and P10/P50/P90 valuation intervals.
3. **Market Trend Prediction** — slope + integrated trend + stochastic uncertainty, evaluated honestly with walk-forward testing.

## Mathematical core

Portfolio utility: `U(w) = μᵀw − (γ/2) wᵀΣw`, subject to `Σwᵢ = 1` and `wᵢ ≥ 0`.

DCF: `V = Σ CF_t/(1+r)^t + TV/(1+r)^n`.

Trend score: `μ_t = α slope(P) + β mean-integrated ΔP`, converted to `P(up)` using recent realized volatility.

## Reproduce

```bash
python -m pip install -e '.[test]'
pytest -q
python scripts/benchmark.py
```

## Benchmark result

The benchmark is deliberately diagnostic, not promotional.

| Module | Result | Interpretation |
|---|---:|---|
| Unit tests | 5/5 pass | Core invariants and numerical behavior pass |
| Equal-weight portfolio utility | 0.096857 | Baseline |
| SA utility | 0.136759 | Reaches the constrained optimum on the synthetic case |
| SLSQP utility | 0.136759 | Same optimum, faster for this convex formulation |
| Hybrid SA→SLSQP utility | 0.136759 | No measurable advantage over SLSQP on this convex case |
| Monte Carlo DCF P10/P50/P90 | 1124 / 1372 / 1665 | Ordered, interpretable valuation uncertainty |
| Trend walk-forward accuracy | 50.1% | Below 56.3% majority-class baseline on this seeded synthetic series |
| Trend Brier score | 0.289 | Probability calibration is not competitive in this test |

### Scientific conclusion

The three algorithms are **not the same algorithm**. Portfolio optimization solves a constrained allocation problem; valuation estimates intrinsic value under uncertain cash flows; trend prediction attempts probabilistic forecasting.

The benchmark also prevents overclaiming. In the convex portfolio test, SLSQP alone already finds the same solution as SA and the hybrid, so the hybrid is not justified by this test. SA becomes more relevant when objectives are non-convex, discontinuous, or contain realistic transaction/tax/cardinality constraints. The trend model does **not** beat its simple baseline in the included walk-forward experiment and should be treated as an experimental hypothesis requiring feature/model refinement, not as a validated predictor.

## Figures

Running `python scripts/benchmark.py` generates publication-style SVG figures in `results/`:

- `portfolio_benchmark.svg` — feasible risk-return cloud and optimizer comparison
- `dcf_distribution.svg` — Monte Carlo intrinsic-value distribution with P10/P50/P90
- `trend_walkforward.svg` — out-of-sample probability trace and decision threshold

## Scope

Educational/research software. It is not investment advice and does not guarantee financial performance.
