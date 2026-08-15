# Financial Algorithms: Optimization, Valuation & Market Prediction

> **A reproducible quantitative-finance research framework for portfolio optimization, valuation under uncertainty, and probabilistic trend analysis.**

This repository contains **three distinct computational models inside one quantitative-finance framework**. The surrounding data, constraints, risk analytics, and backtesting layers are system infrastructure, not separate algorithms.

| Component | Scientific question | Core method | Primary output |
|---|---|---|---|
| **Portfolio Optimization** | How should capital be allocated under risk and realistic constraints? | Simulated Annealing + SLSQP | Optimal weights / efficient frontier |
| **Valuation & Risk Modeling** | What is an asset/company worth under uncertain assumptions? | DCF + Monte Carlo + sensitivity analysis | Intrinsic-value distribution |
| **Trend–Area–Uncertainty Forecasting** | Does recent price geometry contain measurable directional information? | Slope + integrated displacement + regularized probability model | P(up), P(down), uncertainty |

---

## 1. Quantitative Finance & Portfolio Optimization

### Mathematical formulation

The smooth allocation objective is

$$U(w)=\mu^T w-\frac{\gamma}{2}w^T\Sigma w,$$

subject to

$$\sum_i w_i=1,\qquad w_i\geq0.$$

For the academic benchmark the objective is deliberately made non-convex:

$$U_{nc}(w)=U(w)-c\lVert w-w_{prev}\rVert_1-\lambda_c\Phi_{card}(w)-\lambda_m\Phi_{min}(w).$$

This introduces turnover/transaction cost, cardinality and minimum-position effects. **Simulated Annealing (SA)** explores discontinuous supports globally; **SLSQP** performs continuous constrained refinement. The hybrid accepts the refinement only when the true non-convex objective improves.

### Portfolio benchmark

![Portfolio optimization benchmark](results/portfolio_academic.svg)

The plot above is rendered directly from the reproducible benchmark, not a decorative illustration.

---

## 2. Financial Valuation & Risk Modeling

The valuation engine uses discounted cash flow with a Gordon-growth terminal value:

$$V=\sum_{t=1}^{n}\frac{CF_t}{(1+r)^t}+\frac{CF_n(1+g)}{(r-g)(1+r)^n},\qquad r>g.$$

Monte Carlo simulation perturbs future cash-flow growth to obtain a distribution rather than a single-point valuation. The implementation reports P10/P50/P90 and explicitly rejects invalid terminal assumptions where $r\leq g$.

### Monte Carlo valuation distribution and convergence

![DCF valuation benchmark](results/dcf_academic.svg)

This figure exposes the uncertainty distribution and simulation convergence used to support the reported valuation interval.

---

## 3. Unified Trend–Area–Uncertainty Forecast Model

The forecasting model is conceptually organized as:

**Price data → local slope → accumulated trajectory/area → normalized trend features → probabilistic forecast → uncertainty-aware evaluation**

For a price trajectory $P(t)$, the explanatory form is

$$S_t=\frac{\Delta P}{\Delta t},$$

$$A_t=\int_{t-W}^{t}[P(\tau)-M(\tau)]\,d\tau,$$

$$T_t=w_1S_t+w_2A_t.$$

The research implementation uses the numerically more stable log-price feature vector

$$x_t=[\text{normalized slope},\ \text{short momentum},\ \text{medium momentum},\ \text{integrated displacement}],$$

with volatility normalization and regularization before estimating

$$P(P_{t+h}>P_t\mid x_t).$$

This distinction matters: the slope/area equations explain the architecture intuitively, while the regularized implementation is what is actually benchmarked.

### Out-of-sample signal vs null-control benchmark

![Trend model validation](results/trend_academic.svg)

A valid forecasting experiment needs a negative control. The benchmark therefore compares markets containing a controlled persistence signal against random-walk/null markets. Positive performance on the synthetic signal experiment is **not presented as evidence of real-market predictability**.

---

## Reproduced academic benchmark

The academic experiment uses **30 independent synthetic markets**, bootstrap confidence intervals, an explicit forecasting null control, Monte Carlo convergence analysis, and numerical invariant tests.

| Experiment | Reproduced result |
|---|---:|
| Equal-weight non-convex utility | 0.07347 [0.07204, 0.07489] |
| SLSQP on true non-convex objective | 0.11601 [0.11379, 0.11834] |
| Simulated Annealing | 0.11910 [0.11702, 0.12107] |
| **Hybrid SA→SLSQP** | **0.11915 [0.11710, 0.12110]** |
| Monte Carlo DCF, 50,000 paths | **P10 1054.17 / P50 1376.21 / P90 1837.07** |
| Trend controlled-signal skill | **+0.0170 [0.0077, 0.0259]** |
| Trend random-walk/null skill | **−0.0107 [−0.0178, −0.0031]** |
| Local integrated academic tests | **10/10 passed** |

### Interpretation

The hybrid portfolio method performs best in the present controlled non-convex experiment, but its advantage over SA alone is small and should not be exaggerated. The DCF experiment demonstrates stable uncertainty propagation and interpretable valuation quantiles. The trend experiment recovers a deliberately injected persistence signal while failing to manufacture positive skill in the null control, which is the behavior expected from a useful diagnostic experiment.

---

## Validation design

The suite checks more than whether code executes:

- feasibility and simplex constraints for portfolio weights;
- hybrid non-degradation under the **true** discontinuous objective;
- covariance regularization and risk-metric finiteness;
- monotonic DCF behavior and invalid-terminal-condition rejection;
- deterministic Monte Carlo reproducibility under fixed seeds;
- P10 < P50 < P90 ordering;
- probabilistic forecast bounds;
- controlled-signal recovery versus explicit null markets;
- multi-seed experiments and bootstrap confidence intervals.

Full protocol: [`ACADEMIC_VALIDATION.md`](ACADEMIC_VALIDATION.md)  
Detailed evidence: [`results/ACADEMIC_RESULTS.md`](results/ACADEMIC_RESULTS.md)

---

## Reproduce everything

```bash
python -m pip install -e '.[test]'
pytest -q
MPLBACKEND=Agg python scripts/academic_benchmark.py
```

The benchmark regenerates the **same figures displayed on this page**:

![Portfolio optimization](results/portfolio_academic.svg)

![DCF uncertainty](results/dcf_academic.svg)

![Trend validation](results/trend_academic.svg)

---

## What is required before a real-market academic claim?

The current benchmark establishes numerical correctness and controlled synthetic behavior. A conference or paper claim about empirical market performance additionally requires a frozen historical dataset, untouched final test interval, survivorship-aware universe construction, realistic transaction fees/slippage, rolling or expanding training windows, and comparisons against equal-weight, buy-and-hold, minimum-variance and simple momentum baselines.

## Scope

Research and educational software. **Not investment advice.** No guarantee of financial performance.