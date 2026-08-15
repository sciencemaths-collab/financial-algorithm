# Financial Algorithms: Optimization, Valuation & Market Prediction

> **A reproducible quantitative-finance research framework for portfolio optimization, valuation under uncertainty, and probabilistic trend analysis.**

This repository contains **three distinct computational models inside one quantitative-finance framework**.

| Component | Scientific question | Core method | Primary output |
|---|---|---|---|
| **Portfolio Optimization** | How should capital be allocated under risk and realistic constraints? | Simulated Annealing + SLSQP | Optimal weights / efficient frontier |
| **Valuation & Risk Modeling** | What is an asset/company worth under uncertain assumptions? | DCF + Monte Carlo | Intrinsic-value distribution |
| **Trend–Area–Uncertainty Forecasting** | Does recent price geometry contain measurable directional information? | Slope + integrated displacement + regularized probability model | P(up), P(down), uncertainty |

## 1. Quantitative Finance & Portfolio Optimization

The smooth objective is

$$U(w)=\mu^T w-\frac{\gamma}{2}w^T\Sigma w$$

subject to $\sum_iw_i=1$ and $w_i\geq0$. The research benchmark additionally includes turnover, transaction-cost, cardinality and minimum-position penalties, producing a genuinely non-convex allocation problem. Simulated Annealing performs global support exploration and SLSQP performs constrained local refinement.

### Benchmark: non-convex portfolio optimization

![Portfolio optimization benchmark](results/portfolio_academic.svg)

**Result:** Hybrid SA→SLSQP mean utility **0.11915**, 95% bootstrap CI **[0.11710, 0.12110]**, across 30 independent synthetic markets.

## 2. Financial Valuation & Risk Modeling

$$V=\sum_{t=1}^{n}\frac{CF_t}{(1+r)^t}+\frac{CF_n(1+g)}{(r-g)(1+r)^n},\qquad r>g$$

Monte Carlo simulation propagates uncertainty through future cash flows and reports valuation intervals rather than a single deterministic number.

### Benchmark: Monte Carlo DCF convergence

![DCF convergence benchmark](results/dcf_academic.svg)

**50,000-path reference:** P10 **1054.17**, P50 **1376.21**, P90 **1837.07**.

## 3. Unified Trend–Area–Uncertainty Forecast Model

Conceptually:

**Price Data → Slope → Accumulated Area → Trend Score → Probability + Uncertainty**

$$S_t=\frac{\Delta P}{\Delta t},\qquad A_t=\int_{t-W}^{t}[P(\tau)-M(\tau)]d\tau,\qquad T_t=w_1S_t+w_2A_t$$

The benchmarked implementation uses volatility-normalized log-price slope, short/medium momentum, integrated displacement and regularized probabilistic estimation of $P(P_{t+h}>P_t)$.

### Benchmark: controlled signal vs null market

![Trend validation benchmark](results/trend_academic.svg)

Controlled persistence skill: **+0.0170 [0.0077, 0.0259]**. Random-walk/null skill: **−0.0107 [−0.0178, −0.0031]**. This is controlled synthetic signal recovery, not a claim of live-market predictability.

## Reproduced results

| Experiment | Result |
|---|---:|
| Equal-weight non-convex utility | 0.07347 [0.07204, 0.07489] |
| SLSQP on true non-convex objective | 0.11601 [0.11379, 0.11834] |
| Simulated Annealing | 0.11910 [0.11702, 0.12107] |
| **Hybrid SA→SLSQP** | **0.11915 [0.11710, 0.12110]** |
| Monte Carlo DCF | **P10 1054.17 / P50 1376.21 / P90 1837.07** |
| Trend controlled-signal skill | **+0.0170 [0.0077, 0.0259]** |
| Trend random-walk/null skill | **−0.0107 [−0.0178, −0.0031]** |
| Integrated academic tests | **10/10 passed** |

## Validation and reproducibility

The validation suite checks feasibility, non-degradation under the true discontinuous portfolio objective, DCF monotonicity, invalid terminal assumptions, Monte Carlo reproducibility and quantile ordering, probability bounds, controlled-signal recovery, explicit null markets, multi-seed behavior and bootstrap confidence intervals.

```bash
python -m pip install -e '.[test]'
pytest -q
MPLBACKEND=Agg python scripts/academic_benchmark.py
```

Full experimental protocol: [`ACADEMIC_VALIDATION.md`](ACADEMIC_VALIDATION.md)  
Reproduced evidence: [`results/ACADEMIC_RESULTS.md`](results/ACADEMIC_RESULTS.md)

## Scientific scope

The current evidence establishes numerical correctness and controlled synthetic behavior. Claims about empirical market performance require a frozen historical dataset, untouched final test interval, survivorship-aware universe construction, realistic fees/slippage, rolling or expanding training windows, and benchmark comparisons.

**Research and educational software. Not investment advice.**