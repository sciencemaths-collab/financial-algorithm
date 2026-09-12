# Financial Decision Engine: Portfolio, Valuation and Scenario Analysis

A deterministic financial decision-support engine combining **portfolio optimization**, **valuation
under uncertainty**, and **scenario-based market assessment**. Version 0.2.0 is the reference
implementation for RAD capability `rad.decision.financial` through adapter contract 1.0.0.

Its formal qualification covers 25 packaged deterministic contract and benchmark cases. It
produces reviewable recommendations only. It cannot fetch live market data, place orders, access a
broker, authorize a transaction, provide investment advice, or guarantee financial performance.

| Model | Purpose | Core method | Output |
|---|---|---|---|
| **Portfolio Optimization** | Allocate capital under risk and realistic constraints | Simulated Annealing + SLSQP | Portfolio weights, risk/return profile |
| **Financial Valuation & Risk** | Estimate value under uncertain future cash flows | DCF + Monte Carlo | Intrinsic-value distribution, P10/P50/P90 |
| **Trend–Area–Uncertainty Scenarios** | Describe direction, acceleration, accumulated pressure and modeled uncertainty | Slope + curvature + area + stochastic simulation | Scenario distribution and signal decomposition |

## 1. Quantitative Finance & Portfolio Optimization

$$U(w)=\mu^T w-\frac{\gamma}{2}w^T\Sigma w$$

subject to $\sum_iw_i=1$ and $w_i\geq0$. The non-convex implementation additionally accounts for turnover, transaction cost, cardinality and minimum-position effects. Simulated Annealing explores portfolio supports globally; SLSQP performs constrained local refinement.

**Benchmark result:** Hybrid SA→SLSQP mean utility **0.11915**, 95% bootstrap interval **[0.11710, 0.12110]**, across 30 controlled markets. SA alone produced **0.11910** and SLSQP evaluated under the complete non-convex objective produced **0.11601**.

## 2. Financial Valuation & Risk Modeling

$$V=\sum_{t=1}^{n}\frac{CF_t}{(1+r)^t}+\frac{CF_n(1+g)}{(r-g)(1+r)^n},\qquad r>g.$$

Monte Carlo simulation propagates uncertainty through future cash flows and produces a valuation distribution instead of a single deterministic estimate.

**50,000-path reference:** P10 **1054.17**, P50 **1376.21**, P90 **1837.07**.

## 3. Unified Trend–Area–Uncertainty Scenario Model

The unified model separates an operator-supplied price series into **direction**, **acceleration**,
**accumulated displacement from baseline**, and **modeled uncertainty**. Outputs are conditional
scenarios from the supplied snapshot and assumptions, not predictions validated on live markets.

### Architecture

![Unified Trend–Area–Uncertainty architecture](./results/unified_architecture.jpg)

$$S_t=\frac{\Delta P}{\Delta t},\qquad C_t=\frac{dS}{dt}=\frac{d^2P}{dt^2},\qquad A_t=\int(P-M)\,dt.$$

The standardized signals are combined as

$$T_t=0.55Z(S_t)+0.20Z(C_t)+0.25Z(A_t).$$

The stochastic forecast is

$$P_{t+1}=P_t\exp\left[(\mu_t-\tfrac12\sigma^2)+\sigma Z_t\right],\qquad Z_t\sim N(0,1),$$

with $\mu_t=\mu_0+\lambda T_t$.

### Demonstration configuration

| Quantity | Value |
|---|---:|
| Historical observations | **150 sessions** |
| Forecast horizon | **45 sessions** |
| Monte Carlo paths | **2,500** |
| Baseline daily log-return $\mu_0$ | **0.0019567** |
| Trend-adjusted forecast drift $\mu_t$ | **0.0032654** |
| Estimated daily volatility $\sigma$ | **0.0100729** |
| Final unified trend score $T$ | **1.63580** |
| Signal weights | **0.55 slope / 0.20 curvature / 0.25 area** |

### Scenario projection with uncertainty

![Unified forecast with uncertainty](./results/unified_forecast.jpg)

The demonstration trajectory and smoothed trend are followed by a **45-session Monte Carlo
projection**. The central path is the median of 2,500 simulations; the widening interval shows
modeled uncertainty under the demonstration assumptions. It is not a calibrated live-market price
target.

### Signal decomposition

![Unified signal components](./results/unified_components.jpg)

Slope measures current direction, curvature measures whether movement is strengthening or weakening, and area pressure captures accumulated signed displacement relative to the moving baseline. Standardization makes the three signals comparable before weighting.

### Calculation flow

1. Smooth the observed price trajectory and estimate $M(t)$.
2. Compute $S_t=\Delta P/\Delta t$.
3. Compute $C_t=dS/dt$.
4. Integrate signed deviation from baseline to obtain $A_t$.
5. Standardize $S$, $C$ and $A$.
6. Form $T=0.55Z(S)+0.20Z(C)+0.25Z(A)$.
7. Estimate realized log-return volatility $\sigma$.
8. Shift baseline drift through $\mu_t=\mu_0+\lambda T$.
9. Generate 2,500 stochastic price paths.
10. Summarize the distribution with central forecast and uncertainty intervals.

### Controlled signal test

On the packaged synthetic persistence fixture, the measured score was **+0.0170 [0.0077,
0.0259]**; on its random-walk control it was **−0.0107 [−0.0178, −0.0031]**. These fixture results
test implementation behavior and do not establish out-of-sample market forecasting skill.

## Benchmark summary

All figures below come from packaged deterministic or seeded synthetic demonstrations. They are
reproducibility evidence for the software, not historical backtests, live-market validation,
profitability evidence, or expected returns.

| Experiment | Result |
|---|---:|
| Equal-weight non-convex utility | 0.07347 [0.07204, 0.07489] |
| SLSQP on complete non-convex objective | 0.11601 [0.11379, 0.11834] |
| Simulated Annealing | 0.11910 [0.11702, 0.12107] |
| **Hybrid SA→SLSQP** | **0.11915 [0.11710, 0.12110]** |
| Monte Carlo DCF, 50,000 paths | **P10 1054.17 / P50 1376.21 / P90 1837.07** |
| Unified-model controlled-signal skill | **+0.0170 [0.0077, 0.0259]** |
| Random-walk control | **−0.0107 [−0.0178, −0.0031]** |
| Automated tests | See the current CI run for the authoritative count |
| Formal decision-support qualification | **25/25 passed** |

Qualification status: `QUALIFIED_FOR_DECISION_SUPPORT_BENCHMARKS`. The qualified boundary excludes
live feeds, broker connectivity, transaction execution, suitability decisions, autonomous action,
and future-performance claims.

## Install and run

```bash
python -m pip install 'financial-algorithms @ git+https://github.com/sciencemaths-collab/financial-algorithm.git@e12ffbe52dd7f5a4ad32d906c67b08870b2629da'
python - <<'PY'
from financial_algorithms import decide

result = decide({
    "operation": "valuation.estimate",
    "cashflows": [100, 105, 110, 115, 120],
    "discount_rate": 0.10,
    "simulations": 1000,
    "seed": 7,
})
print(result["decision"])
PY
```

Version 0.2.0 is not currently published to PyPI. The commit-pinned command above is the
reproducible public installation path; it prevents an unrelated package or moving branch from
being installed under the same name.

For a source checkout, use `uv sync --extra test`, then run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
uv run pip-audit --progress-spinner=off
uv build
uv run python scripts/validate_contracts.py
uv run python -m financial_algorithms.qualification --output artifacts/qualification.json
uv run python scripts/clean_wheel_acceptance.py
```

See [`docs/ENGINE_CONTRACT.md`](docs/ENGINE_CONTRACT.md),
[`docs/QUALIFICATION.md`](docs/QUALIFICATION.md), and
[`docs/SECURITY.md`](docs/SECURITY.md) for the supported boundary and evidence claims.

## License

Released under the **MIT License**. See [`LICENSE`](LICENSE).

## Disclaimer

Quantitative-finance decision-support software. Not investment advice. No guarantee of financial performance. Independent professional review, current data validation, suitability assessment, and explicit human authorization remain necessary before any external action.
