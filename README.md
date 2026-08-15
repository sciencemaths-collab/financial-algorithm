# Financial Algorithms: Optimization, Valuation & Market Prediction

A quantitative-finance modeling framework combining **portfolio optimization**, **valuation under uncertainty**, and **probabilistic market-trend analysis**. The project is designed around reproducible numerical experiments, explicit constraints, uncertainty quantification, and benchmark comparison.

| Model | Purpose | Core method | Output |
|---|---|---|---|
| **Portfolio Optimization** | Allocate capital under risk and realistic constraints | Simulated Annealing + SLSQP | Portfolio weights, risk/return profile |
| **Financial Valuation & Risk** | Estimate value under uncertain future cash flows | DCF + Monte Carlo | Intrinsic-value distribution, P10/P50/P90 |
| **Trend–Area–Uncertainty Forecasting** | Measure directional price pressure and forecast uncertainty | Slope + accumulated area + probabilistic model | Direction probability and uncertainty |

---

## System architecture

The framework follows a common quantitative workflow:

**Market / fundamental data → preprocessing & estimation → model engine → constraints & uncertainty → outputs → backtesting & validation**

The three model families are distinct. Portfolio optimization solves an allocation problem, valuation estimates intrinsic value, and trend forecasting evaluates directional time-series structure.

---

## 1. Quantitative Finance & Portfolio Optimization

The base risk-adjusted objective is

$$U(w)=\mu^T w-\frac{\gamma}{2}w^T\Sigma w$$

subject to

$$\sum_i w_i=1,\qquad w_i\geq0.$$

The implemented non-convex formulation additionally accounts for transaction/turnover cost, cardinality and minimum-position effects:

$$U_{nc}(w)=U(w)-c\lVert w-w_{prev}\rVert_1-\lambda_c\Phi_{card}(w)-\lambda_m\Phi_{min}(w).$$

**Simulated Annealing** performs global exploration across discontinuous portfolio supports. **SLSQP** performs constrained continuous refinement. The hybrid retains the refined solution only when it improves the complete non-convex objective.

### Portfolio optimization benchmark

![Portfolio optimization benchmark](results/portfolio_academic.svg)

Across 30 independent controlled markets, Hybrid SA→SLSQP produced mean utility **0.11915**, with 95% bootstrap interval **[0.11710, 0.12110]**. SA alone produced **0.11910**, while SLSQP evaluated under the full non-convex objective produced **0.11601**.

---

## 2. Financial Valuation & Risk Modeling

The valuation engine uses discounted cash flow with a Gordon-growth terminal value:

$$V=\sum_{t=1}^{n}\frac{CF_t}{(1+r)^t}+\frac{CF_n(1+g)}{(r-g)(1+r)^n},\qquad r>g.$$

Instead of reporting one deterministic valuation, Monte Carlo simulation propagates uncertainty through future cash-flow growth. The resulting distribution supports downside, central and upside valuation estimates.

### Monte Carlo DCF convergence

![DCF convergence benchmark](results/dcf_academic.svg)

For the 50,000-path reference experiment:

**P10 = 1054.17 · P50 = 1376.21 · P90 = 1837.07**

The convergence experiment varies simulation count and random seed to measure stability of the median intrinsic-value estimate.

---

## 3. Unified Trend–Area–Uncertainty Forecast Model

This model combines instantaneous movement, accumulated deviation from a local baseline, and stochastic uncertainty.

### Model architecture

**Price Data → Slope → Accumulated Area Pressure → Trend Score → Uncertainty → Forecast**

The intuitive formulation begins with local slope:

$$S_t=\frac{\Delta P}{\Delta t}$$

and accumulated area relative to a moving baseline $M(t)$:

$$A_t=\int_{t-W}^{t}[P(\tau)-M(\tau)]\,d\tau.$$

These terms form a trend score:

$$T_t=w_1S_t+w_2A_t,$$

with an uncertainty-aware explanatory forecast of the form

$$P_{next}=P_{now}+T_t+\sigma_t Z,\qquad Z\sim\mathcal{N}(0,1).$$

The executable forecasting implementation uses a numerically stable feature representation based on volatility-normalized log-price slope, short and medium momentum, integrated displacement, and regularized probability estimation:

$$P(P_{t+h}>P_t\mid x_t).$$

This keeps the architecture interpretable while allowing the benchmarked implementation to operate on normalized time-series quantities.

### Trend model validation

![Trend validation benchmark](results/trend_academic.svg)

The controlled persistence experiment produced mean out-of-sample skill **+0.0170 [0.0077, 0.0259]** relative to the majority baseline. The random-walk control produced **−0.0107 [−0.0178, −0.0031]**. The null experiment is intentionally included to test whether the model invents apparent signal where none was introduced.

---

## Benchmark summary

| Experiment | Result |
|---|---:|
| Equal-weight non-convex utility | 0.07347 [0.07204, 0.07489] |
| SLSQP on complete non-convex objective | 0.11601 [0.11379, 0.11834] |
| Simulated Annealing | 0.11910 [0.11702, 0.12107] |
| **Hybrid SA→SLSQP** | **0.11915 [0.11710, 0.12110]** |
| Monte Carlo DCF, 50,000 paths | **P10 1054.17 / P50 1376.21 / P90 1837.07** |
| Trend controlled-signal skill | **+0.0170 [0.0077, 0.0259]** |
| Trend random-walk control | **−0.0107 [−0.0178, −0.0031]** |
| Integrated numerical tests | **10/10 passed** |

---

## Validation

The project evaluates numerical correctness and model behavior through:

- portfolio feasibility and fully-invested constraints;
- non-convex objective evaluation after transaction and discrete penalties;
- global-search versus local-refinement comparison;
- Monte Carlo reproducibility and quantile ordering;
- DCF discount-rate and terminal-growth validity checks;
- convergence across simulation counts and random seeds;
- probabilistic forecast bounds;
- out-of-sample trend evaluation;
- explicit random-walk negative controls;
- multi-seed experiments and bootstrap confidence intervals.

The current benchmark suite is a controlled numerical evaluation. Real-market performance evaluation should additionally use frozen historical datasets, untouched final test periods, survivorship-aware asset universes, realistic fees/slippage and rolling or expanding estimation windows.

---

## Reproduce

```bash
python -m pip install -e '.[test]'
pytest -q
MPLBACKEND=Agg python scripts/academic_benchmark.py
```

The command regenerates the benchmark figures displayed directly on this page.

Experimental methodology: [`ACADEMIC_VALIDATION.md`](ACADEMIC_VALIDATION.md)  
Detailed benchmark results: [`results/ACADEMIC_RESULTS.md`](results/ACADEMIC_RESULTS.md)

---

## Project structure

```text
financial-algorithm/
├── src/financial_algorithms/     # optimization, valuation, forecasting and metrics
├── scripts/                      # reproducible benchmark runners
├── tests/                        # numerical and behavioral tests
├── results/                      # generated figures and benchmark evidence
├── README.md
└── LICENSE
```

## License

Released under the **MIT License**. See [`LICENSE`](LICENSE).

## Disclaimer

This project is quantitative-finance research software. It is not investment advice and does not guarantee financial performance.