from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def portfolio_stats(w, mu, cov):
    w = np.asarray(w, float); mu = np.asarray(mu, float); cov = np.asarray(cov, float)
    return float(w @ mu), float(np.sqrt(max(w @ cov @ w, 0)))


def utility(w, mu, cov, risk_aversion=3.0):
    r, v = portfolio_stats(w, mu, cov)
    return r - 0.5 * risk_aversion * v * v


def slsqp_portfolio(mu, cov, risk_aversion=3.0, x0=None):
    n = len(mu); x0 = np.ones(n) / n if x0 is None else np.asarray(x0, float)
    res = minimize(lambda w: -utility(w, mu, cov, risk_aversion), x0, method="SLSQP",
                   bounds=[(0, 1)] * n, constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                   options={"ftol": 1e-12, "maxiter": 1000})
    if not res.success: raise RuntimeError(res.message)
    return res.x


def sa_portfolio(mu, cov, risk_aversion=3.0, steps=5000, seed=7):
    rng = np.random.default_rng(seed); n = len(mu); w = np.ones(n) / n
    score = utility(w, mu, cov, risk_aversion); best = w.copy(); bests = score
    for k in range(steps):
        t = max(1e-5, 0.05 * (1 - k / steps))
        z = np.clip(w + rng.normal(0, 0.08, n), 0, None); z = z / z.sum() if z.sum() else np.ones(n) / n
        s = utility(z, mu, cov, risk_aversion)
        if s > score or rng.random() < np.exp((s - score) / t): w, score = z, s
        if score > bests: best, bests = w.copy(), score
    return best


def hybrid_portfolio(mu, cov, risk_aversion=3.0, steps=5000, seed=7):
    return slsqp_portfolio(mu, cov, risk_aversion, sa_portfolio(mu, cov, risk_aversion, steps, seed))


def dcf_value(cashflows, discount_rate, terminal_growth=0.02):
    cf = np.asarray(cashflows, float); n = len(cf)
    pv = sum(cf[t] / (1 + discount_rate) ** (t + 1) for t in range(n))
    tv = cf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    return float(pv + tv / (1 + discount_rate) ** n)


def monte_carlo_dcf(base_cf, discount_rate=0.10, growth_mu=0.04, growth_sigma=0.08, n_sims=10000, seed=7):
    rng = np.random.default_rng(seed); base = np.asarray(base_cf, float); vals = []
    for _ in range(n_sims):
        shocks = rng.normal(growth_mu, growth_sigma, len(base)); cf = np.empty_like(base); cf[0] = base[0] * (1 + shocks[0])
        for i in range(1, len(base)): cf[i] = cf[i - 1] * (1 + shocks[i])
        vals.append(dcf_value(cf, discount_rate))
    a = np.asarray(vals)
    return {"values": a, "p10": float(np.quantile(a, .1)), "p50": float(np.quantile(a, .5)), "p90": float(np.quantile(a, .9))}


def trend_signal(prices, window=20, alpha=1.0, beta=0.15):
    p = np.asarray(prices, float); y = p[-window:]; x = np.arange(window)
    slope = np.polyfit(x, y, 1)[0]; area = np.trapezoid(np.diff(y), dx=1.0)
    return float(alpha * slope + beta * area / window)


def forecast_probability(prices, window=20, horizon=5):
    p = np.asarray(prices, float); ret = np.diff(np.log(p)); sigma = float(np.std(ret[-max(window, 2):], ddof=1))
    sig = trend_signal(p, window) / p[-1]; z = sig * np.sqrt(horizon) / (sigma + 1e-12)
    return float(norm.cdf(z))
