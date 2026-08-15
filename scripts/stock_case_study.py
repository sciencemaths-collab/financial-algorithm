"""Generate presentation-quality stock-style examples for the Unified Trend-Area-Uncertainty model.

The default series is synthetic and reproducible so the figure is not presented as
historical market evidence. Replace `synthetic_market` with a frozen historical price
array to run the same calculations on empirical data.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path("results")
OUT.mkdir(exist_ok=True)


def synthetic_market(seed=21, n=260):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    eps = rng.normal(0, 0.0105, n)
    for t in range(1, n):
        drift = 0.00045 + 0.10 * r[t-1]
        r[t] = drift + eps[t]
    return 100 * np.exp(np.cumsum(r))


def moving_average(p, window=20):
    k = np.ones(window) / window
    ma = np.convolve(p, k, mode="valid")
    return np.r_[np.full(window-1, np.nan), ma]


def components(p, window=20):
    y = np.log(p[-window:])
    x = np.arange(window, dtype=float)
    slope_log = np.polyfit(x, y, 1)[0]
    ma = np.mean(p[-window:])
    area = np.trapezoid(p[-window:] - ma, dx=1.0) / window
    sigma = np.std(np.diff(np.log(p[-window:])), ddof=1)
    slope_price = slope_log * p[-1]
    # scale both terms to comparable daily-price units for an interpretable score
    trend = 0.65 * slope_price + 0.35 * (area / window)
    return slope_price, area, sigma, trend


p = synthetic_market()
ma = moving_average(p)
slope, area, sigma, trend = components(p)

# Figure 1: price, moving baseline, positive/negative area pressure.
fig, ax = plt.subplots(figsize=(10, 5.8))
x = np.arange(len(p))
ax.plot(x, p, linewidth=1.8, label="Price")
ax.plot(x, ma, linewidth=1.5, label="20-session moving baseline")
valid = ~np.isnan(ma)
ax.fill_between(x[valid], p[valid], ma[valid], where=p[valid] >= ma[valid], alpha=.16, interpolate=True, label="Positive area pressure")
ax.fill_between(x[valid], p[valid], ma[valid], where=p[valid] < ma[valid], alpha=.16, interpolate=True, label="Negative area pressure")
ax.set(title="Trend–Area Decomposition | Illustrative Market Series", xlabel="Trading session", ylabel="Price index")
ax.legend(frameon=False, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "trend_area_stock_example.svg")

# Figure 2: uncertainty fan from the current state.
rng = np.random.default_rng(91)
horizon = 30
paths = 2500
start = p[-1]
mu = trend / start
shocks = rng.normal(mu, sigma, (paths, horizon))
forecast = start * np.exp(np.cumsum(shocks, axis=1))
q05, q25, q50, q75, q95 = np.quantile(forecast, [.05, .25, .50, .75, .95], axis=0)
h = np.arange(1, horizon + 1)
fig, ax = plt.subplots(figsize=(10, 5.8))
ax.plot(x[-80:], p[-80:], linewidth=1.8, label="Observed price")
future = x[-1] + h
ax.plot(future, q50, linewidth=2, label="Median forecast")
ax.fill_between(future, q05, q95, alpha=.12, label="90% uncertainty interval")
ax.fill_between(future, q25, q75, alpha=.20, label="50% uncertainty interval")
ax.axvline(x[-1], linestyle="--", linewidth=1)
ax.set(title="Trend–Area–Uncertainty Forecast Fan | Illustrative Example", xlabel="Trading session", ylabel="Price index")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "trend_uncertainty_fan.svg")

# Figure 3: calculation card as a clean bar visualization of normalized components.
normalized = np.array([slope / start, area / (20 * start), sigma, trend / start]) * 100
labels = ["Slope", "Area pressure", "Realized volatility", "Trend score"]
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.bar(labels, normalized)
ax.axhline(0, linewidth=1)
ax.set(title="Unified Model Components at Forecast Origin", ylabel="Normalized magnitude (%)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "trend_component_calculation.svg")

print(f"Current price: {start:.4f}")
print(f"Slope contribution: {slope:.6f} price units/session")
print(f"Accumulated area pressure: {area:.6f}")
print(f"Daily log-return uncertainty sigma: {sigma:.6f}")
print(f"Combined trend score: {trend:.6f}")
print(f"30-session median forecast: {q50[-1]:.4f}")
print(f"30-session 90% interval: [{q05[-1]:.4f}, {q95[-1]:.4f}]")
