import sys
sys.path.insert(0, "src")
import numpy as np
import matplotlib.pyplot as plt
from financial_algorithms.core import *

rng = np.random.default_rng(42); n = 6
mu = np.array([.06, .08, .10, .12, .14, .09]); B = rng.normal(size=(n, n)); cov = B @ B.T / 500
methods = {"Equal": np.ones(n)/n, "SA": sa_portfolio(mu,cov,steps=8000,seed=42), "SLSQP": slsqp_portfolio(mu,cov), "Hybrid": hybrid_portfolio(mu,cov,steps=8000,seed=42)}
print("PORTFOLIO BENCHMARK")
for k,w in methods.items():
    r,v = portfolio_stats(w,mu,cov); print(f"{k:8s} utility={utility(w,mu,cov):.6f} return={r:.4f} risk={v:.4f}")

W = rng.dirichlet(np.ones(n),600); R = W @ mu; V = np.sqrt(np.einsum("ij,jk,ik->i",W,cov,W))
fig,ax = plt.subplots(figsize=(8,5)); ax.scatter(V,R,s=8,alpha=.20,label="Feasible portfolios")
for k,w in methods.items():
    r,v=portfolio_stats(w,mu,cov); ax.scatter(v,r,s=80,label=k)
ax.set(xlabel="Annualized volatility",ylabel="Expected return",title="Portfolio Optimization Benchmark"); ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig("results/portfolio_benchmark.svg")

mc=monte_carlo_dcf([100,104,108,112,116],n_sims=20000,seed=42); print(f"DCF P10/P50/P90 = {mc['p10']:.2f} / {mc['p50']:.2f} / {mc['p90']:.2f}")
fig,ax=plt.subplots(figsize=(8,5)); ax.hist(mc["values"],bins=70,density=True,alpha=.55)
for q in ("p10","p50","p90"): ax.axvline(mc[q],linestyle="--",label=q.upper())
ax.set(xlabel="Intrinsic value",ylabel="Density",title="Monte Carlo DCF Valuation Distribution"); ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig("results/dcf_distribution.svg")

prices=100*np.exp(np.cumsum(rng.normal(.0008,.012,1000))); probs=[]; actual=[]
for t in range(60,len(prices)-5): probs.append(forecast_probability(prices[:t+1])); actual.append(prices[t+5]>prices[t])
probs=np.array(probs); actual=np.array(actual); acc=np.mean((probs>=.5)==actual); brier=np.mean((probs-actual)**2); base=max(actual.mean(),1-actual.mean())
print(f"TREND walk-forward accuracy={acc:.4f}; majority baseline={base:.4f}; Brier={brier:.4f}")
fig,ax=plt.subplots(figsize=(8,5)); ax.plot(probs[:250],label="P(up)"); ax.axhline(.5,linestyle="--",label="Decision threshold"); ax.set(xlabel="Walk-forward step",ylabel="Probability",title="Market Trend Probability: Walk-Forward Evaluation"); ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig("results/trend_walkforward.svg")
