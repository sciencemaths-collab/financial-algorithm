import sys
sys.path.insert(0,"src")
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from financial_algorithms.core import slsqp_portfolio,sa_nonconvex_portfolio,hybrid_nonconvex_portfolio,nonconvex_utility,monte_carlo_dcf
from financial_algorithms.academic import fit_probabilistic_trend,trend_probability,classification_metrics,bootstrap_ci

OUT=Path("results"); OUT.mkdir(exist_ok=True); SEEDS=range(30)

# Portfolio: evaluate the true non-convex objective across independent markets.
scores={k:[] for k in ["Equal","SLSQP","SA","Hybrid"]}
for seed in SEEDS:
    rng=np.random.default_rng(seed); n=8; mu=np.linspace(.05,.15,n)+rng.normal(0,.008,n); B=rng.normal(size=(n,n)); cov=B@B.T/800+np.eye(n)*.008; prev=rng.dirichlet(np.ones(n))
    methods={"Equal":np.ones(n)/n,"SLSQP":slsqp_portfolio(mu,cov),"SA":sa_nonconvex_portfolio(mu,cov,prev,steps=3500,seed=seed),"Hybrid":hybrid_nonconvex_portfolio(mu,cov,prev,steps=3500,seed=seed)}
    for name,w in methods.items(): scores[name].append(nonconvex_utility(w,mu,cov,prev))
fig,ax=plt.subplots(figsize=(9,5.4)); ax.boxplot([scores[k] for k in scores],tick_labels=list(scores),showmeans=True); ax.set(title="Non-Convex Portfolio Benchmark | 30 Independent Markets",ylabel="Utility after risk, turnover & discrete penalties"); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUT/"portfolio_academic.svg")

# DCF convergence.
base=[100,104,108,112,116]; ns=[250,500,1000,2500,5000,10000,20000]; med=[]; sd=[]
for n in ns:
    q=[monte_carlo_dcf(base,n_sims=n,seed=s)["p50"] for s in range(12)]; med.append(np.mean(q)); sd.append(np.std(q,ddof=1))
fig,ax=plt.subplots(figsize=(9,5.4)); ax.plot(ns,med,marker="o"); ax.fill_between(ns,np.array(med)-sd,np.array(med)+sd,alpha=.2); ax.set_xscale("log"); ax.set(title="Monte Carlo DCF Convergence | 12 Seeds",xlabel="Simulation paths",ylabel="Median intrinsic value"); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUT/"dcf_academic.svg")

# Trend: controlled signal recovery versus null random walk.
def synthetic(seed,persistence):
    rng=np.random.default_rng(seed); r=np.zeros(1500); noise=rng.normal(0,.011,len(r))
    for t in range(1,len(r)): r[t]=.0002+persistence*r[t-1]+noise[t]
    return 100*np.exp(np.cumsum(r))
def evaluate(p):
    split=750; model=fit_probabilistic_trend(p[:split],40,1,.5); prob=[]; y=[]
    for t in range(split,len(p)-1): prob.append(trend_probability(p[:t+1],model)); y.append(p[t+1]>p[t])
    return classification_metrics(prob,y)
signal=[evaluate(synthetic(s,.30)) for s in SEEDS]; null=[evaluate(synthetic(s,0)) for s in SEEDS]; ds=[m["skill"] for m in signal]; dn=[m["skill"] for m in null]
fig,ax=plt.subplots(figsize=(9,5.4)); ax.scatter(list(SEEDS),ds,label="Persistent process"); ax.scatter(list(SEEDS),dn,marker="x",label="Null random walk"); ax.axhline(0,ls="--"); ax.set(title="Out-of-Sample Trend Skill | Signal vs Null Control",xlabel="Independent seed",ylabel="Accuracy minus majority baseline"); ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUT/"trend_academic.svg")

print("Portfolio")
for name,v in scores.items(): print(name,bootstrap_ci(v))
mc=monte_carlo_dcf(base,n_sims=50000,seed=42); print("DCF P10/P50/P90",mc["p10"],mc["p50"],mc["p90"])
print("Trend signal",bootstrap_ci(ds)); print("Trend null",bootstrap_ci(dn))
