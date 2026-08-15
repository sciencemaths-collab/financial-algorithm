import sys
sys.path.insert(0,"src")
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from financial_algorithms.core import *

OUT=Path("results"); OUT.mkdir(exist_ok=True)
SEEDS=range(20)

def market(seed,n=8):
    rng=np.random.default_rng(seed); mu=np.linspace(.055,.145,n)+rng.normal(0,.008,n); B=rng.normal(size=(n,n)); cov=B@B.T/700+np.eye(n)*.006
    return rng,mu,cov

# 1) Portfolio: convex sanity check plus genuinely non-convex transaction/cardinality benchmark.
rows=[]
for seed in SEEDS:
    rng,mu,cov=market(seed); prev=rng.dirichlet(np.ones(len(mu)))
    eq=np.ones(len(mu))/len(mu); sl=slsqp_portfolio(mu,cov); sa=sa_nonconvex_portfolio(mu,cov,prev,steps=6000,seed=seed); hy=hybrid_nonconvex_portfolio(mu,cov,prev,steps=6000,seed=seed)
    for name,w in [("Equal",eq),("SLSQP-convex",sl),("SA-nonconvex",sa),("Hybrid-nonconvex",hy)]: rows.append((seed,name,nonconvex_utility(w,mu,cov,prev),*portfolio_stats(w,mu,cov),(w>1e-6).sum()))
print("NONCONVEX PORTFOLIO, 20 SEEDS")
for name in ["Equal","SLSQP-convex","SA-nonconvex","Hybrid-nonconvex"]:
    a=np.array([r[2] for r in rows if r[1]==name]); print(f"{name:18s} mean utility={a.mean():.5f}  sd={a.std(ddof=1):.5f}")
fig,ax=plt.subplots(figsize=(9,5.5)); names=["Equal","SLSQP-convex","SA-nonconvex","Hybrid-nonconvex"]
data=[[r[2] for r in rows if r[1]==n] for n in names]; ax.boxplot(data,tick_labels=names,showmeans=True); ax.set_ylabel("True utility after costs & discrete penalties"); ax.set_title("Non-Convex Portfolio Optimization | 20 Independent Markets"); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUT/"nonconvex_portfolio_multiseed.svg")

# 2) DCF: convergence and uncertainty diagnostics.
base=[100,104,108,112,116]; ns=[250,500,1000,2500,5000,10000,20000]; med=[]; widths=[]
for n in ns:
    q=[monte_carlo_dcf(base,n_sims=n,seed=s)["p50"] for s in range(10)]; med.append(np.mean(q)); widths.append(np.std(q,ddof=1))
mc=monte_carlo_dcf(base,n_sims=50000,seed=42); print(f"DCF 50k: P10={mc['p10']:.2f} P50={mc['p50']:.2f} P90={mc['p90']:.2f}")
fig,ax=plt.subplots(figsize=(9,5.5)); ax.plot(ns,med,marker="o",label="Mean P50 across 10 seeds"); ax.fill_between(ns,np.array(med)-np.array(widths),np.array(med)+np.array(widths),alpha=.18,label="±1 SD across seeds"); ax.set_xscale("log"); ax.set_xlabel("Monte Carlo simulations (log scale)"); ax.set_ylabel("Estimated intrinsic value (P50)"); ax.set_title("Monte Carlo DCF Convergence & Reproducibility"); ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUT/"dcf_convergence.svg")

# 3) Trend model: train/test walk-forward on regimes where momentum signal exists, plus null control.
def synth(seed,n=1800,persistence=.10):
    rng=np.random.default_rng(seed); r=np.zeros(n); noise=rng.normal(0,.011,n)
    for t in range(1,n): r[t]=.0002+persistence*r[t-1]+noise[t]
    return 100*np.exp(np.cumsum(r))

def eval_series(p):
    split=900; model=fit_trend_model(p[:split],window=40,horizon=5,ridge=3.0); probs=[]; y=[]
    for t in range(split,len(p)-5): probs.append(forecast_probability(p[:t+1],model=model)); y.append(p[t+5]>p[t])
    probs=np.array(probs); y=np.array(y,float); acc=np.mean((probs>=.5)==y); base=max(y.mean(),1-y.mean()); brier=np.mean((probs-y)**2)
    return acc,base,brier
sig=np.array([eval_series(synth(s,.12)) for s in SEEDS]); null=np.array([eval_series(synth(s,0.0)) for s in SEEDS])
print(f"TREND signal markets: accuracy={sig[:,0].mean():.4f}, baseline={sig[:,1].mean():.4f}, brier={sig[:,2].mean():.4f}")
print(f"TREND null markets:   accuracy={null[:,0].mean():.4f}, baseline={null[:,1].mean():.4f}, brier={null[:,2].mean():.4f}")
fig,ax=plt.subplots(figsize=(9,5.5)); x=np.arange(len(SEEDS)); ax.scatter(x,sig[:,0]-sig[:,1],label="Persistent/momentum markets"); ax.scatter(x,null[:,0]-null[:,1],marker="x",label="Null random-walk control"); ax.axhline(0,linestyle="--"); ax.set_xlabel("Independent random seed"); ax.set_ylabel("Accuracy minus majority baseline"); ax.set_title("Trend Model Out-of-Sample Skill | Signal vs Null Control"); ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUT/"trend_signal_vs_null.svg")

with open(OUT/"phase2_summary.txt","w") as f:
    f.write("Phase 2 professional benchmark\n")
    f.write("20 independent synthetic markets; explicit null control; no claim of real-market predictive validity.\n")
    for name,d in zip(names,data): f.write(f"{name}: mean nonconvex utility {np.mean(d):.6f} +/- {np.std(d,ddof=1):.6f}\n")
    f.write(f"DCF 50k P10/P50/P90: {mc['p10']:.2f}/{mc['p50']:.2f}/{mc['p90']:.2f}\n")
    f.write(f"Trend signal acc/base/brier: {sig[:,0].mean():.4f}/{sig[:,1].mean():.4f}/{sig[:,2].mean():.4f}\n")
    f.write(f"Trend null acc/base/brier: {null[:,0].mean():.4f}/{null[:,1].mean():.4f}/{null[:,2].mean():.4f}\n")
