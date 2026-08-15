from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from .core import portfolio_stats, dcf_value


def estimate_moments(returns, annualization=252, shrinkage=0.10):
    """Annualized mean/covariance with diagonal covariance shrinkage."""
    x=np.asarray(returns,float)
    if x.ndim!=2 or x.shape[0]<3: raise ValueError("returns must be T x N")
    mu=np.nanmean(x,axis=0)*annualization
    sample=np.cov(x,rowvar=False,ddof=1)*annualization
    cov=(1-shrinkage)*sample+shrinkage*np.diag(np.diag(sample))
    return mu,(cov+cov.T)/2


def risk_metrics(w,mu,cov,alpha=.95,rf=0.0):
    r,v=portfolio_stats(w,mu,cov); z=norm.ppf(alpha)
    return {"return":r,"volatility":v,"sharpe":(r-rf)/(v+1e-15),
            "var95":float(-(r-z*v)),"cvar95":float(-(r-v*norm.pdf(z)/(1-alpha)))}


def efficient_frontier(mu,cov,n_points=30):
    mu=np.asarray(mu,float); cov=np.asarray(cov,float); n=len(mu); out=[]
    for target in np.linspace(mu.min(),mu.max(),n_points):
        res=minimize(lambda w:w@cov@w,np.ones(n)/n,method="SLSQP",bounds=[(0,1)]*n,
          constraints=[{"type":"eq","fun":lambda w:w.sum()-1},{"type":"eq","fun":lambda w,t=target:w@mu-t}],
          options={"ftol":1e-11,"maxiter":1500})
        if res.success:
            r,v=portfolio_stats(res.x,mu,cov); out.append((v,r,res.x.copy()))
    return out


def sensitivity_grid(cashflows,discount_rates,terminal_growth_rates):
    return np.array([[dcf_value(cashflows,r,g) for g in terminal_growth_rates] for r in discount_rates])


def trend_features(prices,window=40):
    p=np.asarray(prices,float)
    if len(p)<window: raise ValueError("insufficient price history")
    y=np.log(p[-window:]); ret=np.diff(y); vol=np.std(ret,ddof=1)+1e-12
    return np.array([np.polyfit(np.arange(window),y,1)[0]/vol,
                     np.mean(ret[-5:])/vol,np.mean(ret[-20:])/vol,
                     np.trapezoid(y-y[0])/window**2/vol])


def fit_probabilistic_trend(prices,window=40,horizon=1,ridge=.5):
    p=np.asarray(prices,float); X=[]; y=[]
    for t in range(window,len(p)-horizon):
        X.append(trend_features(p[:t+1],window)); y.append(float(p[t+horizon]>p[t]))
    X=np.asarray(X); y=np.asarray(y); mean=X.mean(0); scale=X.std(0)+1e-12; Z=(X-mean)/scale; A=np.c_[np.ones(len(Z)),Z]
    def objective(b):
        score=A@b
        return float(np.mean(np.logaddexp(0,score)-y*score)+.5*ridge*np.sum(b[1:]**2)/len(y))
    res=minimize(objective,np.zeros(A.shape[1]),method="BFGS")
    return {"coef":res.x,"mean":mean,"scale":scale,"window":window,"horizon":horizon}


def trend_probability(prices,model):
    f=(trend_features(prices,model["window"])-model["mean"])/model["scale"]
    score=float(np.r_[1.,f]@model["coef"])
    return float(1/(1+np.exp(np.clip(-score,-700,700))))


def classification_metrics(probability,outcome):
    p=np.asarray(probability,float); y=np.asarray(outcome,int); accuracy=float(np.mean((p>=.5)==y)); baseline=float(max(y.mean(),1-y.mean()))
    return {"accuracy":accuracy,"majority_baseline":baseline,"skill":accuracy-baseline,"brier":float(np.mean((p-y)**2))}


def bootstrap_ci(values,n_boot=2000,seed=17):
    x=np.asarray(values,float); rng=np.random.default_rng(seed); samples=np.array([np.mean(rng.choice(x,len(x),replace=True)) for _ in range(n_boot)])
    return float(x.mean()),float(np.quantile(samples,.025)),float(np.quantile(samples,.975))
