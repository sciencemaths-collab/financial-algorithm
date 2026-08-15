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


def _project_simplex(z):
    z = np.maximum(np.asarray(z, float), 0)
    return z / z.sum() if z.sum() else np.ones(len(z)) / len(z)


def sa_portfolio(mu, cov, risk_aversion=3.0, steps=5000, seed=7):
    rng = np.random.default_rng(seed); n = len(mu); w = np.ones(n) / n
    score = utility(w, mu, cov, risk_aversion); best = w.copy(); bests = score
    for k in range(steps):
        t = max(1e-5, 0.05 * (1 - k / steps))
        z = _project_simplex(w + rng.normal(0, 0.08, n)); s = utility(z, mu, cov, risk_aversion)
        if s > score or rng.random() < np.exp((s - score) / t): w, score = z, s
        if score > bests: best, bests = w.copy(), score
    return best


def hybrid_portfolio(mu, cov, risk_aversion=3.0, steps=5000, seed=7):
    return slsqp_portfolio(mu, cov, risk_aversion, sa_portfolio(mu, cov, risk_aversion, steps, seed))


def nonconvex_utility(w, mu, cov, previous=None, risk_aversion=3.0, transaction_cost=0.01,
                      cardinality_penalty=0.004, min_position=0.08, max_assets=4):
    w = np.asarray(w, float); previous = np.ones(len(w))/len(w) if previous is None else np.asarray(previous, float)
    turnover = np.abs(w-previous).sum()
    active = w > 1e-6
    small = np.logical_and(active, w < min_position).sum()
    excess = max(int(active.sum())-max_assets, 0)
    return utility(w, mu, cov, risk_aversion) - transaction_cost*turnover - cardinality_penalty*(small+excess)


def sa_nonconvex_portfolio(mu, cov, previous=None, steps=12000, seed=7, **kwargs):
    rng=np.random.default_rng(seed); n=len(mu); w=np.ones(n)/n
    score=nonconvex_utility(w,mu,cov,previous,**kwargs); best=w.copy(); bests=score
    for k in range(steps):
        temp=max(2e-5, .035*(1-k/steps)**2)
        z=_project_simplex(w+rng.normal(0,.07,n))
        # occasional sparse/cardinality jump makes the search genuinely non-smooth
        if rng.random()<.18:
            keep=rng.choice(n, size=rng.integers(2,min(5,n+1)), replace=False); mask=np.zeros(n,bool); mask[keep]=True
            z=_project_simplex(z*mask)
        s=nonconvex_utility(z,mu,cov,previous,**kwargs)
        if s>score or rng.random()<np.exp((s-score)/temp): w,score=z,s
        if score>bests: best,bests=w.copy(),score
    return best


def hybrid_nonconvex_portfolio(mu, cov, previous=None, steps=12000, seed=7, **kwargs):
    # SA selects a basin/support; SLSQP refines continuous weights on that support.
    w=sa_nonconvex_portfolio(mu,cov,previous,steps,seed,**kwargs); support=w>1e-6
    idx=np.flatnonzero(support); sub_mu=np.asarray(mu)[idx]; sub_cov=np.asarray(cov)[np.ix_(idx,idx)]
    refined=slsqp_portfolio(sub_mu,sub_cov,kwargs.get("risk_aversion",3.0),w[idx]/w[idx].sum())
    out=np.zeros(len(mu)); out[idx]=refined
    # accept refinement only if the true discontinuous objective improves
    return out if nonconvex_utility(out,mu,cov,previous,**kwargs)>=nonconvex_utility(w,mu,cov,previous,**kwargs) else w


def dcf_value(cashflows, discount_rate, terminal_growth=0.02):
    cf=np.asarray(cashflows,float); n=len(cf)
    if discount_rate <= terminal_growth: raise ValueError("discount_rate must exceed terminal_growth")
    pv=sum(cf[t]/(1+discount_rate)**(t+1) for t in range(n)); tv=cf[-1]*(1+terminal_growth)/(discount_rate-terminal_growth)
    return float(pv+tv/(1+discount_rate)**n)


def monte_carlo_dcf(base_cf, discount_rate=0.10, growth_mu=0.04, growth_sigma=0.08, n_sims=10000, seed=7):
    rng=np.random.default_rng(seed); base=np.asarray(base_cf,float); vals=[]
    for _ in range(n_sims):
        shocks=rng.normal(growth_mu,growth_sigma,len(base)); cf=np.empty_like(base); cf[0]=base[0]*(1+shocks[0])
        for i in range(1,len(base)): cf[i]=cf[i-1]*(1+shocks[i])
        vals.append(dcf_value(cf,discount_rate))
    a=np.asarray(vals); return {"values":a,"p10":float(np.quantile(a,.1)),"p50":float(np.quantile(a,.5)),"p90":float(np.quantile(a,.9))}


def trend_features(prices, window=40):
    p=np.asarray(prices,float); y=np.log(p[-window:]); r=np.diff(y); x=np.arange(len(y),dtype=float)
    slope=np.polyfit(x,y,1)[0]
    short=np.mean(r[-5:]); medium=np.mean(r[-20:]) if len(r)>=20 else np.mean(r)
    area=np.trapezoid(y-y[0],dx=1.0)/len(y)
    vol=np.std(r,ddof=1)+1e-12
    return np.array([slope/vol, short/vol, medium/vol, area/(len(y)*vol)])


def fit_trend_model(price_series, window=40, horizon=5, ridge=1.0):
    p=np.asarray(price_series,float); X=[]; yy=[]
    for t in range(window,len(p)-horizon):
        X.append(trend_features(p[:t+1],window)); yy.append(float(p[t+horizon]>p[t]))
    X=np.asarray(X); yy=np.asarray(yy); X1=np.column_stack([np.ones(len(X)),X])
    # stable ridge probability model fitted to +/-1 target; sigmoid maps score to probability
    reg=np.eye(X1.shape[1])*ridge; reg[0,0]=0
    coef=np.linalg.solve(X1.T@X1+reg,X1.T@(2*yy-1))
    return {"coef":coef,"window":window,"horizon":horizon}


def forecast_probability(prices, window=20, horizon=5, model=None):
    p=np.asarray(prices,float)
    if model is not None:
        f=trend_features(p,model["window"]); score=float(np.r_[1.0,f]@model["coef"])
        return float(1/(1+np.exp(-2*score)))
    ret=np.diff(np.log(p)); sigma=float(np.std(ret[-max(window,2):],ddof=1)); y=np.log(p[-window:]); slope=np.polyfit(np.arange(window),y,1)[0]
    z=slope*np.sqrt(horizon)/(sigma+1e-12); return float(norm.cdf(z))
