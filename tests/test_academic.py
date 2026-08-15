import numpy as np
from financial_algorithms.academic import *


def fixture():
    mu=np.array([.08,.11,.06,.14]); cov=np.array([[.04,.01,.008,.012],[.01,.07,.01,.02],[.008,.01,.025,.006],[.012,.02,.006,.11]])
    return mu,cov


def test_moment_estimator_psd():
    rng=np.random.default_rng(1); _,cov=estimate_moments(rng.normal(0,.01,(250,4))); assert np.linalg.eigvalsh(cov).min()>=-1e-10


def test_risk_metrics_finite():
    mu,cov=fixture(); m=risk_metrics(np.ones(4)/4,mu,cov); assert np.isfinite(list(m.values())).all()


def test_efficient_frontier_constructs_points():
    mu,cov=fixture(); assert len(efficient_frontier(mu,cov,12))>=6


def test_dcf_sensitivity_has_expected_direction():
    grid=sensitivity_grid([100]*5,[.08,.10,.12],[.01,.02,.03]); assert np.all(np.diff(grid,axis=0)<0); assert np.all(np.diff(grid,axis=1)>0)


def test_probabilistic_trend_is_bounded():
    rng=np.random.default_rng(2); p=100*np.exp(np.cumsum(rng.normal(.001,.01,500))); model=fit_probabilistic_trend(p[:350]); q=trend_probability(p,model); assert 0<=q<=1 and np.isfinite(q)


def test_classification_metrics_and_ci():
    m=classification_metrics([.8,.2,.7,.1],[1,0,1,0]); assert m["accuracy"]==1; mean,lo,hi=bootstrap_ci([1,2,3,4],500); assert lo<=mean<=hi
