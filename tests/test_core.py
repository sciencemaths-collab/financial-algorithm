import numpy as np
from financial_algorithms.core import *


def fixture():
    mu = np.array([.08, .11, .06, .14])
    cov = np.array([[.04, .01, .008, .012], [.01, .07, .01, .02], [.008, .01, .025, .006], [.012, .02, .006, .11]])
    return mu, cov


def test_portfolios_feasible():
    mu, cov = fixture()
    for f in (sa_portfolio, slsqp_portfolio, hybrid_portfolio):
        w = f(mu, cov)
        assert abs(w.sum() - 1) < 1e-8
        assert np.all(w >= -1e-10)


def test_hybrid_not_worse_than_sa():
    mu, cov = fixture(); sa = sa_portfolio(mu, cov); hy = hybrid_portfolio(mu, cov)
    assert utility(hy, mu, cov) >= utility(sa, mu, cov) - 1e-10


def test_dcf_positive_and_discount_monotonic():
    cf = [100, 105, 110, 115, 120]
    assert dcf_value(cf, .08) > dcf_value(cf, .12) > 0


def test_mc_quantiles_ordered():
    r = monte_carlo_dcf([100] * 5, n_sims=1000)
    assert r["p10"] < r["p50"] < r["p90"]


def test_trend_probability_direction():
    up = np.linspace(100, 140, 80); down = np.linspace(140, 100, 80)
    assert forecast_probability(up) > .5
    assert forecast_probability(down) < .5
