import numpy as np
import pytest

from financial_algorithms.core import (
    dcf_value,
    fit_trend_model,
    forecast_probability,
    hybrid_nonconvex_portfolio,
    hybrid_portfolio,
    monte_carlo_dcf,
    nonconvex_utility,
    sa_nonconvex_portfolio,
    sa_portfolio,
    slsqp_portfolio,
    utility,
)


def fixture():
    mu = np.array([0.08, 0.11, 0.06, 0.14])
    cov = np.array(
        [
            [0.04, 0.01, 0.008, 0.012],
            [0.01, 0.07, 0.01, 0.02],
            [0.008, 0.01, 0.025, 0.006],
            [0.012, 0.02, 0.006, 0.11],
        ]
    )
    return mu, cov


def test_portfolios_feasible():
    mu, cov = fixture()
    for f in (sa_portfolio, slsqp_portfolio, hybrid_portfolio):
        w = f(mu, cov)
        assert abs(w.sum() - 1) < 1e-8
        assert np.all(w >= -1e-10)


def test_hybrid_not_worse_than_sa():
    mu, cov = fixture()
    sa = sa_portfolio(mu, cov)
    hy = hybrid_portfolio(mu, cov)
    assert utility(hy, mu, cov) >= utility(sa, mu, cov) - 1e-10


def test_nonconvex_search_feasible_and_finite():
    mu, cov = fixture()
    prev = np.array([0.7, 0.1, 0.1, 0.1])
    w = sa_nonconvex_portfolio(mu, cov, prev, steps=1200, seed=3)
    assert np.isclose(w.sum(), 1)
    assert np.all(w >= 0)
    assert np.isfinite(nonconvex_utility(w, mu, cov, prev))


def test_nonconvex_hybrid_preserves_true_objective():
    mu, cov = fixture()
    prev = np.array([0.7, 0.1, 0.1, 0.1])
    sa = sa_nonconvex_portfolio(mu, cov, prev, steps=1500, seed=4)
    hy = hybrid_nonconvex_portfolio(mu, cov, prev, steps=1500, seed=4)
    assert nonconvex_utility(hy, mu, cov, prev) >= nonconvex_utility(sa, mu, cov, prev) - 1e-12


def test_dcf_positive_and_discount_monotonic():
    cf = [100, 105, 110, 115, 120]
    assert dcf_value(cf, 0.08) > dcf_value(cf, 0.12) > 0


def test_dcf_rejects_invalid_terminal_assumption():
    with pytest.raises(ValueError):
        dcf_value([100, 105], 0.02, 0.02)


def test_mc_quantiles_ordered():
    r = monte_carlo_dcf([100] * 5, n_sims=1000)
    assert r["p10"] < r["p50"] < r["p90"]


def test_mc_reproducible():
    a = monte_carlo_dcf([100] * 5, n_sims=500, seed=9)
    b = monte_carlo_dcf([100] * 5, n_sims=500, seed=9)
    assert np.array_equal(a["values"], b["values"])


def test_trend_probability_direction():
    up = np.linspace(100, 140, 80)
    down = np.linspace(140, 100, 80)
    assert forecast_probability(up) > 0.5
    assert forecast_probability(down) < 0.5


def test_calibrated_trend_model_probability_bounds():
    rng = np.random.default_rng(1)
    p = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 500)))
    m = fit_trend_model(p[:350])
    q = forecast_probability(p, model=m)
    assert 0 <= q <= 1 and np.isfinite(q)
