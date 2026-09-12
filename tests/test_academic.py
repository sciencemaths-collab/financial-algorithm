import numpy as np

from financial_algorithms.academic import (
    bootstrap_ci,
    classification_metrics,
    efficient_frontier,
    estimate_moments,
    fit_probabilistic_trend,
    risk_metrics,
    sensitivity_grid,
    trend_probability,
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


def test_moment_estimator_psd():
    rng = np.random.default_rng(1)
    _, cov = estimate_moments(rng.normal(0, 0.01, (250, 4)))
    assert np.linalg.eigvalsh(cov).min() >= -1e-10


def test_risk_metrics_finite():
    mu, cov = fixture()
    m = risk_metrics(np.ones(4) / 4, mu, cov)
    assert np.isfinite(list(m.values())).all()


def test_efficient_frontier_constructs_points():
    mu, cov = fixture()
    assert len(efficient_frontier(mu, cov, 12)) >= 6


def test_dcf_sensitivity_has_expected_direction():
    grid = sensitivity_grid([100] * 5, [0.08, 0.10, 0.12], [0.01, 0.02, 0.03])
    assert np.all(np.diff(grid, axis=0) < 0)
    assert np.all(np.diff(grid, axis=1) > 0)


def test_probabilistic_trend_is_bounded():
    rng = np.random.default_rng(2)
    p = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 500)))
    model = fit_probabilistic_trend(p[:350])
    q = trend_probability(p, model)
    assert 0 <= q <= 1 and np.isfinite(q)


def test_classification_metrics_and_ci():
    m = classification_metrics([0.8, 0.2, 0.7, 0.1], [1, 0, 1, 0])
    assert m["accuracy"] == 1
    mean, lo, hi = bootstrap_ci([1, 2, 3, 4], 500)
    assert lo <= mean <= hi
