import json
import math
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from financial_algorithms.engine import DecisionEngineError, decide


def portfolio_request(seed=7):
    return {
        "operation": "portfolio.allocate",
        "expected_returns": [0.08, 0.11, 0.06, 0.14],
        "covariance": [
            [0.04, 0.01, 0.008, 0.012],
            [0.01, 0.07, 0.01, 0.02],
            [0.008, 0.01, 0.025, 0.006],
            [0.012, 0.02, 0.006, 0.11],
        ],
        "previous_weights": [0.25, 0.25, 0.25, 0.25],
        "steps": 400,
        "seed": seed,
    }


def test_portfolio_decision_is_bounded_reproducible_and_non_executing():
    first = decide(portfolio_request())
    second = decide(portfolio_request())
    assert first == second
    assert first["execution_authorized"] is False
    assert math.isclose(sum(first["decision"]["weights"]), 1.0, abs_tol=1e-8)
    assert first["decision"]["active_assets"] <= 4
    assert first["request_digest"].startswith("sha256:")


def test_valuation_decision_is_reproducible_and_ordered():
    request = {
        "operation": "valuation.estimate",
        "cashflows": [100, 105, 110, 115, 120],
        "discount_rate": 0.10,
        "simulations": 500,
        "seed": 19,
    }
    first = decide(request)
    assert first == decide(request)
    assert first["decision"]["p10"] < first["decision"]["p50"] < first["decision"]["p90"]
    assert "values" not in first["decision"]


def test_market_assessment_is_non_transactional():
    result = decide({"operation": "market.assess", "prices": list(range(100, 180))})
    assert result["decision"]["signal"] == "POSITIVE"
    assert result["decision"]["recommendation"] == "REVIEW_MARKET_ASSESSMENT"
    assert result["execution_authorized"] is False


def test_public_examples_match_machine_readable_contracts():
    root = Path(__file__).parent.parent
    request_schema = json.loads((root / "schemas/decision-request.schema.json").read_text())
    result_schema = json.loads((root / "schemas/decision-result.schema.json").read_text())
    payloads = [
        portfolio_request(),
        {
            "operation": "valuation.estimate",
            "cashflows": [100, 105, 110],
            "discount_rate": 0.10,
            "simulations": 100,
        },
        {"operation": "market.assess", "prices": list(range(100, 180))},
    ]
    for payload in payloads:
        Draft202012Validator(request_schema).validate(payload)
        Draft202012Validator(result_schema).validate(decide(payload))


@pytest.mark.parametrize(
    "payload",
    [
        {"operation": "trade.execute"},
        {**portfolio_request(), "unexpected": 1},
        {**portfolio_request(), "covariance": [[1, 2], [0, 1]]},
        {**portfolio_request(), "expected_returns": [float("nan"), 0.1]},
        {**portfolio_request(), "steps": 10_000_000},
        {"operation": "valuation.estimate", "cashflows": [100, 110], "discount_rate": 0.02},
        {"operation": "market.assess", "prices": [100] * 19},
    ],
)
def test_invalid_unsafe_or_unbounded_requests_fail_closed(payload):
    with pytest.raises(DecisionEngineError):
        decide(payload)
