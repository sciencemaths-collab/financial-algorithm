"""Strict, versioned boundary for non-executing financial decisions."""

# mypy: disable-error-code=no-untyped-call

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from financial_algorithms.academic import risk_metrics
from financial_algorithms.core import (
    forecast_probability,
    hybrid_nonconvex_portfolio,
    monte_carlo_dcf,
    nonconvex_utility,
)

ENGINE_ID = "financial-algorithm"
ENGINE_VERSION = "0.2.0"
ADAPTER_VERSION = "1.0.0"
CAPABILITY_ID = "rad.decision.financial"
SCHEMA_VERSION = "1.0"


class DecisionEngineError(ValueError):
    """Safe contract or bounded-execution rejection."""


def decide(request: Mapping[str, Any]) -> dict[str, Any]:
    """Produce a deterministic decision-support result without executing a transaction."""
    normalized = _request(request)
    operation = normalized["operation"]
    if operation == "portfolio.allocate":
        decision = _portfolio(normalized)
    elif operation == "valuation.estimate":
        decision = _valuation(normalized)
    else:
        decision = _market(normalized)
    result = {
        "schema_version": SCHEMA_VERSION,
        "engine_id": ENGINE_ID,
        "engine_version": ENGINE_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "capability_id": CAPABILITY_ID,
        "request_digest": _digest(normalized),
        "operation": operation,
        "decision": decision,
        "execution_authorized": False,
        "limitations": [
            "decision_support_only",
            "not_investment_advice",
            "no_profit_or_loss_guarantee",
            "historical_or_assumed_inputs_may_not_predict_future_conditions",
        ],
    }
    return cast(dict[str, Any], json.loads(json.dumps(result, sort_keys=True, allow_nan=False)))


def _request(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DecisionEngineError("decision request must be an object")
    operation = value.get("operation")
    if operation == "portfolio.allocate":
        return _portfolio_request(value)
    if operation == "valuation.estimate":
        return _valuation_request(value)
    if operation == "market.assess":
        return _market_request(value)
    raise DecisionEngineError("decision operation is unsupported")


def _portfolio_request(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "operation",
        "expected_returns",
        "covariance",
        "previous_weights",
        "risk_aversion",
        "transaction_cost",
        "cardinality_penalty",
        "min_position",
        "max_assets",
        "steps",
        "seed",
    }
    _fields(value, allowed)
    returns = _numbers(value.get("expected_returns"), "expected_returns", 2, 64, -5.0, 5.0)
    covariance = value.get("covariance")
    if not isinstance(covariance, Sequence) or isinstance(covariance, (str, bytes)):
        raise DecisionEngineError("covariance must be a square matrix")
    matrix = [
        _numbers(row, "covariance row", len(returns), len(returns), -25.0, 25.0)
        for row in covariance
    ]
    if len(matrix) != len(returns):
        raise DecisionEngineError("covariance must be a square matrix")
    cov = np.asarray(matrix, dtype=float)
    if not np.allclose(cov, cov.T, atol=1e-10) or np.linalg.eigvalsh(cov).min() < -1e-10:
        raise DecisionEngineError("covariance must be symmetric positive semidefinite")
    previous = value.get("previous_weights", [1.0 / len(returns)] * len(returns))
    weights = _numbers(previous, "previous_weights", len(returns), len(returns), 0.0, 1.0)
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-8):
        raise DecisionEngineError("previous_weights must sum to one")
    request = {
        "operation": "portfolio.allocate",
        "expected_returns": returns,
        "covariance": matrix,
        "previous_weights": weights,
        "risk_aversion": _number(value.get("risk_aversion", 3.0), "risk_aversion", 0.01, 100.0),
        "transaction_cost": _number(
            value.get("transaction_cost", 0.01), "transaction_cost", 0.0, 1.0
        ),
        "cardinality_penalty": _number(
            value.get("cardinality_penalty", 0.004), "cardinality_penalty", 0.0, 1.0
        ),
        "min_position": _number(value.get("min_position", 0.08), "min_position", 0.0, 1.0),
        "max_assets": _integer(
            value.get("max_assets", min(4, len(returns))), "max_assets", 1, len(returns)
        ),
        "steps": _integer(value.get("steps", 5000), "steps", 100, 100_000),
        "seed": _integer(value.get("seed", 7), "seed", 0, 2**32 - 1),
    }
    return request


def _valuation_request(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "operation",
        "cashflows",
        "discount_rate",
        "growth_mu",
        "growth_sigma",
        "simulations",
        "seed",
    }
    _fields(value, allowed)
    cashflows = _numbers(value.get("cashflows"), "cashflows", 2, 100, 1e-9, 1e15)
    discount = _number(value.get("discount_rate"), "discount_rate", -0.99, 10.0)
    growth_mu = _number(value.get("growth_mu", 0.04), "growth_mu", -0.99, 2.0)
    growth_sigma = _number(value.get("growth_sigma", 0.08), "growth_sigma", 0.0, 2.0)
    if discount <= 0.02:
        raise DecisionEngineError("discount_rate must exceed terminal growth 0.02")
    return {
        "operation": "valuation.estimate",
        "cashflows": cashflows,
        "discount_rate": discount,
        "growth_mu": growth_mu,
        "growth_sigma": growth_sigma,
        "simulations": _integer(value.get("simulations", 10_000), "simulations", 100, 100_000),
        "seed": _integer(value.get("seed", 7), "seed", 0, 2**32 - 1),
    }


def _market_request(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"operation", "prices", "window", "horizon", "neutral_band"}
    _fields(value, allowed)
    prices = _numbers(value.get("prices"), "prices", 20, 100_000, 1e-12, 1e15)
    window = _integer(value.get("window", 20), "window", 20, min(5_000, len(prices)))
    return {
        "operation": "market.assess",
        "prices": prices,
        "window": window,
        "horizon": _integer(value.get("horizon", 5), "horizon", 1, 252),
        "neutral_band": _number(value.get("neutral_band", 0.05), "neutral_band", 0.0, 0.49),
    }


def _portfolio(request: dict[str, Any]) -> dict[str, Any]:
    mu = np.asarray(request["expected_returns"], dtype=float)
    cov = np.asarray(request["covariance"], dtype=float)
    previous = np.asarray(request["previous_weights"], dtype=float)
    kwargs = {
        key: request[key]
        for key in (
            "risk_aversion",
            "transaction_cost",
            "cardinality_penalty",
            "min_position",
            "max_assets",
        )
    }
    weights = hybrid_nonconvex_portfolio(
        mu, cov, previous, request["steps"], request["seed"], **kwargs
    )
    metrics = risk_metrics(weights, mu, cov)
    return {
        "recommendation": "REVIEW_ALLOCATION",
        "weights": weights.tolist(),
        "active_assets": int(np.count_nonzero(weights > 1e-6)),
        "utility": float(nonconvex_utility(weights, mu, cov, previous, **kwargs)),
        "expected_return": metrics["return"],
        "volatility": metrics["volatility"],
        "sharpe_assuming_zero_risk_free_rate": metrics["sharpe"],
        "value_at_risk_95_normal_assumption": metrics["var95"],
        "conditional_value_at_risk_95_normal_assumption": metrics["cvar95"],
        "seed": request["seed"],
        "steps": request["steps"],
    }


def _valuation(request: dict[str, Any]) -> dict[str, Any]:
    result = monte_carlo_dcf(
        request["cashflows"],
        request["discount_rate"],
        request["growth_mu"],
        request["growth_sigma"],
        request["simulations"],
        request["seed"],
    )
    return {
        "recommendation": "REVIEW_VALUATION_RANGE",
        "p10": result["p10"],
        "p50": result["p50"],
        "p90": result["p90"],
        "simulations": request["simulations"],
        "seed": request["seed"],
    }


def _market(request: dict[str, Any]) -> dict[str, Any]:
    probability = forecast_probability(request["prices"], request["window"], request["horizon"])
    band = request["neutral_band"]
    signal = (
        "POSITIVE"
        if probability > 0.5 + band
        else "NEGATIVE"
        if probability < 0.5 - band
        else "NEUTRAL"
    )
    return {
        "recommendation": "REVIEW_MARKET_ASSESSMENT",
        "signal": signal,
        "up_probability": probability,
        "horizon": request["horizon"],
        "window": request["window"],
    }


def _fields(value: Mapping[str, Any], allowed: set[str]) -> None:
    if set(value) - allowed:
        raise DecisionEngineError("decision request contains unknown fields")


def _numbers(
    value: object, name: str, minimum_items: int, maximum_items: int, minimum: float, maximum: float
) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not minimum_items <= len(value) <= maximum_items
    ):
        raise DecisionEngineError(f"{name} has invalid dimensions")
    return [_number(item, name, minimum, maximum) for item in value]


def _number(value: object, name: str, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise DecisionEngineError(f"{name} is outside bounded limits")
    return float(value)


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise DecisionEngineError(f"{name} is outside bounded limits")
    return value


def _digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()
