"""Formal deterministic qualification for the financial decision engine."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from financial_algorithms.engine import (
    ADAPTER_VERSION,
    CAPABILITY_ID,
    ENGINE_ID,
    ENGINE_VERSION,
    decide,
)


def qualify() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for seed in (7, 19, 43, 101, 211):
        requests = (
            _portfolio(seed, 0.0),
            _portfolio(seed, 0.01),
            {
                "operation": "valuation.estimate",
                "cashflows": [100, 105, 110, 115, 120],
                "discount_rate": 0.10,
                "growth_sigma": 0.08,
                "simulations": 1000,
                "seed": seed,
            },
            {"operation": "market.assess", "prices": [100 + index for index in range(80)]},
            {"operation": "market.assess", "prices": [180 - index for index in range(80)]},
        )
        for index, request in enumerate(requests):
            first, second = decide(request), decide(request)
            valid = (
                first == second
                and first["execution_authorized"] is False
                and bool(first["limitations"])
            )
            cases.append(
                {
                    "case_id": f"decision-{seed}-{index}",
                    "operation": request["operation"],
                    "deterministic_replay": first == second,
                    "execution_authorized": first["execution_authorized"],
                    "request_digest": first["request_digest"],
                    "outcome": "PASS" if valid else "FAIL",
                }
            )
    passed = sum(case["outcome"] == "PASS" for case in cases)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "engine_id": ENGINE_ID,
        "engine_version": ENGINE_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "capability_id": CAPABILITY_ID,
        "qualification": "QUALIFIED_FOR_DECISION_SUPPORT_BENCHMARKS"
        if passed == len(cases)
        else "UNQUALIFIED",
        "case_count": len(cases),
        "passed_count": passed,
        "cases": cases,
        "limitations": [
            "decision_support_only",
            "no_live_market_data_qualification",
            "no_broker_or_transaction_execution",
            "no_profit_or_loss_guarantee",
            "benchmark_qualification_is_not_future_performance",
        ],
    }
    report["report_digest"] = _digest(report)
    return report


def _portfolio(seed: int, transaction_cost: float) -> dict[str, Any]:
    return {
        "operation": "portfolio.allocate",
        "expected_returns": [0.08, 0.11, 0.06, 0.14],
        "covariance": [
            [0.04, 0.01, 0.008, 0.012],
            [0.01, 0.07, 0.01, 0.02],
            [0.008, 0.01, 0.025, 0.006],
            [0.012, 0.02, 0.006, 0.11],
        ],
        "transaction_cost": transaction_cost,
        "steps": 1000,
        "seed": seed,
    }


def _digest(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualify the financial decision engine")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = qualify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    return 0 if report["qualification"] == "QUALIFIED_FOR_DECISION_SUPPORT_BENCHMARKS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
