from pathlib import Path

README = Path(__file__).parents[1] / "README.md"


def test_readme_distinguishes_scenarios_from_market_evidence() -> None:
    text = README.read_text(encoding="utf-8")
    assert "Market Prediction" not in text
    assert "not predictions validated on live markets" in text
    assert "not historical backtests, live-market validation" in text
    assert "cannot fetch live market data" in text
    assert "QUALIFIED_FOR_DECISION_SUPPORT_BENCHMARKS" in text
