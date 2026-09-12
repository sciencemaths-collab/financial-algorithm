from financial_algorithms.qualification import qualify


def test_formal_qualification_is_reproducible_and_complete():
    first = qualify()
    second = qualify()
    assert first == second
    assert first["qualification"] == "QUALIFIED_FOR_DECISION_SUPPORT_BENCHMARKS"
    assert first["case_count"] == first["passed_count"] == 25
    assert all(case["deterministic_replay"] for case in first["cases"])
    assert all(case["execution_authorized"] is False for case in first["cases"])
    assert first["report_digest"].startswith("sha256:")
