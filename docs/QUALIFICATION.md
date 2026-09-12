# Qualification

The formal suite runs 25 cases across portfolio allocation, valuation, positive-trend assessment,
and negative-trend assessment using five fixed seeds. Every case must replay exactly, produce a
content-addressed result, preserve limitations, and deny transaction authorization.

The result `QUALIFIED_FOR_DECISION_SUPPORT_BENCHMARKS` means those deterministic contract and
benchmark checks passed. It does not establish profitability, calibration on live markets,
suitability for a person or institution, regulatory compliance, or future performance.

Run `python -m financial_algorithms.qualification --output artifacts/qualification.json` to
produce the canonical report and SHA-256 digest.
