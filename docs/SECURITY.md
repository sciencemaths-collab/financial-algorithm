# Security and operational boundary

The public engine accepts bounded numeric data only. It rejects unknown operations and fields,
non-finite values, malformed covariance matrices, invalid probability assumptions, excessive
simulation or optimization budgets, and transaction-like operations.

The engine does not read environment credentials, contact networks, load executable objectives,
or place trades. Consumers must treat all supplied market and financial inputs as untrusted,
validate freshness and provenance outside this package, and retain human review. Sensitive data
must not be embedded in requests, logs, qualification reports, or source control.
