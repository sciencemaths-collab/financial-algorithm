# Financial Decision Engine contract 1.0

Engine `financial-algorithm` 0.2.0 implements RAD capability `rad.decision.financial` through
adapter contract 1.0.0. Requests and results are JSON-compatible and content-addressed.

Supported operations are `portfolio.allocate`, `valuation.estimate`, and `market.assess`.
Inputs have explicit dimensional, numeric, simulation, and iteration bounds. Portfolio covariance
must be finite, symmetric, and positive semidefinite. Results declare the exact engine identity,
request digest, limitations, and `execution_authorized: false`.

The engine deliberately has no broker, order, account, credential, network, or transaction API.
A recommendation is an artifact for review; it is never permission to act. Any future execution
system requires a separate capability, risk engine, approval policy, qualification, and release.
