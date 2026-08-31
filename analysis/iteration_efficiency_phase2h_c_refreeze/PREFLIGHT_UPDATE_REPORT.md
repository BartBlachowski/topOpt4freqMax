# Phase 2H preflight update report

Preflight now verifies the frozen contract, Candidate C evaluator identity and digest,
Phase 2H freeze-record digest, unchanged topology/budget/timing/source identities, output
isolation, and the selected Olhoff route. It separately validates three qualification
artifacts against exact schema, scope, Candidate, classifier, evaluator hash, contract
hash, Olhoff variant, nonempty input provenance, nonempty results, and `pass=true`.

Seven negative controls passed: old evaluator hash, old contract hash, wrong Candidate,
wrong classifier, wrong scope, wrong route, and false pass are rejected. Missing precision,
cross-method, and reference-length artifacts all fail closed. Default behavior still throws
`ie2a:PreflightFailed`; `ThrowOnFailure=false` exists only to capture test evidence.

Post-refreeze expected result: **FAIL CLOSED** solely because the three new qualification
artifacts are absent (and authorization is absent when requested). Old Phase 2B precision
evidence is preserved but cannot satisfy the Candidate C schema.

