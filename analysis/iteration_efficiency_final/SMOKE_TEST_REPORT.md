# Smoke test report

Verdict: **PASS — integration executable; no production campaign run.**

MATLAB R2025b ran real 16x2, three-update trajectories for Proposed, Yuksel,
Olhoff-LP, and genuine nested Olhoff-MMA. All stored fields were `double`; all
Candidate-C calls returned finite structural modes; no adaptive search failed.
The deliberately short topologies were not admissible, so the shared topology
grid retained explicitly unavailable cells rather than substituting designs.

Selector runs:

- `lp`: Proposed, Yuksel, and Olhoff-LP only — PASS.
- `mma`: Proposed, Yuksel, and Olhoff-MMA only; zero LP calls — PASS.
- `both`: separate LP and MMA identities/directories — PASS.

The consolidated suite passed 11/11 tests. It covered manifest/contract hashes,
Candidate-C first-batch and ordinal-13 adaptive anchors, exact-count topology,
reference/B_meas/persistence, method accounting, result schema, timing firewall,
all selectors, double checkpoint identity, scaling/renderer pipelines,
stale-artifact and production-lock rejection, and output isolation.

Synthetic scaling output is labelled `SMOKE_SYNTHETIC_*_NOT_SCIENTIFIC`; it is
pipeline validation only. The nine production meshes were not executed.
