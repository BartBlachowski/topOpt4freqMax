# Durable raw scientific evidence

Element-level solver trajectories for `analysis/OlhoffCurrent` diagnostics.

Layout: `evidence/<study>/<run>.mat`, one directory per diagnostic study, names
deterministic (`<arm>_<nelx>x<nely>_trajectory.mat`).

**These files are untracked but not disposable.** Each is declared in its study's
`EVIDENCE.json` with byte size, SHA-256, variable dimensions and precision, and
`olhoffcurrent_evidence_gate` fails that study if a declared-required file is
missing or altered. See [`../EVIDENCE_POLICY.md`](../EVIDENCE_POLICY.md).

If this directory is empty (a fresh clone, or a machine that has not run the
diagnostics), the gate reports `REQUIRED_MISSING` and the affected studies are
correctly reported as unverifiable on this machine. `EVIDENCE.json` names the
exact files and hashes needed to restore that state.
