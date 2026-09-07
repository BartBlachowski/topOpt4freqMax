# Raw-evidence retention policy — `analysis/OlhoffCurrent`

## The defect this policy exists to prevent

Three completed diagnostics — `move_stop`, `admission_rule`, `move_transition` —
each reconstructed a full `NE x nOuter` density trajectory, saved it to a
`.mat` file, drew conclusions from it, and declared themselves frozen. **Every
one of those files is gone.**

They were written under paths matched by `*.mat`, which is ignored both
repo-wide (root `.gitignore`) and again in
`analysis/OlhoffCurrent/diagnostics/.gitignore`, whose comment reasoned that raw
solver state is "fully reproducible from `code/`". Reproducible it may be, at a
cost of hours of solver time per run — and the runs were never re-made. Nothing
ever checked that the files still existed.

`move_stop` had at least hashed its four trajectories in `FINAL_SHA256.txt`, so
its loss is **provable**. `admission_rule` and `move_transition` never
manifested theirs at all, so for those the loss is only inferable from the code
that wrote them. The consequence was concrete: a later offline study had to
record element identity, spatial location and persistence as permanently
unanswerable, and could not compute a single percentile of the very distribution
the study was about.

The defect was **not** that the files were untracked. Large binaries do not
belong in git. The defect was that they were **undeclared**: nothing named them,
nothing hashed them, and nothing failed when they vanished.

## Four tiers

| tier | what | where | tracked? | integrity mechanism |
|---|---|---|---|---|
| **source** | executable production code | `+impl/` | yes | `SOURCE_MANIFEST.json` + `olhoffcurrent_currentness` — **untouched by this policy** |
| **metadata** | compact tracked scientific record: `REPORT.md`, `METRICS.json`, per-iteration CSVs, figures | the study directory | yes | `FINAL_SHA256.txt` |
| **evidence** | large raw scientific evidence: element-level trajectories | `analysis/OlhoffCurrent/evidence/<study>/` | **no** — declared instead | `EVIDENCE.json` + `olhoffcurrent_evidence_gate` |
| **scratch** | disposable intermediates | anywhere | no | none; never asserted |

The two integrity mechanisms are orthogonal and neither weakens the other.
Source integrity asks *"is production code what we recorded?"*. The evidence gate
asks *"is the scientific evidence still on disk?"*. The gate lives outside
`+impl/`, so it cannot change the canonical tree hash.

## Rules

1. **A frozen diagnostic must carry an `EVIDENCE.json`.** A study with no
   declaration fails the gate. "Declares nothing" is precisely the state the
   three lost studies were in, so it cannot be a passing state.

2. **Required evidence must be declared, hashed, and dimensioned.** For each
   artifact: repo-relative path, class, byte size, SHA-256, storage format,
   precision, and for MAT-files the variable names/sizes/classes read from the
   file itself.

3. **Declarations are measured, never asserted.** `olhoffcurrent_evidence_declare`
   hashes the file on disk and **refuses** to declare a `required` artifact that
   does not exist. You cannot freeze a study whose evidence was never produced.

4. **A missing or altered required artifact fails the gate.** Statuses are
   `REQUIRED_PRESENT_MATCH` / `REQUIRED_MISSING` / `REQUIRED_HASH_MISMATCH` /
   `OPTIONAL_*` / `SCRATCH`; the gate passes iff no required artifact is missing
   or mismatched. An unrecognised class **fails closed**, treated as required.

5. **Untracked is a storage decision; undeclared is a defect.** Raw evidence may
   live outside git when it is large. It must then be in the durable evidence
   root, deterministically named, SHA-256 manifested, and existence-checked.
   Solving a large-data problem by committing hundreds of MB of `.mat` to git is
   not acceptable either — one archive in this repository already exceeded
   GitHub's 100 MB hard limit and blocked a push.

6. **An empty evidence root is an honest failure, not a false alarm.** On a
   fresh clone the gate reports `REQUIRED_MISSING`, which correctly says "this
   machine cannot verify these studies", and `EVIDENCE.json` names exactly what
   is needed.

## Required evidence for a move/activity-class study

At minimum:

- full density trajectory, or a losslessly equivalent representation from which
  `rho(k)`, `rho(k-1)` and `Delta rho_e(k)` are exactly recoverable;
- move history `move(k)`;
- objective / eigenvalue history;
- `M_nd` history;
- stopping and transition telemetry;
- the resolved configuration;
- source provenance (`+impl` tree hash, MATLAB version, config hash).

Derived summaries — percentiles, active counts — are **not** a substitute. The
whole point is that future statistics can be computed without rerunning the
optimiser.

## Known residual exposure (not fixed by this policy)

A repo-wide sweep found **193 of 220** `.mat` files on disk untracked *and*
unmanifested. Most are historical and outside this implementation's scope, but
one deserves flagging: `examples/Performance/conference_benchmark/campaign_9mesh_r2/benchmark_records.mat`
(10.6 MB) is the archive `move_stop` verified its production baseline against
**bitwise**. If it disappears, that reproduction claim becomes unverifiable.

This policy covers `analysis/OlhoffCurrent` diagnostics. Extending it to the
performance-campaign archives is a separate, larger piece of work and is
recorded here rather than silently attempted.
