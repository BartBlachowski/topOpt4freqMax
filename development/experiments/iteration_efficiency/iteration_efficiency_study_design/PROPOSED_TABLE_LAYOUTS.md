# Phase 1C paper and supplementary layouts

No main count table may imply that fewer native updates means a better method. R is a
q-dependent self-referenced maturation result; achieved/reference quality is inseparable
from it.

## Main Table 1 — quality–effort landmarks

Use long-form method blocks, not one unmanageably wide mesh grid.

| Mesh | Method | q | robust `k_enter` | robust `k_cert` | native stop | method-gate k | chronological/stage decomposition | E1/E2/E3 quality at enter | E1/E2/E3 ratio to own ref | ratio to best-observed | status + subclass |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|

Rows q=98%, 99%, 99.5% are co-primary. For Yuksel the decomposition is `S1`, `S2 to
endpoint`, and `chronological method-level updates (S1+S2)`. For Olhoff the count is outer
updates; normally redundant LP-call counts move to a footnote. Proposed rows state that
an OC update has no per-loop eigensolve. Every single-number excerpt must retain q,
enter/cert, absolute quality, and status.

Add compact E1/E2/E3-only endpoint columns or an immediately adjacent Table 1B in the
same visual block. No evaluator is called a sensitivity. Mark `MODEL_DEPENDENT` if
evaluator choice changes status/order.

The mandatory best-observed status/ratio appears in Table 1. It is never A or an absolute
engineering requirement. If an independently provenance-locked `Omega_req` exists, put A
in a separate table; otherwise print `A_NOT_INSTANTIATED`.

## Main Table 2 — reference and endpoint quality/validity

| Mesh | Method | reference freeze k | Qref E1 | Qref E2 | Qref E3 | reference stability gains | best-observed ratios | raw relative volume | support path | largest detached physical area | aggregate detached area | n_islands_all | A_sig | method validity | reference/status |
|---|---|---:|---:|---:|---:|---|---|---:|:---:|---:|---:|---:|---:|---|---|

This table exposes absolute quality. Its caption states the completed-campaign fact that
Olhoff led Proposed by 6.2–8.5% and Yuksel by 5.9–7.7% in common raw-E1 `omega1` across the
eight meshes with a complete method triple. The Olhoff 800x100 endpoint is `RUN_ERROR`/E1
`N/A` and is excluded rather than inferred. Do not describe the methods as equal/comparable
quality.

## Main Table 3 — computational support

| Mesh | Method | q | mean native-update s to enter | median `T_enter` | median `T_cert` | replay range | stage-specific means | platform |
|---|---|---:|---:|---:|---:|---|---|---|

Reference/calibration update count and time are a separate block, never added to endpoint
time. Deduplicate equal q endpoints but retain all q labels. Timing is secondary and
reference-platform-specific.

## Main figure package

1. E1/E2/E3 quality and sustained floors versus method-level updates, with q landmarks.
2. Robust `k_enter` versus `Ne` at q=98/99/99.5, with evaluator component panels.
3. Mean native-update cost versus `Ne`, adjacent to count curves.
4. Absolute E1/E2/E3 reference/endpoint quality and ratio-to-best versus `Ne`.
5. `T_enter` by q plus companion `T_cert` and separate reference cost.
6. Yuksel Stage-1/Stage-2/chronological decomposition.
7. Olhoff outer/LP work decomposition (mandatory supplement; genuine solver iterations only).
8. Standardized accepted/last-observed topology grid.
9. Descriptive `k_cert` panels carrying the `+99` convention warning.

Bindings are in `SCALING_AND_FIGURE_SPEC.md`.

## Supplement S1 — complete evaluator endpoints

| Mesh | Method | q | evaluator | `k_enter` | `k_cert` | Q at enter/cert | Qref | ratio | robust endpoint | ordering/status |
|---|---|---:|---|---:|---:|---|---:|---:|---:|---|

## Supplement S2 — fit table

| Quantity | q | evaluator | endpoint/stage | method | support | C | p | R2_log | n_valid | p_LOO range | Ne range | weakly identified | exclusions |
|---|---:|---|---|---|---|---:|---:|---:|---:|---|---|:---:|---|

Full-range and common-support fits occupy adjacent rows. Cross-method comparison is
forbidden for full-range rows with unequal support.

## Supplement S3 — complete accounting and gate floors

| Mesh | Method | q | S1 | S2 enter/cert | chronological enter/cert | outer enter/cert | LP calls | native stop | method-gate k | attempted updates | status/subclass | budgets |
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---|---|

Known Olhoff solver terminations always carry
`GENERIC_LP_ITERATION_LIMIT_ONLY (dual-simplex-highs, recorded MATLAB version)`.

## Supplement S4 — Olhoff backend work

| Mesh | q/endpoint | outer | LP calls | exit flags | genuine solver iterations total/mean/median/max | eig/gradient/LP time |
|---|---|---:|---:|---|---|---|

`nInner=1` is archived only as one LP call and never fills the solver-iteration column.

## Supplement S5 — OAT and reference diagnostics

Rows cover P=50/100/200, topology 1x1/2x2/3x3 physical FE scales, volume tolerance,
Olhoff gap, and the superseded horizon-relative references. Include `F_e(b)`, stabilization
index/gains, q-level endpoints, status/order changes, and no Cartesian preferred scan.

## Phase 2H mandatory columns

Every evaluator evidence table adds Candidate/classifier version, evaluator status,
selected ordinal/frequency for E1/E2/E3, final requested mode count, escalation count,
`voidKE`, `voidSE`, `densityParticipation`, IPR, and minimum classifier margin. Binary
frequency is shown only in a clearly labelled endpoint-diagnostic column. Method tables
use distinct `Olhoff-LP` and `Olhoff-MMA` rows and the route-specific work columns defined
in `ITERATION_ACCOUNTING_SPEC.md`.
