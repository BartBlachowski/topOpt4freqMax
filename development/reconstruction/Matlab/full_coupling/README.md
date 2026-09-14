# Generation 2 — Rebuilt full-coupling OlhoffExact implementation

**This directory intentionally contains no code.**

The rebuilt full generalized-gradient / LMI implementation was **not moved**
during the 2026-08-26 reproduction migration. It remains at its original
location:

> ### [`analysis/OlhoffApproachExact/Matlab/`](../../analysis/OlhoffApproachExact/Matlab/) (top level)

Generation 1 lives in the `legacy/` subfolder *of that same directory*; the two
are distinguished by nesting, not by repository location.

## Why it was not moved

`OlhoffApproachExact` is referenced from 18 places, including
`tools/Matlab/run_topopt_from_json.m` (the `OlhoffExact` dispatch case), the
frozen `examples/Performance/ledger/protocol_ledger.json`, and the `ablations`,
`step_calibration` and `terminal_direction_audit` experiment trees. Moving it
mid-revision would churn all of them for no scientific gain. See the
"Why the layout is not literally ..." section of [`../README.md`](../README.md).

## Status

**Active.** This is the implementation the planned parametric study runs
against.

Entry points at the real location:

- `topopt_freq_exact.m` — the solver
- `run_olhoff_case.m`, `run_all_olhoff_2014.m`, `run_{cc,cs,ss}_n{1,2}.m`
- `subproblem_lp.m`, `subproblem_mma.m`, `subproblem_kkt.m`
- `verify/v_*.m` — forward model, sensitivities, multiplicity, basis invariance

Through the benchmark: `optimization.approach = "OlhoffExact"`.

## Relationship to generation 3

The clean-room reproduction at [`../reproduction2007/`](../reproduction2007/)
is a **separate implementation**, not a replacement and not an upgrade path.
Do not merge the two, do not replace this FE code with the reproduction's, and
do not refactor them onto shared numerical helpers while the paper revision is
open.
