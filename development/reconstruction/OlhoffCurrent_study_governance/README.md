# OlhoffCurrent study governance (archived 2026-09-14)

Study-finalization tooling that lived in `analysis/OlhoffCurrent/` beside the production code:

* `olhoffcurrent_finalization_gate.m`, `olhoffcurrent_evidence_gate.m`,
  `olhoffcurrent_evidence_declare.m`, `EVIDENCE_POLICY.md`
* `tests/test_finalization_gate.m`, `tests/gate_provenance_probes.m`,
  `tests/test_evidence_retention.m`

It exists to freeze and verify the studies in `../OlhoffCurrent_diagnostics/` against
their raw evidence in `../OlhoffCurrent_evidence/` and against git objects at
`analysis/OlhoffCurrent/...` in historical commits. No production script, benchmark file or
other test called it, so it was archived with the studies it governs.

**It is path-bound to the pre-cleanup layout** (`analysis/OlhoffCurrent/{+impl,diagnostics,evidence}`)
and will not run from here. To use it, check out the commit before the cleanup; to use it
for new studies, promote it back into `analysis/Olhoff` and re-point its paths.
