# Migration note (2026-09-14)

These studies were `analysis/OlhoffCurrent/diagnostics/<study>` and their raw evidence was
`analysis/OlhoffCurrent/evidence/<study>` (now `../OlhoffCurrent_evidence/`). The production
tree they were run on is now `analysis/Olhoff` (function names unchanged). Paths inside
the studies — `EVIDENCE.json` `evidenceRoot`, `FINAL_SHA256.txt`, scripts — are the
historical ones and were not rewritten; file contents are byte-identical.
