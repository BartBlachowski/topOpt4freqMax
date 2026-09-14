# Provenance and limits

Preregistration SHA-256: `3a80b05bae0283a77aeae94aabe21f7556e7fff9f52e98940c74f61e61613ee7`.
It was written before any new solver experiment. Upstream archive SHA-256:
`df843fdbf40679383f776f4cc4118c541129249e32c209df0cb15637ea3022fc`. Source: https://www.smoptit.se/GCMMA-MMA-code-1.5.zip.
Its licensing files are retained; no email was sent. Repo HEAD: `013cc48451d33bed61c5c4eea174bbd898d548a2`.

Zero topology runs, zero outer updates, zero accepted rho changes, zero
controller transitions, zero production/config/reference modifications. All
new outputs are inside this diagnostic directory. The newly added Pedersen PDF
outside the audit was supplied by the user. The final protected-file hash audit
covers 1817 original files, including every tracked regular
file, the reference study and the authoritative trajectory. It is recorded in
integrity_final.json; FINAL_SHA256.txt hashes final deliverables.

MATLAB R2025b at /Applications/MATLAB_R2025b.app; one numerical thread per process.
Python/numpy/scipy/matplotlib/h5py generate tables and scientific plots. No new
package installation, FE eigensolve, optimization mesh, or performance campaign.
Independent solver processes share the host; wall times are not controlled
benchmark timings. Only audit-local launch permissions were escalated after the
sandboxed MATLAB launcher silently exited. The official archive download needed
network permission. No approval rejection occurred.

## Disclosed implementation and procedural details

- The initial audit logger failed after one instrumented MMA call because
  MATLAB rejected an assignment into an empty fieldless struct. Production-19
  had already reproduced. Explicit initialization fixed the logger; the valid
  500-call run restarted from zero. Aborted log is retained. No solver formula
  or threshold changed because of that failure.
- GCMMA's nine printed toy iterations reproduce the published rounded values,
  but their KKT residual is 1.447e-5. The validation continued to its KKT bar;
  it did not loosen the bar. The initial failed nine-iteration assertion log is
  retained. This does not invalidate the subsequent analytic validation.
- The GCMMA validation completed while the independent baseline was running,
  before any frozen GCMMA experiment. Some independent prescribed tests overlap
  in wall time; the report's causal comparisons follow the registered factors.
- The previous 5000-call replay is authenticated/re-evaluated rather than wholly
  recomputed, as preregistered. New B0 extends to 500 and proves bitwise fidelity.
- Oracle-witness logging was refined from a conservative rounded prior bound
  to its explicit formula; this changes only a diagnostic at roundoff level.
- Raw per-iterate metric-event counts were converted to actual production-
  evaluator calls in final outputs; algorithm work and audit overhead are
  distinguished. This affects accounting only.
- S4 hit its registered 1800-second budget after 455 calls (1803.726s,
  checked between calls). The 500-call checkpoint is missing and is not
  manufactured or used for a strong-effect claim.
- Tight subsolv accuracy produces near-singular-system warnings; logs retain
  them. They are not suppressed or mistaken for successful convergence.
- The user supplied Pedersen (2000) after preregistration. It prompted no new
  variants. Barrier and first-call model arithmetic checks and exact gray-bound counts are read-only
  mechanistic diagnostics, not extra optimization experiments.
- The prompt's 'every gray element' bound claim is corrected from the actual
  oracle. None of its hashes, objective, constraints or certificates failed.

Production volume arithmetic temporarily evaluates the sum of rho and the
increment to preserve exact bits. No candidate density vector is saved, accepted,
or passed to FE/model/controller functions. innerLoopRho is inspected but never
executed. All resulting increment arrays are evidence only.
