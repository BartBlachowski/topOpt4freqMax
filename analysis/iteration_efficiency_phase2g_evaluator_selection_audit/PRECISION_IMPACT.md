# Precision impact

Stored paired evidence covers 708 evaluator/state records (24x4 and 96x12,
double versus float32 images). Under C, selected ordinals changed in 0 records, hard-gate
decisions changed in 0 records, and the maximum selected-frequency relative change was
5.596e-08.

This is encouraging but not a qualification. The paired samples do not contain the complete
3,200-state density sequence needed to verify `b_ref`, `B_meas`, acceptance, `k_enter`, or
`k_cert`. A fresh post-refreeze qualification is required and must bind its artifact to the
new evaluator and contract hashes.

The Phase-2B single-versus-double negative result remains historically valid for the
Eq.(4) frozen evaluator and its discontinuity mechanism. C does not retroactively invalidate
that experiment, and it does not prove the downstream C decisions precision-invariant.
