# Final iteration-efficiency harness

Run `iteration_efficiency_final('smoke','lp')` from MATLAB for an isolated
integration smoke. Valid Olhoff selectors are `lp`, `mma`, and `both`.

Production remains fail-closed until an independent final pre-production audit
updates the authorization gate. No authorization token is embedded or accepted
in this integration package. Smoke, qualification, and production outputs are
separated under `runs/`.
