# Direct SOCP computational scale

| Repeat | Assembly s | Solve s | Certification s | Iterations | Certified gap | Bitwise oracle |
|---|---:|---:|---:|---:|---:|---|
| 1 | 0.1916 | 45.5851 | 1.2197 | 27 | 9.153e-12 | True |
| 2 | 0.0030 | 45.4560 | 0.8108 | 27 | 9.153e-12 | True |


The x-coordinate/schur/1e-10 configuration is exactly the one chosen in the
reference study; no new tuning was performed. Both repeats satisfy every
preregistered fidelity criterion. Assembly excludes loading the authoritative
trajectory and building the shared model/filter objects.

The process containing both repetitions plus read-only structural diagnostics
peaked at 1,375,125,504 bytes RSS (1.281 GiB), measured by `/usr/bin/time -l`.
This is total MATLAB process memory, not an isolated cone-factorization estimate.
Per-repetition assembly-object byte estimates are in SOCP_COST.json.

coneprog exitflag remains -7. Its returned dual residual is not acceptable;
the separately verified conic dual witness supplies a valid bound and exact KKT
certificate. Both returned-dual and certificate residuals are recorded. Success
is based on the latter's checked feasibility/complementarity/gap, not the exitflag.

These are indicative costs on a shared host with concurrent audit processes.
They are not a nine-mesh performance benchmark. No solver performance tuning.
