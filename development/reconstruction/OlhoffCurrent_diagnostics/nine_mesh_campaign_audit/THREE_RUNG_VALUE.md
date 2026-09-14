# What the three-rung work bought

**THREE_RUNG_WORK_PARTIALLY_JUSTIFIED.** There is strong local computational value and meaningful diagnosis of legacy stopping. Nine-mesh generalization was never tested by the completed campaign.

## Single-factor removal of 0.005

These comparisons use the same E=A OR B controller up to S3. The only architectural difference is whether S3 terminates or descends. For 160/240/400, the three-rung endpoint is a causally valid stored prefix counterfactual; C320 also has an actual validated candidate CSV/record. Its raw candidate MAT is missing now, but the original four-rung raw prefix and all 52 candidate scientific CSV columns remain independently checkable.

| Mesh | Four-rung status | Old outer | Old inner | Old omega1 | Old Mnd % | Three outer | Three inner | Three omega1 | Three Mnd % | Δouter | Δinner | Δomega1 | ΔMnd points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 160x20 | CONVERGED | 219 | 5074 | 170.011 | 12.7041 | 180 | 4283 | 169.975 | 12.7561 | -39 | -791 | -0.0361955 | 0.0520205 |
| 240x30 | CONVERGED | 1358 | 44181 | 167.051 | 12.9425 | 284 | 5506 | 167.039 | 12.9165 | -1074 | -38675 | -0.0113781 | -0.0259621 |
| 320x40 | CAP_HIT | 1600 | 76532 | 166.427 | 12.9233 | 352 | 6498 | 166.426 | 12.9401 | -1248 | -70034 | -0.000392546 | 0.0167823 |
| 400x50 | CONVERGED | 505 | 10301 | 166.456 | 15.3311 | 466 | 8848 | 166.452 | 15.3732 | -39 | -1453 | -0.00397736 | 0.0421525 |


Exact post-update density endpoints are used, with frequencies from the next stored modal analysis. Relative scientific changes are tiny: removing 0.005 changes omega1 by at most about 0.022%, Mnd by at most 0.0521 percentage points, density L1 by at most 0.002121 and threshold-flip fraction by at most 0.139%. C320 saves 1248/1600=78.00% outer and 70034/76532=91.51% inner work. C240 saves 1074/1358=79.09% outer and 38675/44181=87.54% inner work. C160 and C400 save 39 outer and 791/1453 inner respectively. Those are real counterfactual savings, not cheaper computers or a mislabeled capped endpoint.

**Scientific effect:** nearly neutral relative to the four-rung E design. **Computational effect:** eliminate costly low-amplitude late-rung dynamics, dramatic at 240/320. Removal was justified on the tested meshes.

## Against original beta production

This is a broader policy comparison, not a single-factor 0.005 comparison. The physical formulation is common, but continuation/stopping differ; historical wall times are not directly comparable across MATLAB/host conditions. Work counts and density endpoints are informative.

| Mesh | Legacy outer | Legacy inner | Legacy omega1 | Legacy Mnd % | Three outer | Three inner | Three omega1 | Three Mnd % | Δouter | Δinner | Δomega1 | ΔMnd points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 160x20 | 91 | 2241 | 169.495 | 13.4025 | 180 | 4283 | 169.975 | 12.7561 | 89 | 2042 | 0.479893 | -0.646357 |
| 240x30 | 104 | 2334 | 167.07 | 15.6036 | 284 | 5506 | 167.039 | 12.9165 | 180 | 3172 | -0.0310137 | -2.68712 |
| 320x40 | 131 | 2614 | 165.951 | 23.3596 | 352 | 6498 | 166.426 | 12.9401 | 221 | 3884 | 0.475515 | -10.4195 |
| 400x50 | 139 | 2918 | 162.889 | 32.3283 | 466 | 8848 | 166.452 | 15.3732 | 327 | 5930 | 3.56352 | -16.9551 |


Three-rung costs **more** than prematurely stopped beta production. It improves C400 frequency by about 2.19% and reduces Mnd from 32.33 to 15.37%; C320 Mnd falls from 23.36 to 12.94% with a smaller objective gain. At C240 objective is essentially unchanged but the density field is cleaner. Thus the work did not merely deliver faster versions of the original production runs. It delivered more mature designs at additional cost, then removed waste from the longer E-controller solve.

## Answers A–F

A. Relative to four-rung E, no material scientific redesign; relative to legacy, meaningful density improvements and C400 objective improvement.

B. Yes: removing 0.005 preserves the useful E-controller design while eliminating waste on all four known meshes.

C. No material sacrifice is observed against four-rung E. Fine-mesh early-stop sacrifice by three-rung remains untested. The actual legacy campaign is strongly exposed to early stopping.

D. Unknown beyond C400. Legacy cost scaling is not a measurement of three-rung savings.

E. The finer legacy meshes reveal stronger grayness and a large C800 regime change, but not a new three-rung mechanism. The campaign identity error predates every mesh.

F. Two precise simpler alternatives have evidence. First, after the same S1 E declaration, **39 updates at 0.02 then 39 at 0.01, then stop** reproduces the historical three-rung endpoints on all four meshes because both late stages reach their earliest legal E declaration. This is retrospective equivalence, not prospective validation. Second, two-rung E stopping at S2 is close but C160 remains in branch A with amplitude 2.19×tol; the historical preregistered 0.10% objective materiality gate rejected deleting both remaining rungs because the gain to the four-rung endpoint was 0.114%. Relaxing that bar after observing it would be post-hoc rescue. Plain fixed move=0.04 is not universal: C160 hits its cap, whereas C240/C400 have coherent amplitude decay. Stored fixed-move summaries and limits are in [FIXED_MOVE_EVIDENCE.csv](FIXED_MOVE_EVIDENCE.csv).

The available fixed-move reports support different terminal regimes, but several raw histories are missing or have stale container declarations. They do not support a new fine-mesh counterfactual or a claim that one scalar stagnation rule would work everywhere.
