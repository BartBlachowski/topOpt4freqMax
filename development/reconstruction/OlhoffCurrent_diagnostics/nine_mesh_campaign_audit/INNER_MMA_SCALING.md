# Nested MMA behavior

The nine-mesh observations below are the **legacy beta/four-rung campaign**, not a three-rung E-controller campaign. This distinction applies to every numerical table and plot unless explicitly labelled historical E-controller evidence.

| Mesh | Outer | Inner total | Inner/outer | Reported nonconverged | MMA share % |
| --- | --- | --- | --- | --- | --- |
| 160x20 | 91 | 2241 | 24.6264 | 0 | 97.5973 |
| 240x30 | 104 | 2334 | 22.4423 | 0 | 96.8446 |
| 320x40 | 131 | 2614 | 19.9542 | 0 | 95.5133 |
| 400x50 | 139 | 2918 | 20.9928 | 0 | 95.1925 |
| 480x60 | 164 | 3463 | 21.1159 | 0 | 94.7912 |
| 560x70 | 190 | 3922 | 20.6421 | 0 | 94.4739 |
| 640x80 | 199 | 4324 | 21.7286 | 0 | 93.9771 |
| 720x90 | 223 | 4831 | 21.6637 | 0 | 91.1351 |
| 800x100 | 170 | 3713 | 21.8412 | 0 | 90.087 |


Cumulative counts grow 2241→4831 through 720 then fall to 3713 at 800. Mean inner/outer falls from 24.63 to approximately 20–22, so there is no aggregate sign of growing nested iteration difficulty. Dimension-dependent per-inner cost grows from 0.05216 to 0.48211 s. The optimizer is still the dominant wall-time expense, despite FE/eigensolve growth. Cumulative fine-mesh work is driven by both outer trajectory length and more expensive inner steps, not an observed explosion in inner solves per outer.

Every actual row reports zero nonconverged inner solves; the driver independently maps any positive count to SOLVER_FAILURE. Per-outer maxima, histograms, stage distributions and isolated spikes cannot be recovered from these aggregates. The known maxInner=500 is a cap, **not an observed maximum**. No such maximum is inserted into MASTER_TABLE.csv.

For the retained historical E-controller, [HISTORICAL_STAGE_WORK.csv](HISTORICAL_STAGE_WORK.csv) supplies stage counts, means, medians, p90, maxima and failures. Stage 3 mean inner counts are 16.31/20.85/19.13/20.79; maxima 25/35/34/35; no failed inner solves. Stage 4 at C240 takes 1074 outer and 38675 inner (mean 36.01, max 97); C320 takes 1248 outer and 70034 inner (mean 56.12, max 139). C160/C400 stage 4 costs only 39 outer each. This is the directly observed late-rung pathology that removal addresses.

The historical C320 cap was reached despite every inner solve meeting its own convergence criterion. Inner success does not imply outer maturity. Conversely, no evidence permits saying the five fine three-rung runs avoid this pathology: those runs are absent.
