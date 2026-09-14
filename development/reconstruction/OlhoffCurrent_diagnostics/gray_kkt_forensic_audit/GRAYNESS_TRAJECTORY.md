# Trajectory of grayness

| mesh | stage | iterations | Mnd start % | Mnd end % | gray end % | mid end % | broad core end % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 400 | 1 | 1–388 | 99.686 | 15.6649 | 18.22 | 3.98 | 0 |
| 400 | 2 | 389–427 | 15.6396 | 15.4413 | 18 | 3.93 | 0 |
| 400 | 3 | 428–466 | 15.428 | 15.3732 | 17.96 | 3.92 | 0 |
| 480 | 1 | 1–308 | 99.7472 | 26.8099 | 28.9097 | 13.1944 | 13.1875 |
| 480 | 2 | 309–347 | 26.7827 | 26.4442 | 28.7222 | 12.0486 | 13.0347 |
| 480 | 3 | 348–386 | 26.4338 | 26.3416 | 28.7292 | 11.8611 | 13.0139 |
| 800 | 1 | 1–390 | 99.9599 | 35.2626 | 38.2275 | 17.135 | 21.75 |
| 800 | 2 | 391–429 | 35.2145 | 34.6671 | 37.88 | 16.3125 | 21.0575 |
| 800 | 3 | 430–468 | 34.644 | 34.4123 | 37.7625 | 16.09 | 20.87 |

The initial field is rho=.5 everywhere: all elements start gray/mid. It is misleading to ask when gray material is first “generated” without this fact. What changes is which regions become discrete and which persist as broad gray patches. 400 resolves its broad cores away; 480/800 retain broad end patches by stage-1 end. The final-stage grayness is largely inherited from stage 1. Stages 2/3 decrease Mnd and mid/broad fraction slightly; they do not create the mesh-growing gray phase.

Mnd stays within .1 percentage point of the endpoint after iterations 419/348/441 at 400/480/800. Thus it is not literally frozen hundreds of iterations before termination, especially at 800. It is nevertheless very flat in the final 20 iterations: Mnd ranges .03196/.03554/.05919 points; pre-update omega1 ranges .00995/.00464/.01289%. Small residual evolution is not evidence of local KKT convergence.

F13/F14 plot omega1, Mnd, gray/mid/broad fractions, move, A/B/E, gap12/gap23 and warnings, with stage boundaries. F15 combines warning/stage/grayness timelines. F20 snapshots show warning-end and stage-end densities in identical coordinates. CSVs preserve pre-update frequency labeling; saved densities are post-update. Raw histories, final-patch histories and stage statistics are retained.
