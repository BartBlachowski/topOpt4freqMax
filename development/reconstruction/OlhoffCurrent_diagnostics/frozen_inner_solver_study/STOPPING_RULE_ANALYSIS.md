# Offline stopping diagnosis

STOPPING_RULE_NOT_A_VALID_OPTIMALITY_PROXY

| Method | Relative step | Objective gap | KKT | Distance | Feasible | Active stable 20 | Joint fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0_CURRENT_REPEATED_MMA | 19 | never observed | never observed | never observed | 1 | 21 | never observed |
| S1_PERSISTENT | 19 | never observed | never observed | never observed | 1 | 21 | never observed |
| S2_ASYINIT_001 | 33 | never observed | never observed | never observed | 1 | 21 | never observed |
| S3_CANONICAL_CLAMP | 22 | never observed | never observed | never observed | 1 | 21 | never observed |
| S4_SUBSOLV_ACCURACY | 16 | never observed | never observed | never observed | 1 | 160 | never observed |
| S5_UNIT_BOX | 17 | never observed | never observed | never observed | 1 | 21 | never observed |
| G0_UNSAFE | 23 | never observed | never observed | never observed | 1 | 21 | never observed |
| G1_GCMMA | 23 | never observed | never observed | never observed | 1 | 21 | never observed |
| G2_GCMMA_ACCURATE | 33 | never observed | never observed | never observed | 1 | 111 | never observed |
| S34_CLAMP_ACCURACY | 35 | never observed | never observed | never observed | 1 | 86 | never observed |
| B0_RETAINED_5000 | 19 | never observed | never observed | never observed | 1 | never observed | never observed |


Sparse retained first hits are only upper bounds on event time, not exact
iteration timestamps. Active stabilization is unavailable between retained
checkpoints. KKT first hits here use the preregistered scalar bars; feasibility,
objective/design distance and all remaining KKT bars are required by joint fidelity.

B0 first stops at 19, with d2=0.994774 and gain recovery=0.277%.
Its fresh-500 relative-step/d2 Pearson correlation is
0.351437, Spearman 0.766970;
these trend correlations do not validate optimality. Every relative-stop hit
in B0 fails joint fidelity. The retained 5000 endpoint has step=2.077e-03
but d2=0.829748, KKT=8.752e-02, and bound agreement=0.
B0's gain decreases 210 times over 500 new iterations;
small relative steps coexist with nonmonotone objective progress.

For absolute oracle objective gap, fresh-B0 Pearson correlation with relative
step is 0.414661 and Spearman is
0.842190. The production step criterion holds at
482 of 500 recorded iterates,
with a longest consecutive streak of
482; joint
fidelity holds at zero. METRICS.json records hit counts, terminal status and
longest consecutive streaks for every diagnostic and method. Sparse retained
records cannot establish consecutive-iteration persistence and are marked null.

Lowering tolInner alone did not solve the known 5000-call replay and is not
shown capable of meeting fidelity. No broad tolerance sweep was run. The fixed
subsolv barrier floor provides an independent accuracy limitation, so merely
increasing the number of repetitions is not a demonstrated cure. This is not a
claim that every scalar trend correlation is zero.
