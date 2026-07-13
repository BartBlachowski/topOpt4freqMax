# Revision_v1 execution plan

Scope: the experiments still required after (a) the `OlhoffApproachExact` migration and
(b) the **EXP1/EXP5 obsolescence decision** ([SCIENTIFIC_DECISION_EXP1_EXP5.md](SCIENTIFIC_DECISION_EXP1_EXP5.md)).

Everything under `archive/` is excluded and must not be rerun, cited, or re-entered into a
manifest.

Authoritative formulation for every stage below:

```
F(x) = omega0^2 * M(x) * Phi0      reference design = solid, frozen
```

Acceptance gate applied to every run (declared before execution):
not capped **and** design_change <= tol **and** feasibility <= tol **and**
tracked MAC >= 0.8 **and** A5 lowest-mode check recorded **and** all artifacts present.
**A run that reaches the iteration cap is a failure, not a result.**

## Retired stages (not part of this campaign)

| Stage | Status | Reason |
|---|---|---|
| **EXP1** performance table | **RETIRED — governed archive** | Supports zero surviving manuscript claims. Construct-invalid as a benchmark: the local Olhoff comparator performs roughly twice the required eigensolves (Heaviside + grayness penalty + trial eigensolve per MMA update), and the local Yuksel comparator does not reproduce published iteration counts. Instrumentation cannot repair a validity defect. Preserved in `archive/obsolete_evidence/exp1_exp5/`. |
| **EXP5** scaling fit | **RETIRED — governed archive** | Consumed EXP1 only, and its `O(n_e^1.3)` claim is already withdrawn from the manuscript. Preserved in `archive/obsolete_evidence/exp1_exp5/`. |
| **EXP4** sensitivity ablation | RETIRED (earlier) | Pre-authoritative load; superseded by CR2. **Not** an A4 implementation. |

Consequence: **no cross-code performance, timing, memory, or scaling stage remains
active.** Comparator telemetry is therefore not required and is not implemented.

## Runtime basis

`measured` = taken from a saved artifact. `estimated` = scaled from a measured run by
element count and iteration count. Estimates are planning figures, not results.

## Stages still required

| Stage | Current status | Reusable? | Needs rerun? | Reason | Estimated runtime | Dependencies |
|---|---|---|---|---|---|---|
| **I1** smoke (fail-loud gate) | ACTIVE_VALID | **Yes** | No | `output/smoke/i1_smoke_result.json` = PASSED. The runner correctly fails loud on a capped run and names the failed condition. | ~1 min (measured) | none |
| **S1** low-mode mitigation | ACTIVE_NEEDS_RERUN | Partly | **Yes** | The mitigation *run* passed its gates (1579/2000, MAC 0.973) but the *scientific goal* failed: 9 of 10 modes remain localized low-density modes. `pmass=6` is not a working mitigation. A documented alternative (RAMP / Heaviside / higher void-mass penalization) must be compared, or the no-spurious-mode claim must be dropped. | ~3 h (estimated) | none — **gates EXP2b and EXP3** |
| **EXP2** clamped-beam alpha sweep | ACTIVE_NEEDS_RERUN | No | **Yes** | 0 of 5 alphas accepted. α=1.00 converged (1052 it, MAC 0.992) but α=0.75 and α=0.00 "converged" at **iteration 1 with grayness = 1.0** — degenerate; the optimizer never left the solid reference design. α=0.50 mode-invalid (MAC 0.748). α=0.25 **capped 2000/2000**. The α=0.75 non-monotonicity demand (CR1) remains unresolved. | ~3 h (estimated) | S1 |
| **EXP2b** building benchmark | ACTIVE_NEEDS_RERUN | No | **Yes** | α=1.00 and α=0.75 **capped 2000/2000**. Many of the first ten topology modes have max MAC < 0.01 → the "no spurious low-density modes" claim is unsupported and must be retracted or evidenced via S1. | ~2 h (measured legacy: 6 542 s) | S1 |
| **EXP3** mesh convergence | ACTIVE_NEEDS_RERUN | No | **Yes** | 400x50 is **mode-invalid** (MAC 0.786 < 0.8). Relative tracked-ω change between meshes = **0.546** against a declared threshold of 0.05; topology correlation = −0.088. Currently this **contradicts** mesh convergence. | ~6 h (estimated) | S1, EXP2 |
| **CR2** omitted load sensitivity | ACTIVE_NEEDS_RERUN | No | **Yes** | Every attempt failed. Production: both variants **capped 400/400**, claim `withheld`. Rerun: both capped. MMA diagnostic: both capped — which **rules out OC as the cause**. Needs the authoritative load, a raised cap, and an FD check on Variant B. **Not a runner stage** (see blockers). | ~3 h (estimated) | authoritative load; raised cap |
| **A4** eigenpair-refresh study | **NOT IMPLEMENTED** | No | **Yes — must be written first** | No A4 script or config exists anywhere. The plan requires N = {1, 5, 10, 50, ∞} on SS 400x50; only a single N=50 variant was ever run (inside the retired EXP4, unconverged). This is the evidence for the frozen-eigenpair reliability claim, and it has **zero artifacts**. It also now carries the accuracy question formerly proxied by EXP1, and can supply a *legitimate* efficiency number (eigensolves/iteration, frozen vs refreshed) with no unfaithful comparator. | ~16 h (estimated) | new script + configs |

**Estimated total compute: ~33 h** (was ~48 h before EXP1/EXP5 were retired).

## Critical path

**A4 is the new critical path.** It is both the longest single stage (~16 h) and the only
mandatory stage with *no implementation and zero artifacts*. It is also independent of the
S1 → EXP2 → EXP3 chain (~12 h), so it should be started first.

```
A4        [not implemented]  ================================  ~16 h   <-- CRITICAL PATH
S1 -> EXP2 -> EXP3           ==========================        ~12 h
S1 -> EXP2b                  ==========                        ~5 h
CR2       [independent]      ======                            ~3 h
```

## Recommended order

1. **A4** — implement, then run. Critical path; zero evidence today.
2. **S1** — settle the localized-mode mitigation. EXP2b and EXP3 keep failing the MAC gate
   at fine meshes until this is resolved.
3. **CR2** — cheap, and it decides whether the omitted-sensitivity claim survives.
4. **EXP2 → EXP3** — the clamped-beam chain, once S1 is settled.
5. **EXP2b** — building, once S1 is settled.

## Blockers that must be cleared before launching

1. **The two proposed patches must be applied, in order** — `proposed/stage_rewiring.patch`
   then `proposed/acceptance_gates.patch`. Until then the master runner still dispatches
   EXP2/EXP3 to pre-authoritative scripts and still references the now-archived
   `exp1_perf_table` / `exp5_scaling`. **The runner will not work until it is patched.**
2. **A4 does not exist.** Preflight P3 blocks any `full`/`fast` campaign until it does.
   Individual stages remain runnable via `stage` mode.
3. **CR2 is not a runner stage.** `run_cr2_production()` takes no `outDir` and writes to a
   hardcoded path, so registering it would require changing the experiment's signature.
   It must be run and cited manually, under archive governance.
4. **`run_all:OutputConflict`.** Keep stage directories empty before each launch; preflight
   P4 now reports conflicts for all stages up-front rather than after hours of compute.
