# AUDIT PREREGISTRATION — scientific_delta_olhoff_migration

Frozen before any trajectory analysis, same-state evaluation or optimization run.
Its SHA-256 is recorded in `evaluations/preregistration_sha256.txt` immediately after
writing. Criteria below are not changed after results are seen; any unavoidable
deviation is recorded as a numbered amendment with its own hash and reason.

Work performed before freezing (identity only, no scientific computation): recording
repository identities; `git archive` snapshot and blob verification; reading source and
target code, presets, run configs (`describe.txt`) and prior target reports; listing the
*variable names* inside retained `.mat` containers. No trajectory values beyond those
already printed in the task brief and in committed markdown tables were analysed.

---

## 1. Identities under audit

| | |
|---|---|
| Source repository (read-only) | `/Users/piotrek/Programming/Matlab/Olhoff` |
| Source commit | `6b0870850d7407a0f2fc31dbf0d0f318c2e5f2f7` ("Nine resolution test - ultimate results", branch `repro/natural-convergence`, parent `695f03bdac20c423a4e1d389cf9db9187597bcc3`, tree `809c671e2d1b15232f8b75ca0f07237c0e8e37ac`) |
| Target repository | `/Users/piotrek/Programming/topOpt4freqMax`, branch `benchmark-methodology-r2`, HEAD `013cc48451d33bed61c5c4eea174bbd898d548a2` |
| Target dirty state at start | 7 untracked diagnostic directories under `analysis/OlhoffCurrent/diagnostics/` (all pre-existing, preserved untouched) |
| Target implementation | `analysis/OlhoffCurrent/+impl`, manifest tree `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (75 files) — to be verified in Part 2 |
| Audit branch | none created. The dirty state consists only of untracked diagnostics; switching branches is unnecessary and the audit stays uncommitted for review. |

## 2. Source snapshot procedure

1. `git archive --format=tar <commit>` of the full tree is streamed to SHA-256 (not stored).
2. `git archive --format=tar <commit> -- <subset>` with subset = `.gitattributes .gitignore
   CLAUDE.md NOTES.md EVIDENCE_MANIFEST.sha256 OLHOFFEXACT_FAILURE_POSTMORTEM.md setpaths.m
   top88.m algo fem filter mma mma_published architecture/{+olh,olhoffSolve.m,README.md,docs,
   legacy,tests,anchors/code} repro` is hashed and extracted.
3. Extracted under `source_snapshot/+olhoff_6b08708/`. The `+` prefix makes the tree invisible
   to `genpath`, so `addpath(genpath(<repo>/analysis))` elsewhere in the repository can never
   put a second Olhoff implementation on a production MATLAB path.
4. Every extracted file is re-hashed as a git blob and must equal the committed blob id
   (`scripts/sd_snapshot_manifest.py`). Excluded (not scientific evidence for this audit):
   legacy `audit_*`, `results/`, `runs/`, `docs/` PDFs, `architecture/anchors/{candidate,reference}`.
5. The snapshot is never edited. All executions of source code read from it.

## 3. Difference categories and classes

Categories: the 43 categories of the task brief (stiffness interpolation … timing/reporting),
each row of `SCIENTIFIC_DELTA_TABLE.md` / `MASTER_DELTA.csv` carrying: category, source
behaviour, target behaviour, identical?, scientifically material?, paper-specified?,
reconstruction choice?, likely causal?, evidence, class.

Classes (a row may carry several): **A** paper-specified fidelity; **B** standard-method
implementation; **C** reconstruction choice; **D** new scientific formulation; **E**
software/reporting only; **F** potential bug fix. Rules: a change whose effect is to alter the
relaxed material problem, the objective, the constraints or the filter operator is **D**
unless the paper prescribes exactly the target's form (then the target is the fidelity
reference). A change is **F** only if the target behaviour is shown internally inconsistent or
incorrect independent of whether results improve. Better results never promote C/D to F.

"Paper-specified" means printed in Du & Olhoff (2007) or its erratum. Pedersen (2000) being
*named* in Du & Olhoff §2.2 as an alternative counts as "paper-named alternative", not as the
paper's chosen formulation.

## 4. Semantic comparison method

1. Three-way file comparison: upstream base `695f03b` (target's promoted base) vs target
   `+impl` vs source snapshot. Every differing hunk is assigned to a category.
2. Effective-configuration comparison in MATLAB: every leaf of the canonical configuration for
   (i) source S480x60 as stored in its `res.mat`; (ii) source `duOlhoffAdaptiveMove` +
   `move.initial=0.10` at 480×60 resolved from the snapshot (M1, §6); (iii) target C480 canary
   config stored in its trajectory container; (iv) target canonical production
   (`olhoffcurrent_config(480,60)`).
3. Same-state operator comparison (§8) decides identity of kernels numerically, not by reading.

## 5. Matched case

Mesh 480×60, physical filter radius R = 0.06·b (3.6 elements), ρ₀ = 0.5, V = 0.5, ρ_min = 1e-3,
ε = 0.05·√(NE/3200) = 0.15.

**Level M0 — native.** Source: retained committed `repro/results/S480x60/` (preset
`duOlhoffAdaptivePedersen`). Target: retained authoritative three-rung C480 canary
(`diagnostics/three_rung_canary_preflight`, full RHO/DRHO in
`evidence/three_rung_canary_preflight/C480x60_three_rung_trajectory.mat`). The target's
*canonical production* realization (four-rung/beta/designChange) has only a final state at
480 (September-11 campaign) and is used for endpoint rows only. No M0 rerun on either side.

**Level M1 — maximally matched common core.** Alignment allowed only through committed
configuration without code change or new method: the source code is a superset on the
material axis, so the source is run with the *target's* material law. M1 = source snapshot
code, committed preset `duOlhoffAdaptiveMove` with the committed sweep override
`move.initial = 0.10` (exactly the configuration of the committed source runs A2 at 240×30 and
C3 at 800×100), at 480×60, `runtime.diagnostics = true`, `runtime.verbose = true`. The
target cannot realize the adaptive box or Pedersen stiffness without code change, so the
target's M1 counterpart is its retained C480 canary (no target run).

**M1 launch preflight (fail-closed).** Before launch, the resolved M1 config is compared leaf by
leaf with the stored S480x60 config. Allowed differences: `material.stiffness.model`
(pedersen→simp), `material.mass.model` (eq2→eq4b), and runtime-only `runtime.name`,
`runtime.diagnostics`. Any other difference → do not launch. Path check: `olhoffSolve`,
`innerLoop`, `mmasub`, `olh.move.limit` must resolve inside the snapshot and nowhere else.

**Preregistered prediction P-prefix.** For all ρ with every element > 0.1, Pedersen+eq.(2) and
SIMP+eq.(4b) give identical K, M and derivatives. Therefore M1 must reproduce retained S480x60
`hist` and `aux` **bitwise** for every outer iteration k whose start state satisfies
min ρ_{k−1} > 0.1. Let k* be the first iteration with min ρ_{k*−1} ≤ 0.1 (computed from the M1
trajectory). Result classes: PREFIX_BITWISE_PASS (all k < k* bitwise equal); PREFIX_FAIL
(any k < k* differs). PREFIX_FAIL demotes every M1-vs-S480 statement to WEAK ASSOCIATION.

## 6. Compute budget (hard)

- Source-side 480×60 trajectories: **one** (M1). No M0 rerun.
- Target-side 480×60 trajectories: **zero** (retained C480 has full RHO/DRHO).
- No other mesh, no 160 preflight, no nine-mesh rerun, no SOCP C480 run.
- Offline same-state evaluations (FE, eigen, gradients, filter, rows, a single inner solve at a
  frozen state, SOCP/fmincon-free) are not trajectories and are unlimited, with zero ρ updates
  applied to any trajectory.
- M1 runs to natural convergence or the preset cap of 400. It is not killed for scientific
  reasons (spikes, grayness). It is stopped only for MATLAB error, host resource exhaustion, or
  wall time > 5 h; then the partial trajectory is used and flagged.

## 7. Trajectory quantities

Per outer iteration, on each side where the data exist (M0-source from `hist`/`aux`; M1 and
target from full RHO/DRHO):
ρ hash (SHA-256 of little-endian doubles), ω₁, ω₂, λ₁, λ₂, gap12 = (ω₂−ω₁)/ω₁, volume, M_nd =
4·mean(ρ(1−ρ)), gray fraction mean(0.1<ρ<0.9), mid fraction mean(0.4≤ρ≤0.6), β, move/box (max,
mean, per-element where available; per-element source box is reconstructed by replaying the
snapshot's own `olh.move.limit` on the trajectory and must match `hist.move` and
`aux.moveMean` bitwise), N, dOff, multiplicity/next-mode flags, max|Δρ|, ‖Δρ‖₂, sign-reversal
fraction mean(Δρ_k·Δρ_{k−1}<0), box-saturation fraction mean(|Δρ_e| ≥ 0.99·d_e), predicted gain
(β − λ_n at the step, and first-order Δλ from `deltaLambda`), realized gain λ_n(k+1)−λ_n(k),
stopping metric ‖Δρ‖₂ vs ε, controller state (stage / A,B,E for target; box statistics for
source), inner call count, inner stop metric (relative step), inner status.
Localized-mode spike event: ω₁(k) < 0.7·ω₁(k−1).

## 8. Same-state cross-evaluation

States: ρ₀ (uniform 0.5); target C480 iterations 10, 20, 100 and 386 (final); source S480
final; M1 iterations k*−1, k*+5 and final (chosen by rule, after the run). For each state,
three evaluators in separate MATLAB sessions with isolated paths:
**T** target `+impl` (SIMP/4b); **S1** source snapshot with SIMP/4b; **S0** source snapshot with
Pedersen/eq2. Quantities: K, M (Frobenius relative difference), λ₁…λ₅, raw f_sk (N=2, with
diagonal λ_j and off-diagonal λ̃), f_JJ, filtered versions, dOff, N, problem-(25) rows
(constraint values and gradients at Δρ=0), and one full `innerLoop` solve with (a) a common
scalar box 0.04 and (b) each side's native box at that state.

Identity tolerance: two quantities *agree* if max relative difference ≤ 1e−9 (eigenvector
signs aligned by the sign of the largest component); *bitwise* is reported separately.
S1 vs T answers implementation identity; S0 vs S1 answers formulation operator difference.

## 9. First-divergence definition (verdict from M1)

Ordered levels evaluated on the M1 pair (source S1 + adaptive box vs target three-rung) at the
earliest shared state ρ₀:
D0 problem/physics configuration differs (material, filter, volume, bounds, ρ₀, multiplicity
model, problem-(25) form, inner solver settings) — controller and stopping fields are excluded
from D0; D1 K/M/eigenpairs differ at identical ρ; D2 raw f_sk; D3 filtered f_sk; D4 N/cluster/
dOff/next-mode row; D5 problem-(25) rows (25a–e) and gradients; D6 inner solve returns a
different Δρ for identical rows **and identical box**; D7 outer box/move/controller supplies a
different box (the move part of (25f)) or scales/accepts Δρ differently; D8 stopping decision.
Verdict = lowest level failing at ρ₀; if all of D0–D7 agree at ρ₀, the earliest later shared
state decides. The M0 comparison is reported separately (the formulation axis is expected to be
D0-different but inactive at ρ₀); it does not set the FIRST_DIVERGENCE verdict.
A first divergence is a *location*, not a cause; it never alone supports STRONG causality of
the diverging component for the final outcome.

## 10. Causal attribution standards

Evidence levels: STRONG CAUSAL EVIDENCE requires same-state direct comparison, first-divergence
isolation, or a single-factor pair (effective configs differ only in the named factor, verified
leaf by leaf, same mesh, same code). MODERATE: single-factor at another mesh, or a two-factor
pair with one factor proven inert. WEAK ASSOCIATION: final tables, multi-factor pairs.
EVIDENCE AGAINST: a single-factor or same-state test in which the factor is changed and the
outcome does not change beyond thresholds. IDENTICAL BETWEEN IMPLEMENTATIONS: same-state
identity or byte-identical code on the executed path. NOT ISOLATED otherwise.

Outcome thresholds ("materially different"): termination status differs, or |ΔM_nd| ≥ 0.03, or
|Δω₁|/ω₁ ≥ 1 % under a common re-evaluation model (SIMP p=3 + eq.(4b)), or spike count 0 vs ≥ 1,
or outer-iteration ratio ≥ 1.5.

Decomposition at 480 along the path C (target) → M1 → S (source):
ΔTotal = M_nd(C) − M_nd(S); ΔController = M_nd(C) − M_nd(M1) (at SIMP/4b);
ΔMaterial = M_nd(M1) − M_nd(S) (under the source controller); share_material = ΔMaterial/ΔTotal.
Dominant verdict: share ≥ 0.75 → PRIMARILY_FORMULATION; share ≤ 0.25 and M1 terminates
naturally without spikes → PRIMARILY_OUTER_BOX_CONTROLLER; 0.25 < share < 0.75 → MULTICAUSAL.
If M1 shows spike events, CAP_HIT or collapse, the material factor is additionally recorded as
necessary for stability under the source controller and share is computed on its endpoint; if
M1 has no interpretable endpoint (crash before iteration 50) → CAUSE_NOT_ISOLATED.
INNER_SOLVER is eligible only if inner code or inner settings differ on the executed paths; box-
mediated changes of inner behaviour are attributed to the controller. The Pedersen-under-ladder
cell is not run; all statements carry "interaction not isolated".

Retained single-factor pairs admitted if the leaf-by-leaf config check passes: A2_240_d010 vs
S240x30 and C3_800_d010_R06 vs S800x100 (material factor under the source controller).

## 11. Stopping-criterion analysis

Recover the executed stop formula, threshold, monitored quantity, persistence and guards for
both sides from code and effective config. Endpoint stationarity is measured with the frozen
definition of `gray_kkt_forensic_audit/scripts/stationarity.py` (`kkt` with gray-fit multiplier,
normalized by interior raw RMS); the implementation must reproduce the prior C480 value 0.334039
(±1e−4) before being applied to S480 and M1 endpoints. Natural convergence is classified:
BETTER_LOCAL_OPTIMALITY if the source gray-fit RMS is ≤ 0.5× the target's under the same
formulation; HEURISTIC_STOP if the source criterion fires while the residual is ≥ the target's;
otherwise TRAJECTORY_CONSEQUENCE with the measured values.

## 12. Grayness milestones

Iterations 1, 5, 10, 20, 40, 60, 80, 100, 112, 150, 200, 300, 386 (where present) and first
crossings of M_nd ≤ 0.75, 0.50, 0.35, 0.25, 0.15. Grayness divergence onset = first iteration
at which |M_nd(side A) − M_nd(side B)| ≥ 0.03 and stays ≥ 0.03 for 10 consecutive iterations.

## 13. Low-density mechanism (Part 18)

At the final C480, S480 and M1 states, both material laws: λ₁…λ₅; per mode, kinetic-energy
fraction in elements with ρ < 0.1 and ρ < 0.3; a mode is *localized* if ≥ 50 % of its kinetic
energy is in ρ < 0.3 elements. Element stiffness/mass factors of low-density elements under both
laws. Spike events in all retained histories. No extension to the Proposed method.

## 14. Migration classifications and gate

Per change: PROMOTE NOW / PROMOTE AS NEW NAMED PRESET / KEEP HISTORICAL ONLY / DO NOT PROMOTE /
NEEDS CAUSAL TEST. Gate: exactly one of READY / READY_WITH_NAMED_FORMULATION_SPLIT /
REQUIRES_CAUSAL_TEST / BLOCKED. If the formulations differ and the source method is
scientifically defensible, READY_WITH_NAMED_FORMULATION_SPLIT is preferred over READY. Phase 6
(other methods) is always classified separately and never authorized here.

## 15. Stop conditions for the audit

- Source commit absent, or claimed sweep evidence not committed at the hash → STOP, report.
- Any modification of `+impl`, the source repository, or the snapshot detected → STOP, report.
- M1 preflight fails → no run; M1-dependent verdicts become NOT ISOLATED.
- No step writes outside `analysis/OlhoffCurrent/diagnostics/scientific_delta_olhoff_migration/`
  (plus reading retained evidence).
