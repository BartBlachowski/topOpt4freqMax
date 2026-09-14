# Phase 2B-R v3 — governing specification (v2 + 10 repository-grounded patches)

This is the executed specification: v2 of the Phase-2B prompt with the ten corrections
from the third review applied. Patches 1-3 are outcome-critical. Clauses unchanged from
v2 are summarized; patched clauses are given in full because they govern what was run.

## Binding ruling (preserved in force from v2)

Exact binary-field identity after exact-count projection is IMPORTANT DIAGNOSTIC
EVIDENCE but is NOT itself a frozen acceptance requirement. `binary_double ~=
binary_single` does not automatically imply qualification failure. What must remain
invariant are the FROZEN SCIENTIFIC DECISIONS: topology PASS/FAIL, spectral/quality
PASS/FAIL, reference classification, b_ref, B_meas, k_enter, k_cert, final status.
Every binary difference must be counted, localized, characterized and propagated
through the full topology and acceptance machinery. An unexplained binary difference is
not acceptable evidence of equivalence. Both STRICT_BINARY_IDENTITY and
FROZEN_DECISION_EQUIVALENCE are reported; only the latter governs authorization.

Normative sources, quoted. IMPLEMENTATION_REQUIREMENTS.md section 4: "A compressed
single-precision path for **new** trajectories is permitted only after a
no-solve/short-run validation proves that every gate decision, `k_enter`, `k_cert`, and
E1/E2/E3 frequency within a documented bound is identical to the double path."
Contract `trajectory_storage.single_precision_permission`:
"only_after_no_solve_or_short_run_validation_proves_identical_gate_decisions_k_enter_k_cert_and_documented_E1_E2_E3_bound".
Contract `trajectory_storage.historical_olhoff_single_precision`:
"historical_evidence_only_not_automatic_authorization_for_new_trajectories".

## PATCH 1 (critical) — snapshot indexing

`res.rho_snapshots` is 2-D, `NE x (nDone+1)` (olhoffOptStabilized.m:21,65,80). v2's
`res.rho_snapshots(:,:,end)` and `(:,:,k)` are wrong: on a 2-D array MATLAB resolves
`A(:,:,end)` with `end`=1 in the trailing singleton dimension and silently returns the
ENTIRE matrix. Column 1 is the INITIAL density, so the state after k accepted updates is
column k+1. Governing accessors:

    x_double = res.rho
    x_single = res.rho_snapshots(:, end)
    state after k accepted updates = res.rho_snapshots(:, k+1)

with asserted preconditions ndims==2, size(...,2)==nOuter+1,
size(...,1)==numel(res.rho), and isequal(res.rho_snapshots(:,end), single(res.rho)).

Execution finding: the pre-existing `+ie2b` code already used the correct 2-D accessors.
Patch 1 corrects the v2 prompt text, not that code. The ~2.67% E2/E3 tail is therefore
NOT a mispairing artifact and required a separate root cause (see WP8B).

## PATCH 2 (critical) — pairing within one capped run

Pairs are constructed WITHIN a single checkpoint-limited run:

    run with cfg.maxOuter = k
    x_double = res.rho
    x_single = res.rho_snapshots(:,end)
    assert isequal(x_single, single(x_double))

Cross-run comparison is used ONLY for the mandatory bit-identical prefix determinism
test, never as the pairing mechanism.

## PATCH 3 (critical) — reference reachability

reference_phase.m asserts b_ref >= P + L_ref = 600; block endpoints are multiples of
P=100; the floor at b needs a valid P-window (H0 true over 100 consecutive states); the
freeze test needs (F(b)-F(b-500))/F(b) <= 1e-3 for all of E1,E2,E3. Qualification runs
target >=1000 and preferably >=1600 accepted updates. Prior horizons of 200/320 were
structurally incapable of producing a b_ref. Where reference still does not establish,
the report states WHICH frozen condition failed and distinguishes "the case does not
stabilize" from "too few iterations".

## PATCH 4 — mesh constraints

topology_metrics.m asserts mod(nely,2)==0; the frozen domain is 8:1 (L=8, H=1).
Qualification meshes must be even-nely and 8:1: 48x6, 80x10, 96x12, 128x16. The prior
24x4 is 6:1 and off-aspect; Phase-2A's 40x5 smoke mesh is odd-nely and throws.

## PATCH 5 — A_sig regime shift

A_sig = 0.01 is fixed in AREA, so in ELEMENTS it scales:
a_sig_elements = A_sig*nelx*nely/8. Production: 4, 9, 16, 25, 36, 49, 64, 81, 100.
Qualification: 96x12 -> 1.44, 80x10 -> 1.00, 48x6 -> 0.36, 24x4 -> 0.12. Where
a_sig_elements <= 1 the gate degenerates to "zero detached elements permitted", a
qualitatively different regime from production. Tiny meshes are CONSERVATIVE for
topology equivalence and BLIND to the production regime tolerating 4..99 detached
elements. a_sig_elements is reported per mesh with a
STRICTER_THAN_PRODUCTION / COMPARABLE / WEAKER_THAN_PRODUCTION classification.

## PATCH 6 — monotonic-rounding / tie-collapse proof

exact_count_binary.m selects by strictly decreasing density with an ASCENDING GLOBAL
INDEX tiebreak, and nSolid depends only on numel(x) and volfrac, so it is identical for
both representations. IEEE double->single rounding is MONOTONIC NON-DECREASING and
cannot strictly invert the order of any two values; it can only COLLAPSE two distinct
doubles onto one single. Hence the exact-count binary field can differ by exactly one
mechanism: a newly-created tie spanning the cutoff rank, resolved by the index tiebreak
against the double ordering. At-risk states are exactly those where the single values at
ranks nSolid and nSolid+1 are equal, decidable from the single trajectory alone. Any
observed binary difference not explained by this mechanism indicates a harness bug.

The same monotonicity argument governs the branch tests in the mass/stiffness laws and
is used to build a rigorous two-sided bracket on the unrecoverable double state:
for the E2/E3 mass law g(x) = x^6 (x<=0.1) else x, an element is branch-ambiguous under
single storage if and only if its stored single value equals single(0.1). Forcing all
such elements onto the x^6 branch yields Q_hi; leaving them as stored yields Q_lo; and
because omega_1 is monotone decreasing in element mass (Rayleigh quotient), the true
double value satisfies Q_double in [Q_lo, Q_hi].

## PATCH 7 — the frozen gate is the full hard gate

topology_metrics.m defines hard_gate_pass = volume_pass && topology_pass, and
analyze_trajectory.m builds H0 = H_health & H_volume & H_topology & H_method, which is
what reference_phase and scan_persistence consume. All three flags are compared and the
volume-tolerance margin is reported alongside quality margins. Correct diagnostic field
names: aggregate_detached_elements, aggregate_detached_area, max_detached_elements,
max_detached_area, n_islands_all, n_islands_significant, largest_component_fraction,
a_sig_elements, element_area. `det_total` does not exist.

## PATCH 8 — early-terminated frozen runs

480x60 (358 states), 560x70 (400) and 640x80 (1067) terminated before the 1601 cap. On
the SOLVER_FAILURE path the loop breaks before the rho update, so res.rho still equals
the final stored column, but this is VERIFIED per run, not assumed: record res.status
and res.failure_iteration, confirm nDone+1 == size(rho_snapshots,2), and assert
isequal(rho_snapshots(:,end), single(res.rho)) individually.

## PATCH 9 — artifact path and preflight hashes

The qualification artifact must be written where +ie2a/paths.m resolves:
analysis/iteration_efficiency_phase2a/validation_outputs/olhoff_new_trajectory_precision_qualification.json,
with pass and scope=='new_olhoff_trajectory'. Creating a NEW file there is permitted and
is not a modification of Phase-2A evidence; overwriting an existing file is not. Any
authorized minimal edit to production_preflight.m requires recorded pre/post SHA-256,
the literal diff, and a staleness test transcript. If qualification does not pass, no
artifact is created and no preflight edit is made.

## PATCH 10 — additional final-summary items

35. Was the snapshot indexing convention verified by assertion before any metric?
36. a_sig_elements per qualification mesh with regime classification.

## Preservation

analysis/iteration_efficiency_phase2b_precision/ is preserved byte-for-byte. All new
evidence is written to analysis/iteration_efficiency_phase2b_recheck/. Superseded prior
artifacts are marked in EXISTING_EVIDENCE_DISPOSITION.csv, never overwritten.

## Execution boundary

No production campaign. No methodology change. No protected-source modification or fork.
No authorization token. Phase 2B-R ends at a four-way verdict.
