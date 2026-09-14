# CAMPAIGN_DECISION — Part J / Part K

## Decision

```
THREE_RUNG_CONTROLLER_GENERALIZES_BUT_FINE_MESH_SCIENCE_SUSPICIOUS
CORRECT_NINE_MESH_CAMPAIGN_BLOCKED
```

This is **preregistered Case 4** (`PREREGISTRATION.md` §11): 480 PASS, 800
controller PASS, but the 800 endpoint is scientifically poor. The rule written
before any run says *"normally `CORRECT_NINE_MESH_CAMPAIGN_NOT_YET_AUTHORIZED`,
because the next question is spectral/multiplicity/discretization, not
stopping."* The required final vocabulary admits only AUTHORIZED or BLOCKED, so
that maps to **BLOCKED** — and the reason is the one Case 4 names, not a defect
in the controller.

## 1. What the canaries settled

**The controller generalizes.** Six meshes now, spanning NE = 3 200 → 80 000, a
factor of 25, under one frozen policy:

| mesh | S1 | S2 | S3 | total | terminal branch | amp/ε at S3 | status |
|---|---|---|---|---|---|---|---|
| 160×20 | 102 | 39 | 39 | 180 | B | 0.1036 | CONVERGED |
| 240×30 | 206 | 39 | 39 | 284 | B | 0.0477 | CONVERGED |
| 320×40 | 274 | 39 | 39 | 352 | B | 0.0327 | CONVERGED |
| 400×50 | 388 | 39 | 39 | 466 | B | 0.0633 | CONVERGED |
| **480×60** | **308** | 39 | 39 | **386** | B | 0.0524 | CONVERGED |
| **800×100** | **390** | 39 | 39 | **468** | B | 0.0845 | CONVERGED |

Every mesh: one S1 declaration, then both lower rungs at the minimum dwell of
39 = (W−1) + P, terminal declaration on branch B with full persistence, terminal
amp/ε inside a narrow band. Twenty gates evaluated across the two canaries, all
twenty pass. **No new controller behaviour appeared at either canary, and no
cap was approached** (headroom 1214 and 1132 of 1600).

This is a genuine positive result and it answers the question the September 11
campaign could not: the validated three-rung controller does not break beyond
400×50.

**The science does not follow.** Under that same correct controller:

| mesh | M_nd % | ω₁ |
|---|---|---|
| 160×20 | 12.756 | 169.975 |
| 240×30 | 12.917 | 167.039 |
| 320×40 | 12.940 | 166.426 |
| 400×50 | 15.373 | 166.452 |
| 480×60 | 26.342 | 163.932 |
| 800×100 | 34.412 | 161.946 |

M_nd rises monotonically and accelerates; ω₁ falls monotonically; neither shows
an asymptote. The three-rung 800×100 design (34.41 % gray) is **as gray as the
legacy 480×60 design** (34.67 %). Refining the mesh 2.8× under the correct
controller bought a design no more discrete than a coarser mesh produced under
the wrong one.

Crucially, this is **not** a stopping artefact. At both canaries the terminal
window is flat — ω₁ moved +0.0037 % and +0.014 %, M_nd −0.036 and −0.064 points
over the final 20 iterations. The controller stopped where the design had
stopped moving. The design stopped moving while still a third gray.

## 2. Why that blocks the campaign

The nine-mesh campaign's purpose is a **mesh-convergence and performance
benchmark**. Running it now would produce nine correctly-controlled,
correctly-instrumented, fully-evidenced runs whose objective and discreteness
diverge monotonically with refinement. That is an expensive way to document a
problem the two canaries have already documented for a twentieth of the cost.

The next question is not "does the controller work at 560/640/720" — the
canaries bracket that range and the answer is almost certainly yes. It is **why
the design degrades under refinement**, and that is a spectral, discretization
or formulation question that no stopping rule can answer.

## 3. Candidate explanations — none tested here, none preferred

Recorded so the next task starts from a list rather than from scratch. This
study changed none of them and has no evidence favouring one.

| candidate | what the canaries show | what would test it |
|---|---|---|
| **Fixed p = 3 without continuation** | A known CLASS C reconstruction ruling (`duOlhoffFrozenM4`). Gray mass concentrates in the low-modal-strain end regions, where a fixed penalty has weak gradients. | p continuation, already implemented (`material.stiffness.continuation`) and never enabled in production |
| **Sensitivity filter at fixed physical radius** | R = 0.06·b is mesh-independent by construction (rminEl 3.6 → 6.0), so physical smoothing is constant — yet gray grows anyway | density filter vs sensitivity filter at matched radius |
| **No projection** | `projection.enabled = false`; a tanh Heaviside is implemented but is CLASS D and absent from every source | projection continuation, at the cost of leaving the published formulation |
| **Spectral crowding during coalescence** | 81 next-mode warnings at 800×100, but confined to iterations 26–113 and resolved 355 iterations before the endpoint | larger `eigen.maxCluster`, or a defined procedure for multiple ω_J |
| **Genuine continuum behaviour** | The optimum becomes bimodal only at fine mesh (gap12 3.49e−05 at 800×100); possibly the continuum problem has no discrete optimum at this volume fraction | finer meshes with a discreteness measure, i.e. the campaign — but only after the above are excluded |

The last row is why the campaign is *blocked* rather than *cancelled*: if the
first four are excluded, the nine-mesh series becomes the right experiment.

## 4. The other two blockers, unchanged by the canaries

**Production is still unpromoted.** `olhoffcurrent_preset()` names
`duOlhoffFrozenM4` — ladder `[0.04 0.02 0.01 0.005]`, `boundVariable`,
`designChange`. `three_rung_promotion_closure` records
`PRODUCTION_THREE_RUNG_CONTROLLER_NOT_PROMOTED` and `PRODUCTION_FREEZE_FAIL`. A
campaign resolving through `olhoffcurrent_config` would run the legacy policy
again, exactly as the September 11 one did. These canaries sidestepped it by
applying the validated overrides explicitly through `tr_config`, which is
disclosed in `cp_config.m`, `DEPLOYMENT_PREFLIGHT.md` and
`PREFLIGHT_MANIFEST.json`. That is acceptable for two canaries; for a campaign,
promotion should be completed or the explicit-override driver should be made the
documented production entry point.

**Retention and clean timing remain mutually exclusive.** The canaries needed
`runtime.diagnostics = true` to retain trajectories; the legacy campaign ran
`false`. Per-outer cost differs by ~1.3–1.4× across that boundary, mixed with
stage-mix and session effects. One nine-mesh series cannot be both an evidence
campaign and a clean performance campaign.

## 5. Frozen specification for the NEXT task — not executed here

Recorded so the next attempt inherits a decided design. It does **not**
authorize the campaign; §1–§4 block it.

**Preconditions, all required before the first solve**

1. A spectral/discretization diagnosis that explains, or excludes, the M_nd
   growth in §1 — the substantive precondition, and the only new one the
   canaries added.
2. Promotion completed, or the explicit-override driver adopted as the
   documented production entry point (§4).
3. The two-pass retention/timing design below adopted (§4).
4. `THREE_RUNG_DEPLOYMENT_PREFLIGHT_PASS` on the campaign's own driver, with
   per-mesh config hashes frozen before the first solve.

**Series** — 160×20, 240×30, 320×40, 400×50, 480×60, 560×70, 640×80, 720×90,
800×100. One machine, one frozen effective config per mesh, one source/config
identity, one telemetry schema, one evidence-retention policy, one preflight
assertion mechanism, all nine meshes.

**Two-pass design.** Pass E (evidence) with `diagnostics = true`, retaining full
trajectories, reporting all science and *no* headline timings. Pass P
(performance) with `diagnostics = false`, identical in every other field,
reporting timings. The two passes must produce **bitwise identical** ω, ρ and
controller events; the recorder is documented as bitwise inert and the campaign
must **verify** that per mesh rather than assume it. Any mesh where they differ
invalidates the timing pass for that mesh.

**Frozen policy** — ladder `[0.04 0.02 0.01]`, `stageExhaustion` for both
continuation and stopping, frozen E = A OR B (W = 20, P = 20, Wnp = 10), beta
with no authority, cap 1600, ε = 0.05·√(NE/3200), R = 0.06·b, everything in
`PREREGISTRATION.md` §3 unchanged.

**Cap.** 1600 for all nine, unchanged. The canaries used 386 and 468, leaving
headroom of 1214 and 1132, and S1 length is non-monotone in mesh (102, 206, 274,
388, 308, 390) so no reliable extrapolation exists — which is an argument for
keeping the generous cap, not for tuning it. A CAP_HIT is reported as a CAP_HIT
and the cap is not raised afterwards.

**Budget, now measured rather than extrapolated.** Per-outer cost scales as
NE^0.83 and outer counts sit in the 180–470 band across the six measured meshes.
Interpolating the measured canaries: the full nine-mesh series is roughly
5–6 hours per pass, ~11 hours for both, and ~2.5 GB of retained trajectory.
(The pre-run projection in `PREREGISTRATION.md` §4 was wrong by 1.5× at 480 and
2.4× at 800; these figures are measurements, and are still interpolations at the
four unmeasured meshes.)

**Instrumentation additions the canaries showed are needed**

* Instrument `mmasub`'s internal dual iterations. Seconds per MMA step rises
  ×2.9 (480) and ×2.0 (800) from stage 1 to stage 3 while step counts do not
  explain it, and the present telemetry cannot say why. The terminal rungs are
  16–20 % of iterations but 28–34 % of wall time, so this is not a minor line
  item.
* Export ω₃…ω₅, gap23 and the three timing channels in the primary schema. The
  frozen `cv_export` schema omits them; this study added `cp_supplement.m`
  rather than edit another study's artefact, but a campaign should carry them
  natively.

**Splicing rules.** Do **not** splice the September 11 legacy campaign into the
new performance series. Do **not** splice these two canaries into it either —
they ran with `diagnostics = true`, in a separate session, under this study's
driver. Do **not** splice the earlier C160/C240/C320/C400 validation runs. All
of them remain historical comparators only, and must be labelled as such in
every table they appear in.

## 6. What would unblock it

A diagnosis of the M_nd growth. Two of the five candidates in §3 are testable
cheaply and without touching the controller — p continuation and projection are
both already implemented and both currently disabled — and either could be
tested at 480×60 and 800×100 against the canary endpoints now in hand, which is
exactly what makes those endpoints worth keeping.

If such a test shows the grayness is a formulation choice rather than a
discretization limit, the correct nine-mesh campaign becomes the right next
experiment and this specification is ready to run. If it shows the continuum
problem genuinely has no discrete optimum here, the campaign would document a
negative result and the programme should say so instead.

**No controller change is recommended, and none is justified by this study.**
