# A4 — Independent Scientific Audit

**Date:** 2026-07-20
**Auditor:** independent review of completed A4 run (`examples/Revision_v1/output/a4/`)
**Specification audited against:** `examples/Revision_v1/A4_SPECIFICATION_V3.md`
**Run audited:** `created_utc 2026-07-19T22:25:38Z`, `commit_sha 2c945de`, elapsed 526 s
**Method:** artifacts read first; implementation read only to interpret them; primary
endpoints re-derived from an **independent** FE model; failure mechanism re-tested by
**direct re-execution** in MATLAB R2025b.

---

## 1. Executive summary

> ## Verdict: **FAIL**

Not because the machinery broke — it did not — but because the experiment's headline
outcome is an artifact of an implementation constant, and I was able to falsify it
directly.

A4 reports that 3 of 5 arms (`N = 10, 5, 1`) suffered **B3 spurious-mode contamination**
— "the refresh locked onto a disconnected-island mode" — and are therefore disqualified
as accuracy references, yielding decision `INDETERMINATE`.

**That classification is false, and the run's own artifacts already contradict it.** All
three arms record `Solid components = 1` in their own exception messages: a single
connected solid body cannot produce a disconnected-island mode. I re-ran the frozen
trajectory at the exact iterations where those arms died and measured the spectrum
directly:

| Iteration | 20-mode window (as run) | 60-mode window |
|---|---|---|
| 25 (where `N=5` died) | best MAC to Φ₀ = **0.0000**, 0 admissible → **B3 declared** | Φ₁-type mode at **index 49**, MAC **0.9775**, **admissible** |
| 30 (where `N=10` died) | best MAC to Φ₀ = **0.0000**, 0 admissible → **B3 declared** | Φ₁-type mode at **index 37**, MAC **0.9664**, **admissible** |

A clean, support-connected, MAC ≈ 0.97 Φ₁-type mode **existed at both failure points and
passed the full §4.3.1 screen.** It was simply outside the hard-coded 20-mode search
window. The arms were not contaminated; they were truncated.

Three further defects compound this:

- The **frozen arm's own trajectory** fails the same screen at iterations 2, 20, 25 and 30.
  The screen failure is a property of *early designs*, not of refreshing. `N = 50` survived
  only because its first refresh falls at iteration 50 — just past the window in which every
  arm fails. **`N` is perfectly confounded with "how early the screen is first applied,"**
  which voids the single-factor design the specification was written to protect.
- `ω₁ᵗʰʳᵉˢʰ = 26.5` — published in Table A4-1 next to `ω₁ᵗʳᵃᶜᵏ = 159.6` — is an artifact of an
  undeclared `1e-3` void floor in the threshold routine. The correct value is **162.5**
  (MAC 0.9997). The table currently tells a reviewer the design is a gray-material fiction.
  It is not.
- `base_config_hash` is the constant `ffffffff` for *any* input file (MATLAB `uint32`
  saturation). Every provenance and factor-drift guard in the campaign is vacuous.

**What A4 did produce, and then discarded:** `N = 50` converged cleanly in 541 iterations
to `ω₁ᵗʳᵃᶜᵏ = 159.6012` against the frozen arm's `159.5656` — a difference of **0.022%,
227× inside the pre-declared δ = 5%** — with visually and numerically near-identical
topologies. I verified both endpoints independently to seven significant figures. That is
a real, clean, single-factor result supporting H₀ on this benchmark. The decision logic
threw it away on a technicality (the *reference* arm capped), and the primary figure is
auto-scaled to a 0.025% y-range so the two points appear far apart.

---

## 2. Findings

### CRITICAL

---

**C-1 — The B3 classification of `N = 10, 5, 1` is refuted by direct measurement.**

*Description.* The three arms are labelled B3 ("spurious-mode contamination… the refresh
locked onto a disconnected-island mode… DISQUALIFIED as an accuracy reference"). The
mechanism named does not exist in the data. The true cause is the 20-mode search window
(`nModes = 20`, spec §4.1 / §4.3.1), which excludes the physical mode on intermediate
designs.

*Evidence.*
- The arms' own exception messages record `Solid components = 1` (`N=10`, `N=5`) — a single
  connected body, so no island mode is possible.
- Direct re-execution (`scratchpad/a4_audit_window.m`, MATLAB R2025b):

```
ITER 25 nModes  20 | omega range [ 44.566 .. 130.120] | best MAC to Phi0 = 0.0000 at index 19
                   | nComponents 1 | nAdmissible 0 | selected 0
ITER 25 nModes  60 | omega range [ 44.566 .. 174.315] | best MAC to Phi0 = 0.9775 at index 49 (omega 160.770)
                   | nComponents 1 | nAdmissible 1 | selected 49
ITER 30 nModes  20 | omega range [106.820 .. 144.040] | best MAC to Phi0 = 0.0000 at index 14
                   | nComponents 1 | nAdmissible 0 | selected 0
ITER 30 nModes  60 | omega range [106.820 .. 182.435] | best MAC to Phi0 = 0.9664 at index 37 (omega 160.469)
                   | nComponents 1 | nAdmissible 1 | selected 37
```

Extending the window from 20 to 60 converts "no admissible mode → terminate as B3" into
"admissible Φ₁-type mode found, MAC 0.98". Extending further to 120 changes nothing —
index 49 / 37 is the true location.

*Underlying mechanism.* `E_min/E₀ = ρ_min/ρ₀ = 1e-9`, so the void has **exactly the same
wave speed** as the solid (`c = √(E/ρ)` identical). Void regions are therefore a
full-fledged elastic continuum with a dense low spectrum, and on intermediate designs
dozens of genuine void modes sit below the structural mode (`ω₁ᵐⁱⁿ = 44.6` at iteration 25
vs the physical `160.8`). This is a real and reportable property of the formulation — and
it is *not* the disconnected-island mechanism A4 attributes it to.

*Affected files.* `a4_result.json` (arms 2–4), `a4_table.md`, `a4_manifest.json`,
`a4_eigenpair_refresh_results.mat`, `a4_fig1_omega1_vs_N.png`,
`scripts/revision_v1/a4_mode_screen.m`, `analysis/ourApproach/Matlab/topopt_freq.m` (R-1),
`A4_SPECIFICATION_V3.md` §4.1/§4.3.1.

*Recommended fix.* Make the mode window adaptive: expand until an admissible candidate is
found or a declared ceiling (e.g. 200) is reached, and **record the required window size
per refresh** — it is itself the diagnostic. Re-run all finite-`N` arms. Re-classify: what
was observed is mode migration under void-mode descent (**B1**), not **B3**.

---

**C-2 — `N` is confounded with the iteration at which the admissibility screen is first
applied.**

*Description.* The specification's central guarantee is that arms differ in exactly one
respect. They do not. The screen is evaluated only at refresh events, so the refresh
interval also controls *when the design is first screened* — and the screen fails on all
early designs regardless of refreshing.

*Evidence.* Re-running the **frozen** (`N = ∞`) trajectory, which never refreshes and was
never screened during its run (`scratchpad/a4_audit_screen.m`):

```
ITER    2 | nComp    4 | nAdmissible  0 | selected  0 | minX 1.000e-01
ITER    5 | nComp    5 | nAdmissible  1 | selected 20 | minX 1.162e-07
ITER   10 | nComp    1 | nAdmissible  1 | selected 13 | minX 8.424e-38
ITER   15 | nComp    1 | nAdmissible  1 | selected 19 | minX 5.954e-68
ITER   20 | nComp   13 | nAdmissible  0 | selected  0 | minX 7.933e-98
ITER   25 | nComp    1 | nAdmissible  0 | selected  0 | minX 1.180e-127
ITER   30 | nComp    1 | nAdmissible  0 | selected  0 | minX 1.550e-157
ITER   40 | nComp    1 | nAdmissible  1 | selected 18 | minX 1.883e-217
ITER   50 | nComp    1 | nAdmissible  1 | selected  5 | minX 2.071e-277
ITER  100 | nComp    1 | nAdmissible  1 | selected  1 | supKinFrac 0.9687
```

The published method itself would be declared B3 at iterations 2, 20, 25 and 30. It escapes
only because it is never asked. `N = 50`'s first refresh at iteration 50 lands one step
after the window closes — and even there it selects **index 5**, not 1.

*Cross-validation.* `a4_result.json` records for `N=50` at iteration 50: `"index": 5,
"omega": 159.8795, "n_admissible": 1`. My independent re-run of the same trajectory
selects **index 5** at iteration 50. The driver's telemetry is faithful; the *interpretation*
is what fails.

*Affected files.* Same as C-1, plus `examples/Revision_v1/a4_preflight_spectral_screen.m`.

*Recommended fix.* Screen every arm on a **common iteration grid** independent of `N`
(e.g. `{2, 5, 10, 20, 50, 100, 300, 600, final}`), so screen exposure is not a function of
the independent variable. Report screen outcome as a per-iteration observable for all arms
including `N = ∞`.

---

**C-3 — `ω₁ᵗʰʳᵉˢʰ` is an artifact of an undeclared `1e-3` void floor; the published value is
wrong by a factor of 6.**

*Description.* `a4_endpoint_eval.m > localVolumePreservingThreshold` sets thresholded void
elements to `1e-3` ("keep the declared lower bound"). The declared lower bound in
`a4_ss_400x50_base.json` is `rho_min = 1e-9`. The floor injects **10⁶× too much void mass**
at ~zero stiffness, manufacturing void modes and destroying mode tracking.

*Evidence.* Independent FE replication from the published topology CSVs
(`scratchpad/verify_a4.py`):

| Quantity | `N = ∞` | `N = 50` |
|---|---|---|
| `ω₁ᵗʳᵃᶜᵏ` reported | 159.56562699 | 159.60117295 |
| `ω₁ᵗʳᵃᶜᵏ` **independently replicated** | **159.565627** ✓ | **159.601173** ✓ |
| `ω₁ᵗʰʳᵉˢʰ` reported (floor 1e-3) | 26.5193 | 26.4802 |
| `ω₁ᵗʰʳᵉˢʰ` replicated, floor 1e-3 | 26.5193 (j\*=**13**, MAC **0.0002**) | 26.4802 (j\*=13, MAC 0.0002) |
| `ω₁ᵗʰʳᵉˢʰ`, floor **1e-9** (declared) | **162.4677** (j\*=1, MAC **0.9997**) | **162.4788** (j\*=1, MAC 0.9997) |
| `ω₁ᵗʰʳᵉˢʰ`, floor 0 | 162.4677 | 162.4788 |

At the coded floor the "tracked" mode is found at index 13 with **MAC = 0.0002** — the
routine reports the frequency of an arbitrary void mode and labels it `ω₁ᵗʰʳᵉˢʰ`. With the
declared floor, `ω₁ᵗʰʳᵉˢʰ ≈ ω₁ᵗʳᵃᶜᵏ` to within **1.8%**: the result is **not** a
gray-material artifact, which is the opposite of what Table A4-1 currently implies.

*Affected files.* `scripts/revision_v1/a4_endpoint_eval.m:~L150`, `a4_result.json`,
`a4_table.md`, `a4_eigenpair_refresh_results.mat`.

*Recommended fix.* Use the config's `rho_min` / `E_min_ratio` as the floor, and guard the
result with a MAC assertion (`MAC ≥ 0.8`, else report `NaN` with an explicit reason rather
than a number). Re-issue Table A4-1.

---

**C-4 — `base_config_hash` is a constant; every factor-drift and provenance guard is vacuous.**

*Description.* `a4_eigenpair_refresh.m > localHashFile` implements FNV-1a with MATLAB
`uint32` arithmetic, which **saturates** rather than wrapping. After the first
multiplication the accumulator pins at `0xFFFFFFFF` and never moves.

*Evidence.*

```
saturating (MATLAB semantics), a4_ss_400x50_base.json : ffffffff
saturating, ss_beam.json (a completely different file) : ffffffff
saturating, the 1-byte input "a"                       : ffffffff
wrapping (correct FNV-1a), a4_ss_400x50_base.json      : c141e407
```

All artifacts record `fnv1a32_ffffffff`.

*Consequence.* V-A4-2 ("all arms share an identical base-config hash") and the acceptance
gate's check `numel(unique({res.arms.base_config_hash})) == 1` **pass unconditionally**,
including in the exact scenario they exist to catch — arms silently running different
configs. This is the structural guard the specification (§7.3, §9) relies on to prevent a
repeat of the EXP4 four-factor drift, and it is inoperative.

*Affected files.* `examples/Revision_v1/a4_eigenpair_refresh.m:477-486`,
`examples/Revision_v1/run_all_revision_experiments.m > localAccept_A4`, and every emitted
artifact.

*Recommended fix.* `hash = mod(double(hash) * 16777619, 2^32)`, or call
`Simulink.getFileChecksum` / a `java.security.MessageDigest` SHA-256. Then add a
regression asserting two different files hash differently.

---

### MAJOR

---

**M-1 — The classifier omits a mandatory Class B criterion.**

Spec §5.2 Class B requires `ω₁ᵗʳᵃᶜᵏ ≈ ω₁ᵐⁱⁿ` **and** `≈ ω₁ᵗʰʳᵉˢʰ`. `check_a4_run.m` never
reads `omega1_thresholded`. The only Class B arm in the experiment (`N = 50`) was declared
"Eligible as an accuracy reference" without ever being tested against half of that
criterion — while carrying a reported 6× divergence.

Note the interaction: had M-1 been implemented, C-3's artifact would have disqualified
**both** surviving arms and A4 would have had zero usable data. Two defects cancelled.
Fix both together.

*Files.* `scripts/revision_v1/check_a4_run.m`.
*Fix.* Add the `ω₁ᵗʰʳᵉˢʰ` test after C-3 is corrected; add a classifier fixture for it.

---

**M-2 — `N = ∞` is labelled B4 with no B4 evidence.**

B4 is defined (spec §5.2) as *limit cycle* **and** *omitted-term ratio above threshold*.
The arm records `limit_cycle: false` and `omitted_term_ratio: null`. The classifier's own
reason string concedes it: *"B4 (unattributed): reached the iteration cap … **without a
limit-cycle/omitted-term signature**."*

`check_a4_run.m` uses B4 as a catch-all for "capped with no other signature". The published
method therefore carries a breakdown code that asserts a sensitivity-omission instability
which was never measured, and this label propagates into `a4_table.md`, `a4_fig1`
("DISQUALIFIED — NOT accuracy evidence") and `a4_fig4`. Presenting it in the manuscript
would be an unsupported causal claim about the paper's own method.

*Files.* `check_a4_run.m`, `a4_result.json`, `a4_table.md`, figs 1 and 4.
*Fix.* Introduce a distinct code (e.g. **B0 — unattributed non-convergence**) for capped
runs with no mechanism. Reserve B4 for the case where its two conditions are actually
measured.

---

**M-3 — `limit_cycle` is a default, not a measurement.**

Every arm reports `limit_cycle: false`. No `max|Δx_e|` history exists in any artifact, so
there is no basis for the value. It is a hard-coded default presented as data, and it gates
the B4 branch (M-2).

*Fix.* Either populate it from a retained Δx history with a stated detector, or emit `null`
and have the classifier treat "not measured" distinctly from "measured false".

---

**M-4 — Every per-iteration history required by the specification is absent.**

Spec §4.3 requires MAC history, mode identity, objective, `max|Δx_e|`, feasibility, and
`ω₁/ω₂` separation *per iteration for every arm*; §7.5 mandates a `histories` block. The
`.mat` contains only scalars:

```
res/arms fields: N, base_config_hash, baseline, breakdown, cap, class, class_reason,
eigensolves_analytic, exception_*, feasibility, final_design_change, grayness, iterations,
limit_cycle, load_sensitivity, mac_to_phi0, mode_index_jstar, n_components, n_refresh,
n_refresh_predicted, omega1_min, omega1_omega2_gap, omega1_thresholded, omega1_tracked,
omitted_term_ratio, pmass, refresh_events, refresh_inadmissible, success, tag, tol,
topology, wall_clock_s
```

No `design_change[]`, `objective[]`, `feasibility[]`, `mac[]`, `jstar[]`.

*Consequence.* Spec figures 2 (MAC vs iteration), 3 (`max|Δx_e|` log — the limit-cycle
detector), 4 (`j*` vs iteration) and 7 (`ω₁/ω₂` separation vs iteration) **cannot be
produced from the stored data at all.** This is not a missing-plot issue; the data were
never retained. Re-running is required.

---

**M-5 — Total telemetry loss on the three failed arms, including refreshes that demonstrably succeeded.**

All three B3 arms record `iterations: 0`, `n_refresh: 0`, `n_refresh_predicted: 0`,
`refresh_events: []`, every endpoint `null`. This is false: `N=5` must have refreshed
successfully at iterations 5, 10, 15 and 20 before failing at 25; `N=10` at 10 and 20 before
failing at 30; `N=1` at iteration 1 before failing at 2. The driver's `catch` block
(`a4_eigenpair_refresh.m:~L118`) discards everything the solver accumulated.

For an experiment whose stated purpose is *"to characterize failure"* (spec §5.1), the
observations immediately preceding each failure are the primary evidence — and they are the
exact records destroyed. Had they survived, C-1 would have been visible in the artifacts
without re-execution.

*Fix.* Have R-1 attach its accumulated event log and histories to the exception (or write
them to disk before throwing), and have the driver harvest them in the `catch`.

---

**M-6 — Gate A4-Pre samples only where the answer is always PASS.**

The gate checkpoints `{100, 300, 600, 2000}`. The screen fails at iterations 2, 20, 25, 30
and recovers by 100 (C-2). The gate is structurally blind to the only regime that matters,
and its recorded verdict —

> *"PASS: an admissible Phi1-type mode is identifiable at every checkpoint… **the refreshed
> arms have a clean spectrum to refresh into.** A4 proceeds."*

— was falsified within seconds by 3 of the 4 refreshed arms it cleared. A gate the
specification calls *"the only thing standing between A4 and a repeat of EXP4"* (§0.2) and
*"now decisive"* (§6.1) provided no protection.

*Files.* `examples/Revision_v1/a4_preflight_spectral_screen.m`, `a4_pre_screen.json`.
*Fix.* Add early checkpoints `{2, 5, 10, 20, 30, 50}` and require the gate to report the
mode window needed at each.

---

**M-7 — `INDETERMINATE` is not a pre-registered outcome.**

Spec §5.3 pre-registers exactly four outcomes. The emitted decision is `INDETERMINATE`,
justified as *"only a Class B arm may serve as the accuracy reference."* The specification
says only that Class B arms may serve as the reference; it does not define the case where
the `N = ∞` arm itself is Class C, nor authorize a fifth outcome. Introducing one after
seeing the data is precisely the retrofitting §5.3 exists to prevent — even though here it
runs *against* the authors' interest.

*Fix.* Either amend §5.3 in advance of the re-run to define this case, or make the frozen
arm converge (see PR-1) so the question does not arise.

---

**M-8 — Required validators produced no artifacts.**

| Validator | Required assertion | Artifact found |
|---|---|---|
| V-A4-1 | `MAC(Φ₀ᵘⁿⁱᶠᵒʳᵐ, Φ₀ˢᵒˡⁱᵈ) ≥ 0.9999`; ω₀² ratio matches `a/b`; design invariant | **none** |
| V-A4-5 | Determinism: replay of any arm reproduces its result | **none** |
| V-A4-6 | `N = ∞` bit-identical to pre-R-1 solver | pre-run unit tests only, at 40×5 |
| V-A4-3 | Observed count `= 1 + ⌊(n_iter−1)/N⌋` | implementation checks `⌊n_iter/N⌋` — a different formula |

Spec §7.7 additionally requires a determinism replay record in the provenance block. There
is none. `A4_IMPLEMENTATION_REPORT.md` §5 documents these as passing on a **40×5 mesh with
≤6 iterations** and explicitly states *"the tiny-mesh validation numbers are plumbing only
and are never evidence."* No production-scale validator evidence exists.

---

**M-9 — No baseline frequency is recorded anywhere, so no gain is computable.**

Spec §3.3 makes the reference-design choice load-bearing precisely because it fixes every
reported gain ratio `ω̃₁/ω₁⁽⁰⁾`. A4 records no `ω₁⁽⁰⁾`. From my independent model:

- solid domain (A4's declared baseline): `ω₁⁽⁰⁾ = 136.483` → SS-beam gain **1.169×**
- uniform `x = V_f = 0.5`: `ω₁⁽⁰⁾ = 68.24` → gain **2.338×**

A reader of A4's artifacts cannot compute either. The factor-of-two spread between the two
readings is exactly Reviewer 2's item C2, which §3.4 tasks A4 with settling.

---

### MINOR

- **m-1** — **Figure 1 misrepresents its own data.** The spec's primary figure requires the
  `±δ` equivalence band; it is absent. The y-axis auto-scales to
  `[159.565, 159.605]` — a **0.025%** range — while δ = 5% (≈ ±8 rad/s). Two points
  differing by 0.022% are rendered at opposite corners of the plot. Printed in a
  manuscript this communicates the opposite of the finding. Fix: draw the δ band and set
  the y-limits to contain it.
- **m-2** — Figure 4 shows 2 of the 5 panels required by §7.6 (unavoidable given C-1), and
  the class label renders as `ACCEPTED_WITH_BREAKDOWN` with MATLAB TeX subscripting the
  underscores (`ACCEPTED_WITH_BREAKDOWN` → `ACCEPTED̲WITH̲BREAKDOWN`). Set
  `'Interpreter','none'`.
- **m-3** — Only 3 of 7 specified figures exist (§7.6); see M-4 for why the other 4 cannot
  currently be produced.
- **m-4** — `a4_eigenpair_refresh_results.mat`, all 4 PNGs and both topology CSVs are
  **not git-tracked** (only the 6 JSON/MD files are). The `.mat` is a *required artifact*
  in `a4_stage_result.json` yet cannot be recovered from the repository.
- **m-5** — `a4_manifest.json > files` omits `a4_stage_result.json` and
  `a4_stage_manifest.json`, which `a4_stage_manifest.json` does list. The two manifests
  disagree on the artifact set (10 vs 12 files).
- **m-6** — `a4_mode_screen.m` defaults `lowDensityThreshold = 0.05`; spec §4.3.1's
  reference values were computed at `x < 0.1`. Undeclared threshold drift.
- **m-7** — Manuscript `main.tex:665` quotes SS-beam MAC `0.9998`; A4 measures
  `0.99963`. Small, but the manuscript cites a number A4 does not reproduce.
- **m-8** — `N = ∞` serializes to JSON `null`, indistinguishable from a missing field. The
  `.mat` correctly stores `Inf`. Use the string `"inf"` in JSON.
- **m-9** — `commit_sha` records `2c945de`; HEAD is `03729b4`. I verified
  `git diff 2c945de HEAD` is empty for all A4 code and the solver, so the artifacts are not
  stale — but the recorded SHA no longer identifies a reachable working state.
- **m-10** — `estRuntimeSeconds = 57600` (16 h) vs 526 s actual. The registry estimate is
  110× off, which would have masked a silently-truncated run.

---

### What is sound (verified, not assumed)

These held up under independent replication and are worth stating, because the failures
above are interpretive, not numerical:

- **The FE model and eigensolve are correct.** My independent Q4 plane-stress model,
  written from the documented element convention and driven only by the published topology
  CSVs, reproduces `ω₁ᵗʳᵃᶜᵏ` to **7 significant figures** for both surviving arms
  (159.565627 / 159.601173).
- **`ω₁ − ω₂` gaps verified**: 67.373 (`N=∞`), 65.976 (`N=50`) — matches reported. No
  coalescence at the endpoint; Reviewer 2's concern is answered negatively for this case.
- **Cross-file consistency is clean.** I diffed every scalar across `a4_result.json`, the
  `.mat`, `a4_table.md`, `a4_manifest.json` and `a4_stage_result.json` for all 5 arms: **no
  value changes between files.** Stage 2 of this audit found zero internal inconsistencies
  of that kind.
- **The driver's refresh telemetry is faithful.** The recorded `N=50` refresh at iteration
  50 (`index 5`, ω 159.8795) is reproduced exactly by my independent re-run.
- **The final topologies are physically sound**: single 4-connected solid component,
  material present in both support columns, `|V−V_f|/V_f ≈ 9e-5` and `3e-5`, grayness 0.097.
- **Determinism holds** in practice: my re-runs of the frozen trajectory reproduced the
  published endpoint and the recorded refresh selection exactly.

---

## 3. Scientific assessment

> **A4 does not demonstrate its intended scientific claim.**

**What it set out to do** (§1.1): locate the refresh interval `N*` at which the frozen
eigenpair ceases to be a valid proxy, and by what mechanism.

**What it actually established:** that the §4.3.1 admissibility screen, applied with a
20-mode window, cannot find the physical mode on designs between roughly iteration 2 and
40 of *any* arm. This is a statement about the diagnostic, not about the approximation.

The reported causal chain — *refreshing at small `N` → contaminated spectrum → refresh
reference unavailable → INDETERMINATE* — has an unsupported first link. The correct chain
is *small `N` → screened earlier → screened inside the void-mode window → window too narrow
→ terminated*. Refreshing is not implicated.

**Alternative interpretations that remain fully open**, none excluded by the evidence:

1. Refreshing at `N ≤ 10` is entirely benign and the arms would have converged normally
   (most consistent with `N = 50`'s clean result and near-identical topology).
2. Refreshing at small `N` genuinely destabilizes via the omitted `∂f/∂x` (the B4
   mechanism) — **untestable as run**, because `omitted_term_ratio` is `NaN` in every arm
   (`A4_IMPLEMENTATION_REPORT.md` §8 flags this honestly in advance).
3. The void-mode population is itself design-path-dependent, so `N` does affect
   diagnosability even if not accuracy.

The experiment cannot distinguish these. That is the definition of an inconclusive result,
and `INDETERMINATE` is — by accident — the honest label. But the *stated reason* for it is
wrong, and the reason is what would be published.

**The one defensible positive result A4 owns, and discards:**

> `N = 50` (Class B, converged 541/2000, MAC 0.99963, `j* = 1`, feasible, single component)
> attains `ω₁ᵗʳᵃᶜᵏ = 159.6012` against the frozen arm's `159.5656`: **+0.022%**, i.e.
> **227× inside the pre-declared δ = 5%**, with a visually indistinguishable topology.

Ten eigenpair refreshes changed the answer by two parts in ten thousand. On this benchmark
that is meaningful evidence for **H₀** and directly bears on `main.tex:704` ("the frequency
gain will be suboptimal relative to formulations that update the eigenpair") — which it
does **not** support. It is weakened, though not voided, by the frozen arm having capped
(M-2) rather than converged.

**Manuscript claims and their status after A4:**

| Claim | Status |
|---|---|
| `main.tex:661` — "robustness … across structurally different problem classes" | **Unsupported.** A4 covers one class, as the spec's own Part 8 concedes. Must be narrowed. |
| `main.tex:704` — refresh beats frozen | **Contradicted, weakly.** The one clean refreshed arm gains 0.022%. Must be softened, not evidenced. |
| `main.tex:665` — "MAC 0.9998, cleanest tracking" | Measured 0.99963. Reconcile the number. |
| Efficiency claim (one eigensolve) | **Supported.** V-A4-4 confirmed: `N=∞` performs zero in-loop eigensolves, `eigensolves_analytic = 2`. |

---

## 4. Reviewer #2 perspective

Reading the current artifacts cold, with no goodwill:

1. **"Three of your five arms produced no data, and you call it a result."** `iterations: 0`
   on 60% of the design. You do not report what happened in the iterations before each
   crash. I cannot audit a failure you did not record.
2. **"Your contamination diagnosis contradicts your own printout."** Your exception message
   says `Solid components = 1` and your classifier says "disconnected island". Which is it?
3. **"Your Table A4-1 says thresholding drops the frequency from 159.6 to 26.5."** By your
   own §4.2, that means the whole result is a gray-material artifact. You then classify the
   arm as clean and eligible as an accuracy reference. You cannot have both.
4. **"Your published method did not converge."** `N = ∞` hits 2000/2000 with
   `Δx = 3.0e-3`, three times the tolerance. The manuscript claims convergence without
   continuation. Which arm supports that claim?
5. **"You labelled the published method B4 and then wrote '(unattributed)' in the same
   sentence."** Delete the code or measure the mechanism.
6. **"Your primary figure has a y-axis spanning 0.025% and no equivalence band."** The two
   points differ by less than one part in four thousand and you have plotted them at
   opposite corners. Redraw it with the δ = 5% band you pre-registered.
7. **"Your gate checked iterations 100–2000 and your arms died at 2–30."** What was that gate
   for?
8. **"Your `N = 1` arm ran for 1.27 seconds."** You budgeted 2.5 hours. Nothing was learned
   about the fully design-dependent limit — the regime your own §6.2 identifies as the most
   fragile.
9. **"You promised the omitted-term ratio as the covariate distinguishing B3 from B4."**
   It is `NaN` everywhere. Your B3/B4 assignment therefore rests on code ordering, not
   evidence — `check_a4_run.m` gives B3 unconditional priority.
10. **"Your config hash is `ffffffff`."** That is not a hash. What exactly did V-A4-2 verify?
11. **"You still ran only the easy case."** Your own spec (Part 8, point 1) predicted I would
    say this. The clamped beam and building are where the mode migrates to topo-mode 3 and
    where topo-modes 1–2 show MAC < 0.10. You tested where you already believed the answer.
12. **"α = 1 only."** A frozen `Φ₂` under a weighted load is *a priori* the fragile case.
13. **"My first suggestion is still unaddressed"** — a benchmark where initial and optimized
    mode shapes differ substantially. Nobody built one.
14. **"Where is `ω₁⁽⁰⁾`?"** You cannot report a gain, and my item C2 (which reference design?)
    is untouched by these artifacts.

Items 1–10 are new and are all self-inflicted; 11–14 the specification anticipated.

---

## 5. Publication readiness

> **Not publishable in its current form.** Not as a figure, not as a table, not as a
> reviewer response, not as supplementary material.

Publishing Table A4-1 or Figure 1 as they stand would put three demonstrably false claims
into the record: that refreshing contaminates the spectrum (C-1), that the design's
frequency is a gray-material artifact (C-3), and that the published method exhibits a
sensitivity-omission instability (M-2).

### Required before A4 can be cited

**Blocking — must be fixed and the sweep re-run**

| # | Action | Addresses |
|---|---|---|
| PR-1 | Make the mode window adaptive (expand to a declared ceiling; record the window used per refresh). Re-run all five arms. | C-1 |
| PR-2 | Screen every arm on a common `N`-independent iteration grid; screen `N = ∞` too. | C-2 |
| PR-3 | Fix the threshold void floor to the declared `rho_min`; add a MAC guard. Re-issue Table A4-1. | C-3 |
| PR-4 | Fix the hash (wrap, or SHA-256); add a two-different-files regression. | C-4 |
| PR-5 | Implement the `ω₁ᵗʰʳᵉˢʰ` Class B criterion; add a classifier fixture. | M-1 |
| PR-6 | Separate "capped, unattributed" from B4. | M-2 |
| PR-7 | Retain and persist per-iteration histories for all arms; harvest them in the `catch`. | M-3, M-4, M-5 |
| PR-8 | Extend Gate A4-Pre to early checkpoints. | M-6 |
| PR-9 | Record `ω₁⁽⁰⁾` for both candidate baselines; run V-A4-1. | M-9 |
| PR-10 | Produce a determinism replay record at 400×50 (V-A4-5). | M-8 |

**Needed to make the frozen arm a usable reference**

- PR-11 — `N = ∞` capped at 2000 with `Δx = 3.0e-3`. Either raise the cap until it converges
  (`N = 50` reached `9.8e-4` in 541 iterations, so the frozen arm's non-convergence is itself
  a finding worth reporting), or amend §5.3 to define the decision when the reference arm is
  Class C. As it stands, the manuscript's convergence claim is contradicted by A4's own
  primary arm.

**Needed to answer Reviewer 1 rather than merely engage them**

- PR-12 — Authorize the omitted-term-ratio export (`A4_IMPLEMENTATION_REPORT.md` §8). Without
  it B3 and B4 cannot be distinguished by evidence, and the CR2 interaction at small `N`
  stays unattributable.
- PR-13 — One adversarial case where `Φ₀` is a known-poor proxy. This is Reviewer 1's *first*
  suggestion and remains untouched; small and cheap is fine.
- PR-14 — Either extend to the clamped beam / building, or state explicitly in the
  limitations that the frozen-eigenpair penalty is quantified **on the SS beam at α = 1
  only**. Narrow `main.tex:661` regardless.

### After PR-1…PR-11, the likely outcome

Based on what is already measurable: expect all five arms to complete, with
`ω₁ᵗʳᵃᶜᵏ` clustered within ~0.1% and decision-rule **outcome 1 (H₀ retained)**. That is a
publishable, defensible result — "refreshing the eigenpair confers no measurable benefit on
this benchmark; `main.tex:704`'s directional claim is softened" — and it is well within
reach. A4 is roughly one corrected re-run away from a genuine finding. It is not there yet.

---

## Appendix — audit reproduction

Scripts written for this audit (scratchpad, not committed):

| File | Purpose |
|---|---|
| `verify_a4.py` | Independent Q4 plane-stress FE replication of `ω₁ᵗʳᵃᶜᵏ`, `ω₁ᵐⁱⁿ`, `ω₁ᵗʰʳᵉˢʰ` at three void floors, from the published topology CSVs only. |
| `a4_audit_screen.m` | Applies `a4_mode_screen` to the frozen trajectory at iterations {2,5,10,15,20,25,30,40,50,100}. Establishes C-2. |
| `a4_audit_window.m` | Re-screens iterations 25 and 30 with 20 / 60 / 120-mode windows. Establishes C-1. |

The hash defect (C-4) is reproducible in three lines of Python simulating MATLAB's
saturating `uint32` multiply.
