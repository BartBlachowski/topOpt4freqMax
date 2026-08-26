# Diagnostic — premature termination and `omega_1 = 0` for `OlhoffDu2007Repro`

**Date:** 2026-08-26
**Primary case:** 240×30
**Status:** root cause identified; **benchmark configuration mapping corrected and verified** (§9, §10).
**Constraint honoured:** no `reproduction2007` numerical source file modified — all 61 remain
SHA256-identical to the clean-room import; no full performance campaign re-run.

---

## 1. Verdict

**Root cause: CONFIGURATION MAPPING BUG**, with two consequential secondary
defects that turn it into a silent failure.

`tools/Matlab/run_topopt_from_json.m` mapped the benchmark JSON's
`void_material.rho_min` onto the reproduction's `cfg.rhomin` (pre-fix line 368;
corrected in §9):

```matlab
runCfg.rho_min = rho_min;    % rho_min = cfg.void_material.rho_min = 1e-6
```

These are **different physical quantities**:

| | meaning | value |
|---|---|---|
| `void_material.rho_min` (benchmark JSON) | void **material density** floor for the benchmark's own interpolation | `1e-6` |
| `cfg.rhomin` (reproduction) | **design-variable lower bound** ρ_min, Du & Olhoff (2007) **eq. (7e)** | `1e-3` |

The reproduction therefore ran with ρ_min = 1e-6 instead of 1e-3. Under its
mass interpolation eq. (4) (`ρ ≤ 0.1 → ρ⁶`) and SIMP `p = 3`, that changes the
void element contributions by **nine and eighteen orders of magnitude**:

| ρ_min | SIMP stiffness ρ³ | eq. (4) mass ρ⁶ | K diag ratio | M diag ratio |
|---|---|---|---|---|
| **1e-3** (documented) | 1e-09 | 1e-18 | 4.0e+09 | 4.0e+18 |
| **1e-6** (benchmark) | 1e-18 | 1e-36 | **4.0e+18** | **4.0e+36** |

MATLAB's own eigensolver flags this directly. At ρ_min = 1e-6, `eigs` warns:

```
First input matrix is close to singular or badly scaled (RCOND = 1.589452e-19)
```

`RCOND = 1.6e-19` is below double precision `eps`. No warning is issued at
ρ_min = 1e-3.

---

## 2. First point of deviation

### In the execution chain

`tools/Matlab/run_topopt_from_json.m`, the `OlhoffDu2007Repro` dispatch case
(pre-fix line 368). Everything upstream (`performance_comparison.m`) and downstream
(`run_repro2007` → `repro2007_config` → `olhoffOpt`) faithfully carries the
wrong value.

### In the numerics

**Outer iteration 101.** Iterations 1–100 are **bit-identical** to the frozen
240×30 reproduction (`baseline/lp240_rmin1.3.mat`, `max|Δω| = 0`).

```
 iter      frozen w1       bench w1       max|dw|
    99   149.54032982   149.54032982     0.000e+00
   100   150.38449796   150.38449796     0.000e+00
   101   151.40087629    84.63550704     5.456e+02   <<< FIRST DIVERGENCE
   102   151.64434030   150.57566383     1.597e+01
   103   152.10908694    46.32358666     5.446e+02
   104   152.32753803   150.79659642     1.067e+01
```

At iteration 101 the benchmark spectrum is `84.6355 / 84.6424 / 88.1404` —
three modes crushed together, the signature of spurious modes from a
numerically singular pencil, not a physical change. It then flickers: 102
recovers, 103 collapses again, 104 recovers.

**Why exactly iteration 101 is not a coincidence.** The initial design is
uniform ρ = 0.5 and the move limit is 0.005, so the first elements reach the
void floor after 0.5 / 0.005 = **100** iterations. `cfg.rhomin` enters the
computation only through the LP box `lo = max(rhomin - rho, -move)` and the
update clamp — i.e. only once densities arrive at the floor. Volume first
diverges at iteration **100**; the spectrum at **101**. The wrong floor becomes
observable at precisely the iteration it first becomes reachable.

---

## 3. Configuration audit at the `olhoffOpt` boundary

Captured from the real benchmark path at 240×30 and compared field by field
against `repro2007_config('fig3a_best')`. Only four fields differ:

| field | frozen | benchmark path | verdict |
|---|---|---|---|
| `rhomin` | **0.001** | **1e-06** | **UNINTENDED — root cause** |
| `rminEl` | 1.3 | 2 | intended (benchmark's shared filter), documented |
| `tolOuter` | 0.001 | 0.003 | unintended, from `optimization.convergence_tol`; **not causal here** (see §4) |
| `verbose` | true | false | cosmetic |

All of `nelx`, `nely`, `rminPhys`, `move`, `tolMult`, `innerSolver`,
`filterMode`, `massInterp`, `offDiag`, `support`, `axial`, `bc`, `elemType`,
`massType`, `p`, `rho0`, `volfrac`, `n`, `Nmax`, `maxInner`, `tolInner`,
`minInner`, `solver`, `threads`, `E`, `nu`, `rhom`, `a`, `b`, `t` are identical.

---

## 4. Why the run terminates before `maxOuter = 1600`

The comment in `performance_comparison.m` — that this route is move-saturated
and therefore always stops at `max_outer_iterations` — is an **over-claim**.
Move saturation is a property of *successful* LP solves, not of the method.

`olhoffOpt` carries a native stop test:

```matlab
if dxOuter < cfg.tolOuter,  break,  end     % algo/olhoffOpt.m
```

and `innerLoopLP` converts any LP failure into a zero step:

```matlab
if flag ~= 1 || isempty(x)
    drho = zeros(NE,1);        % <<< failure becomes "no change"
    st.beta = ctx.lam(1);
    return
end
```

A failed LP therefore yields `dxOuter = 0`, which satisfies **any** positive
tolerance. The observed `final_max_density_change = 0` **exactly** — not merely
small — is the fingerprint.

`tolOuter = 3e-3` vs the documented `1e-3` is a real second mapping slip, but it
is **not** what ended these runs: `dxOuter` was exactly `0`, which is below both.

---

## 5. Last iterations before termination (case B, r_min = 2.0, ρ_min = 1e-6)

```
 iter      omega1      omega2      omega3   gap_rel   N   objective    d_inf     vol lpflag conv
  584  0.00064123      20.033      22.406 3.124e+04   1  0.00112528    0.005  0.22049     1    1
  585      47.499      53.102      69.149    0.1179   1     64.6776    0.005  0.21729     1    1
  586  0.00043612  0.00051185  0.00077777    0.1736   1 0.000605995    0.005  0.21453     1    1
  ...
  592           0  0.00016152  0.00024032       Inf   1           0    0.005  0.19982     1    1
  593           0   0.0003044  0.00035899       Inf   1           0        0  0.19982    -2    0
```

Answering the four alternatives posed:

- **Does the Eq. (22) LP legitimately return `drho = 0`?** No.
- **Does `linprog` fail and get converted to zero?** **Yes.** `lp_flag = -2` at
  iteration 593 — `linprog` reports the problem **infeasible** — and
  `innerLoopLP` returns `drho = zeros(NE,1)`.
- **Does clipping/projection zero a nonzero increment?** No. `d_inf = 0.005`
  (full move) right up to iteration 592.
- **Does a wrapper stopping rule cause termination?** No. The termination is
  native `olhoffOpt`. The wrapper only *labels* it.

**Why the LP became infeasible.** `innerLoopLP` normalizes the whole constraint
system by `lamref = ctx.lam(1)`, the current λ₁:

```matlab
A(j,1:NE) = -ctx.F(:,j,j).'/lamref;      b(j) = ctx.lam(j)/lamref;
```

By iteration 592 the corrupted eigensolve returns λ₁ = 0 exactly (`omega1 = 0`,
`objective = 0`). The normalization divides by zero, the constraint matrix is
no longer finite, and `linprog` returns `-2`.

Note the collateral damage upstream of the failure: **volume bled from 0.5 to
0.1998** between roughly iterations 269 and 593, because the LP was ascending
gradients computed from spurious modes. Final `max(ρ) = 0.68` — nothing in the
design is solid.

---

## 6. Four frequency sources, separated

For 240×30, r_min = 2.0, ρ_min = 1e-6 (the failing benchmark case):

| source | ω₁ | ω₂ | ω₃ |
|---|---|---|---|
| native `info.native.omega` (`olhoffOpt` final re-analysis) | **0** | 3.044e-4 | 3.590e-4 |
| returned by `run_repro2007` | **0** | 3.044e-4 | 3.590e-4 |
| returned by `run_topopt_from_json` | **0** | 3.044e-4 | 3.590e-4 |
| benchmark post-hoc eigensolve (`localSolveTopologyModes`) | **73.1793** | — | — |

The first three are identical by construction: `run_repro2007` returns
`localToVec3(res.omega)`, and `run_topopt_from_json:428` assigns `omega` once
and **never overwrites it** — `modesT.omega` from the post-hoc eigensolve is
used only for snapshots and correlation output.

**`omega_1 = 0` therefore originates in the native solver, not in benchmark
postprocessing.** The post-hoc evaluator, which uses `E_min_ratio = 1e-6` on
*stiffness* (a far better-conditioned pencil), reports 73.18 rad/s on the very
same density field. That is exactly the FE/void-interpolation divergence already
warned about in `MIGRATION_REPRODUCTION2007_REPORT.md` §14.2 — here it is
incidental, and it is the healthier of the two numbers.

---

## 7. The two controlled 240×30 diagnostics (as requested)

Both through `run_topopt_from_json`, `move = 0.005`, `maxOuter = 1600`, nothing
else changed.

| | **A** r_min = 1.3 (frozen control) | **B** r_min = 2.0 (benchmark setting) |
|---|---|---|
| iterations | 1600 | **593** |
| stop reason | `max_outer_iterations` | `outer_increment_below_tolerance` |
| native ω₁–ω₃ | 170.9281 / 173.9301 / 280.0587 | **0 / 3.044e-4 / 3.590e-4** |
| benchmark returned ω₁–ω₃ | identical to native | identical to native |
| post-hoc eigensolve ω₁ | 170.4199 | 73.1793 |
| LP status (final) | flag 1, converged | **flag −2, failed** |
| LP failures over run | 0 | 1 (iteration 593) |
| final max increment | 0.005 | **0** |
| final N / eigengap | 2 / 0.01796 | 1 / Inf |
| final volume | 0.5 | **0.1998** |
| final grayness | 0.0952 | 0.4437 |
| density checksum | n=7200 sum=3600 norm=58.5548 min=1e-06 max=1 | n=7200 sum=1438.674 norm=25.2977 min=1e-06 max=**0.68** |

**Case A did not escape either.** With ρ_min = 1e-6 its ω₁ dips below 1 at
iteration 393 (min 0.6147) and it ends at 170.93 / 173.93 / 280.06 with a
1.8 % gap — *not* the frozen 170.4709 / 170.8659 / 285.1939 with its 0.23 % gap.
The control run completes, but it is already numerically corrupted.

### Confirmation run — the causal test

Same benchmark entry point, **only** `void_material.rho_min` changed to `1e-3`:

| | r_min = 1.3 | r_min = 2.0 |
|---|---|---|
| iterations | **1600** | **1600** |
| stop reason | `max_outer_iterations` | `max_outer_iterations` |
| ω₁–ω₃ | **170.4709086 / 170.8658865 / 285.1939392** | 167.0409 / 176.8491 / 292.6652 |
| LP failures | **0** | **0** |
| min ω₁ over run | 68.3209 | 68.3209 |
| final volume | 0.5 | 0.5 |
| final N / gap | 2 / 0.00293 | 1 / 0.0585 |
| density min | 0.001 | 0.001 |

At r_min = 1.3 the benchmark path now reproduces the frozen result **to full
double precision** — the measured residual `4.030562e-08` is exactly the
rounding of the 10-digit literal used in the comparison
(`170.47090864030562 − 170.4709086 = 4.030562e-08`).

At r_min = 2.0 it runs all 1600 iterations, holds volume at 0.5, never goes near
zero, and lands at 167.04 / 176.85 / 292.67 — consistent with the
`NOTES.md` §6 240×30 family (r_min = 2.2 → 166.98 / 178.99 / 291.6, non-bimodal).

---

## 8. Classification

| class | verdict |
|---|---|
| **CONFIGURATION MAPPING BUG** | **CONFIRMED — root cause.** `void_material.rho_min` (1e-6) mapped onto the design-variable bound ρ_min (paper eq. 7e, 1e-3). |
| **LP FAILURE / STATUS-HANDLING BUG** | **CONFIRMED — secondary.** `innerLoopLP` returns `drho = 0` on `flag ≠ 1`, making failure indistinguishable from a converged zero step. Also: normalizing by `lamref = λ₁` has no guard against λ₁ → 0. |
| **FALSE CONVERGENCE / STOPPING BUG** | **CONFIRMED — secondary.** `olhoffOpt` reads that zero step as convergence; `run_repro2007`'s `localStopping` reports `outer_increment_below_tolerance` even though `inner_converged = false` and `lp_flag = -2` were both available to it. |
| NATIVE OPTIMIZATION FAILURE AT R_MIN = 2 | **REFUTED.** With ρ_min = 1e-3, r_min = 2.0 completes 1600 iterations at volume 0.5 with sane frequencies. |
| POST-HOC EIGENSOLVE / EVALUATOR BUG | **REFUTED.** `ω₁ = 0` is produced by the native solver; the post-hoc evaluator reports 73.18 on the same field and never overwrites the returned value. |
| OTHER | Minor: `tolOuter` silently mapped 1e-3 → 3e-3 from `convergence_tol`; not causal here but would move the stop point elsewhere. And the `performance_comparison.m` comment claiming this route always stops at `max_outer_iterations` is an over-claim. |

Note that the LP/stopping defects are **pre-existing in the frozen, unmodified
`olhoffOpt.m` / `innerLoopLP.m`** — they are masked in the frozen
configurations because the LP never fails there. The mapping bug is what exposed
them. This is consistent with the migration regression: it verified that the
migrated code reproduces the frozen behaviour, not that the code is defect-free
under configurations the clean-room study never ran.

---

## 9. Corrections applied

Scope as instructed: **benchmark configuration mapping only.** Nothing inside
`Matlab/reproduction2007/` was touched — the 61 imported files remain
SHA256-identical to the clean-room source, so the WP2/WP5 provenance guarantee
is intact.

### 9.1 `tools/Matlab/run_topopt_from_json.m` — stop inheriting two settings

The `OlhoffDu2007Repro` dispatch case no longer copies `void_material.rho_min`
or `optimization.convergence_tol` into the solver configuration:

```diff
-            runCfg.tol_outer     = convTol;
...
-            runCfg.rho_min       = rho_min;
+            % NOT INHERITED FROM THE SHARED BLOCK -- see this document.
+            %   rho_min   : void_material.rho_min is a void MATERIAL DENSITY
+            %               floor (1e-6).  cfg.rhomin is the DESIGN VARIABLE
+            %               bound of eq. (7e) = 1e-3.  Different quantities.
+            %   tol_outer : convergence_tol (3e-3) is the benchmark's shared
+            %               tolerance; this method's documented value is 1e-3.
+            % Both now come from the reproduction's own configuration, and are
+            % overridable only through the explicit optimization.repro2007
+            % block, so that any deviation is visible in the task file.
```

`rho_min` was added to the `optimization.repro2007` optional-key list
(`tol_outer` was already present), so both remain settable — but only
deliberately, by name, in the task file.

### 9.2 `examples/Performance/performance_comparison.m` — state them explicitly

```matlab
data.optimization.repro2007 = struct( ...
    'support_type', 'SS', ...    % bc.supports are closest_point at mid-height
    'move',         0.005, ...   % documented fig3a_best value
    'max_outer',    1600, ...    % documented fig3a_best budget
    'rho_min',      1e-3, ...    % paper eq. (7e); NOT void_material.rho_min
    'tol_outer',    1e-3);       % documented fig3a_best value
```

The surrounding comment block was corrected at the same time. Two statements in
it were wrong and are now fixed:

- it claimed the native stop test "can never fire". It can — on LP failure. The
  text now says move saturation holds *while the LP solves successfully*, and
  adds: **a stop reason other than `max_outer_iterations` means the LP failed
  and must be investigated, not read as convergence.**
- it listed "two" non-transferable settings; there are four.

### 9.3 Deliberately NOT applied

- **`localStopping` reporting `lp_failure`** (follow-on 2 of the original §9).
  This is a stopping-label fix, not a configuration mapping, and was outside the
  instructed scope. `run_repro2007` still reports
  `outer_increment_below_tolerance` when the LP fails, even though
  `inner_converged == false` and `lp_flag` are available to it. **Still open.**
- **`innerLoopLP`'s failure-to-zero return and its unguarded `1/lamref`
  normalization** (follow-on 3). These are inside the byte-identical frozen
  files. **Still open, and still requiring an explicit provenance decision.**

Both remain latent. The configuration fix removes the trigger observed here; it
does not remove the failure modes themselves.

## 10. Post-fix verification

Two 240×30 controls re-run through the benchmark entry point
(`run_topopt_from_json`), with the shared block deliberately left at its
hostile values to prove they no longer leak:

```
shared void_material.rho_min = 1e-06   shared convergence_tol = 0.003   shared move_limit = 0.2
```

| | **r_min = 1.3** (frozen control) | **r_min = 2.0** (benchmark setting) |
|---|---|---|
| iterations | **1600** | **1600** |
| stop reason | `max_outer_iterations` | `max_outer_iterations` |
| ω₁–ω₃ | **170.47090864030562 / 170.86588649270706 / 285.19393916368136** | 167.04089205 / 176.84910752 / 292.66517568 |
| final max increment | 0.005 | 0.005 |
| LP failures | **0** | **0** |
| lp_flag / converged (last) | 1 / true | 1 / true |
| final N / eigengap | 2 / 0.00293194 | 1 / 0.0585134 |
| final volume | **0.5** | **0.5** |
| final grayness | 0.1032074006 | 0.1349726891 |
| min ω₁ over the run | 68.3209 (never below) | 68.3209 (never below) |
| outer / inner | 1600 / 1600 | 1600 / 1600 |
| density checksum | n=7200 sum=3600 norm=58.4313843662145 min=0.001 max=1 | n=7200 sum=3600 norm=57.9400479767184 min=0.001 max=1 |
| **ρ floor in force** | **0.001** (shared value 1e-6 did not leak) | **0.001** |
| **tolOuter in force** | **0.001** (shared value 3e-3 did not leak) | **0.001** |
| wall | 163.6 s (0.1016 s/iter) | 288.6 s (0.1801 s/iter) |

### Bit-identity against the frozen baseline

`r_min = 1.3` versus `Matlab/reproduction2007/baseline/lp240_rmin1.3.mat`:

| quantity | result |
|---|---|
| final frequencies (n=3) | **PASS — bit-identical** |
| final density field (n=7200) | **PASS — bit-identical** |
| omega history (n=8000) | **PASS — bit-identical** |
| multiplicity N (n=1600) | **PASS — bit-identical** |
| max\|Δρ\| history (n=1600) | **PASS — bit-identical** |
| volume history (n=1600) | **PASS — bit-identical** |
| beta history (n=1600) | **PASS — bit-identical** |

**7 / 7 bit-identical.** The benchmark entry point now reproduces the frozen
240×30 clean-room result exactly, not merely closely.

### Every §1 symptom resolved

| symptom (pre-fix, 240×30) | post-fix |
|---|---|
| terminated at 593 iterations | runs the full 1600 |
| `outer_increment_below_tolerance` | `max_outer_iterations` |
| `final_max_density_change = 0` | 0.005 (full move) |
| `linprog` flag −2 (infeasible) | flag 1, 0 failures |
| ω₁ = 0 | 170.4709086403 |
| volume melted to 0.1998 | 0.5 |
| max ρ = 0.68 (nothing solid) | 1.0 |
| divergence from frozen at iteration 101 | no divergence at any iteration |

`r_min = 2.0` is confirmed sound as a benchmark setting: 1600 iterations,
volume held at 0.5, no LP failures, ω₁ never near zero, landing at
167.04 / 176.85 / 292.67 — consistent with the `NOTES.md` §6 240×30 family
(r_min = 2.2 → 166.98 / 178.99 / 291.6, non-bimodal).

### Not re-run

The full performance campaign was **not** re-run, as instructed. The existing
`examples/Performance/table1_performance.csv` and `benchmark_results.json` on
disk still contain the **pre-fix** Olhoff rows (240×30, 320×40 and 400×50 with
`ω₁ = 0` and false convergence) and must be regenerated before use. The
160×20 row is not affected by the ρ_min bug in the same way — it completed 1600
iterations — but it too ran with ρ_min = 1e-6 and `tolOuter = 3e-3`, so it is
also superseded.

### Verification commands

```matlab
% both controls, through the benchmark entry point
data.optimization.approach = 'OlhoffDu2007Repro';
data.optimization.repro2007 = struct('support_type','SS','move',0.005, ...
    'max_outer',1600,'rho_min',1e-3,'tol_outer',1e-3);
[x, omega, tIter, nIter, mem, nIterStage, tel] = run_topopt_from_json(data);
```

```matlab
% migration gates, re-confirmed unaffected
addpath('Matlab/reproduction2007/runner');
repro2007_regression('prefix');
repro2007_verify_isolation();
```
