# Candidate-C, topology gate, reference and endpoint verification

Scope: is the **frozen** choice implemented correctly? Whether Candidate C was the right methodology is closed and not reopened.

## 1. Common evaluator — `analysis/three_method_parametric_study/study_evaluate_design.m`

Reached only through `ie2a.evaluate_common`, which asserts the frozen evaluator is on the path and passes `IncludeBinaryDiagnostic=false` in production.

| requirement | implementation | verdict |
|---|---|---|
| actual gray density | `x = max(0,min(1,double(x(:))))`; `Q = [selected_omega_raw_E1, _E2, _E3]` on the gray field | correct |
| E1 linear mass law with floor | `rr = 1e-6 + (1-1e-6)*z`; `Ee = 1e7*(1e-6+(1-1e-6)*z^3)` | correct |
| E2 Eq. (4a) | `g = z`, `g(z<=0.1) = 1e5*z^6`; `rr = 1e-9+(1-1e-9)*g`. Continuity at z=0.1: `1e5*1e-6 = 0.1` | correct |
| E3 Eq. (4a) + effective-density floor | `zeff = max(z,1e-3)`; Eq. (4a) applied to `zeff`; `Ee = 1e7*zeff^3`; `rr = g` | correct |
| unanimous classifier | `margins = [0.5-voidKE, 0.5-voidSE, dwp-0.5]`; `valid = eigenpairValid & diagnosticFinite & all(margins>0,2)` — strict `voidKE<0.5 AND voidSE<0.5 AND densityParticipation>0.5` | correct |
| IPR diagnostic only | `ipr` computed and stored; `IPR_role='NONBINDING_QA'`; absent from `margins` | correct |
| adaptive search 3→6→12→24→48→… | `requested = min(3, technicalLimit)`, then `requested = min(2*requested, technicalLimit)` | correct |
| no scientific mode ceiling | `TechnicalMaxModes = Inf` from `evaluate_common`, so `technicalLimit = nFree-1` | correct |
| fail closed | returns `STRUCTURAL_MODE_NOT_FOUND` when the batch reaches `technicalLimit` with no valid mode | correct |
| determinism | `deterministic_v0` seeds `RandStream('twister', 42)` for the `eigs` start vector | correct |

**No hidden fallback.** `valid = find(batch.valid_structural, 1, 'first')` selects the lowest *valid* mode — never the lowest eigenmode, never a fixed first-three, never Candidate D, never Eq. (4), never a native E2/E3 interpretation. Binary spectra are computed only when explicitly requested and are stamped `ENDPOINT_MANUFACTURABILITY_TOPOLOGY_DIAGNOSTIC_EXCLUDED_FROM_Q`; `evaluate_common` builds `Q` exclusively from `selected_omega_raw_*`. Candidate-D / binary spectral evaluation has **not** entered `Q`.

BC in the evaluator pencil: `jMid = round(nely/2)`, both translational DOFs fixed at the left and right mid-height nodes — identical to `study_base_config`'s supports and to the frozen contract's `both_translational_dofs_at_both_midheight_end_nodes`.

### Anchors reproduced independently

| anchor | expected | reproduced | verdict |
|---|---|---|---|
| 480×60, k=194, E3 selected ordinal | 13 | **13** | matches |
| 480×60, k=194, batch schedule | `[3 6 12 24]`, 3 escalations | `[3 6 12 24]`, 3 | matches |
| 480×60, k=194, ordinals (E1,E2,E3) | — | `[1 7 13]` | ordinal > 3 and > 12 both exercised |
| 480×60, k=194, classifier margins | all > 0 | `[0.4995, 0.4999, 0.4458]` | unanimous, comfortable |
| 96×12, k=252 checkpoint identity | binary + gate identical | identical | matches |

**Maximum verified anchor ordinal: 13.** Adaptive-search failures observed: 0.

## 2. Hard topology gate — `ie2a.topology_metrics`

```
topology_pass = requiredConnected && strictDetachedPass
hard_gate_pass = volume_pass && topology_pass
```

- **exact-count projection**: `ie2a.exact_count_binary` — `nSolid = round(vf*numel(x))`, `sortrows([-x, index], [1 2])` so ties break on increasing global index. Matches the contract exactly.
- **support connectivity**: four-neighbour BFS; `supportRows = unique([nely/2, nely/2+1])` at columns 1 and `end` — the incident Q4 footprints of both mid-height support nodes. `requiredConnected` demands exactly one component spanning both.
- **per-component significance**: `strictDetachedPass = isempty(detachedAreas) || all(detachedAreas < A_sig)` with `A_sig = 0.01`. This is the **repaired per-component** rule.
- **no aggregate veto**: `aggregate_detached_area` and `n_islands_all` are computed and stamped `DIAGNOSTIC_ONLY`; neither appears in `topology_pass`. **Confirmed absent.**
- **volume**: `raw_volume_relative_error = |mean(x)-0.5|/0.5 <= 0.001`.
- `a_sig` by mesh derives to the contract's `[4 9 16 25 36 49 64 81 100]` (A_sig / (8·1/(nelx·nely))).
- `assert(mod(nely,2)==0)` — all nine production meshes have even `nely`.

## 3. Lossless storage

Authoritative Olhoff dtype: **`double`**. Enforced at four independent layers — `run_trajectory` asserts `isa(tr.x_post,'double')`; `validate_results` rejects `trajectory_dtype ~= 'double'`; `RESULT_SCHEMA.json` pins `const: "double"`; `preflight` requires `manifest.trajectory.authoritative_dtype == 'double'` (verified rejecting `'single'`).

Identity verified live: `isequal(r.rho, r.rho_snapshots(:,end))` holds; representation error 0 on the double path against 2.62e-08 on the historical single path. `reference_length_replay` re-checks exact-count-binary and hard-gate identity at checkpoints `[80 252 453 552 2100 3200]`, including the early regime responsible for the historical float32 failures.

**No authoritative float32 path is reachable by production.**

Proposed/Yuksel trajectory sufficiency (focused test): observer states are `double`, all finite, the final observed state bit-equals the returned design, `first_xPhysPrev` makes state 0 explicit, and exact-count-binary and hard-gate decisions are stable. Sufficient for Candidate C, projection, hard gate, reference and persistence.

## 4. Reference workflow

Chain: trajectory → `analyze_trajectory` (Candidate C + hard gate per state) → `H0 = Hhealth & Hvol & Htop & Hmethod` → `reference_phase` → `b_ref`, `Q_ref` → `measurement_budget` → `B_meas`.

`reference_phase` implements the frozen rule: `F(b)` is the cumulative best over base-valid P-windows; `gain(b) = (F(b)-F(b-L_ref))/F(b)`; `b_ref` is the first block endpoint `b = t·P` with all three gains `<= epsilon_ref`; `Q_ref = F(b_ref)`. `B_ref=3200`, `P=100`, `L_ref=500`, `epsilon_ref=0.001`. Guarded by `assert(bRef >= P + L_ref)` = 600.

- **No horizon-relative max has reappeared** — `F` is a cumulative *minimum-over-window*, then a running max; it never reads the trajectory's terminal or horizon value.
- **No cap fallback** — when no block endpoint qualifies, `b_ref` stays `NaN` and the status becomes `REFERENCE_NOT_ESTABLISHED` (verified live on a 400-state trajectory).
- `n = min(size(Q,1), B_ref)` truncates the tail at `B_ref`; verified stable when the trajectory is extended past 3200.

`measurement_budget`: `B_meas = min(max(B0, b_ref + P - 1), B_ref)` — **exactly** the frozen formula.

### Reference and endpoint anchors, independently reproduced

96×12 Olhoff-LP, lossless double, B_ref=3200:

| quantity | claimed | reproduced |
|---|---|---|
| reference status | PASS | **PASS** |
| `b_ref` | 2100 | **2100** |
| `Q_ref` | 162.66009, 162.99776, 162.99776 | **162.6601, 162.9978, 162.9978** |
| `requested_end` | 2199 | **2199** |
| `B_meas` (B0=3200) | 3200 | **3200** |
| tail truncated | false | **false** |

## 5. Endpoints

`scan_persistence`: `k_cert` is the first index at which the pass-run reaches `P`; `k_enter = k_cert - P + 1`. Causal — a single forward pass with no look-ahead. Partial trailing runs set `tail_incomplete` and leave both endpoints `NaN` (`NOT_REACHED`).

| P | q | k_enter (claimed) | k_cert (claimed) | reproduced |
|---:|---:|---:|---:|---|
| 100 | .98 | 229 | 328 | **229 / 328** |
| 100 | .99 | 309 | 408 | **309 / 408** |
| 100 | .995 | 453 | 552 | **453 / 552** |
| 50 | .98 / .99 / .995 | — | — | **229/278, 309/358, 453/502** |
| 200 | .98 / .99 / .995 | — | — | **229/428, 309/508, 453/652** |

`k_cert - k_enter = P-1` in all nine. Look-ahead check: truncating the pass matrix at `k_cert` leaves `k_enter` unchanged. **Verified.**

P = 50/100/200 and q = .98/.99/.995 are all produced per row by `build_rows` (9 rows per cell). Interrupted runs propagate through `classify_status`, whose precedence matches the contract.

Residual (MODERATE, F-06): `B_meas` and `tail_truncated` are computed at `P_primary` only and copied onto the P=50 and P=200 rows.
