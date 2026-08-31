# Iteration accounting audit

Audit-only. `ITERATION_ACCOUNTING_SPEC.md` was not modified. Every claim below was traced
to source or to a frozen artifact.

## 1. Proposed — verified correct

Source: `analysis/ourApproach/Matlab/topopt_freq.m`, profile
`proposed_practical_move02_tol001`, optimizer OC.

The counted loop (lines 412–678) performs, per iteration:

1. assemble `M(x)` from `rhoPhys = rho_min + xPhys.^pmass * (rho0 - rho_min)`;
2. assemble `K(x)` from `Emin + xPhys.^penal * (Emax - Emin)`;
3. one FE solve on the free DOFs for all load cases;
4. filter sensitivities (`ft == 0`: density-weighted sensitivity filter);
5. one OC update (`oc(...)`) with move limit 0.2, then restore passive densities;
6. re-filter to `xPhys = localPhysicalFieldFromDesign(...)`;
7. `change = max|x − x_old|` on the **raw design field**.

`nIter = loop` is the count. The post-loop eigensolve (~line 718) is outside the loop and is
correctly excluded from the count and timed separately. `xOut = xPhys(:)` (line 762), so the
return-equivalent state at `k` is the post-update filtered field — consistent with
`ACCEPTANCE_GATE_SPEC.md` §1.

**Conclusion: one counted Proposed iteration is one meaningful method-level OC update.
Claim verified.**

### One fact the spec omits and should state

With `semi_harmonic_baseline = 'solid'` and `reference_refresh_interval = 0`, line 441 reads

```matlab
dueFlags(k) = (loop == 1) && ~strcmp(harmonicBaseline, 'solid');
```

which is **false at every iteration**. The reference eigenpair is computed once before the
loop, from the fully solid design, and never refreshed. **A Proposed iteration contains no
eigensolve.**

This is the single largest source of per-iteration non-equivalence in the study. An Olhoff
outer iteration contains one generalized eigensolve, and at 800x100 that eigensolve is
**75 %** of the outer iteration cost (median `tEig` 1.368 s against `tEig+tGrad+tInner` =
1.817 s). It belongs in the accounting spec and in the Main Table 1 footnote, numerically
(Finding Mi5).

## 2. Yuksel — one factual error, one sound conclusion

Source: `analysis/YukselApproach/Matlab/top99neo_inertial_freq.m`, profile
`yuksel_practical_move01_tol001`.

**Stage 1** (`localComplianceLoop`): current physical field, `K(x)U = F_point`, compliance
sensitivity, one OC update. **Stage 2** (`localInertialLoop`): build `M(x)`, form the moving
inertial load from the current mode estimate, solve `K(x)U = F`, update the normalized mode
estimate, partial compliance sensitivity, one OC update. Both correctly described.

### The error

`ITERATION_ACCOUNTING_SPEC.md` states:

> "In the current MATLAB implementation its locally updated design variable is not returned
> into Stage 2; Stage 1 provides the displacement/mode estimate used to initialize the
> inertial stage."

The implementation, lines 230–247:

```matlab
info.stage1.xFinal = xPhys;
info.stage1.UFinal = U;
...
% Use stage-1 outputs as initial guesses for stage 2
x     = xPhys;          % <-- line 237
U_est = U;
...
[xPhys_stage2, U_stage2] = deal(xPhys, U);
```

**Stage 1's physical field becomes Stage 2's design variable, and Stage 1's displacement
becomes Stage 2's initial mode estimate.** The design state is carried forward. What *is*
discontinuous is the raw/physical identification: Stage 1's *filtered* field is adopted as
Stage 2's *unfiltered* design variable, a one-time re-filtering shift at the handoff.

**Why it matters:** the sentence is the spec's stated basis for excluding Stage 1 from
acceptance, and a referee who opens the file finds it in under a minute.

**What survives:** the exclusion itself. Stage 1 minimises compliance under a point load,
not the eigenfrequency objective — the protocol gives that reason too, and it is sound and
sufficient. And the correction *strengthens* the case for the chronological sum, because
the density trajectory really is continuous across the handoff.

### Is `N_total = N_stage1 + N_stage2` scientifically meaningful?

**Yes, as chronological work required before a Stage-2 result can be returned** — exactly
the framing the spec uses. The two iteration types are not equal work: Stage 2 additionally
assembles `M(x)` and forms a design-dependent inertial load each pass. Frozen per-iteration
times confirm the stages differ in cost, and the spec correctly requires separate stage
means.

### Main-table phrasing

`N_total` must never be column-headed "iterations". Head it **"chronological method-level
updates (S1 + S2)"** with `N_stage1` and `N_stage2` adjacent. `PROPOSED_TABLE_LAYOUTS.md`
Main Table 1 already does exactly this, with separate `Yuksel S1`, `Yuksel S2 enter/cert`
and `Yuksel total enter/cert` columns. **That layout is correct as drawn** and needs no
change.

### Budget interaction (Finding M6)

Stage 1 hit its own 1000-iteration cap at 640x80, 720x90 and 800x100 in the frozen campaign
(`IterStage1 = 1000` at all three; `max_iters: 1000` in the freeze manifest). The new
`B0_stage1 = 2000` lets Stage 1 run longer at those meshes, changing its endpoint and hence
Stage 2's initial design. Raising a previously binding cap is not a neutral safety horizon —
it changes the realized algorithm at a third of the mesh set, and it inflates `N_stage1` and
`N_total`. Must be disclosed.

## 3. Olhoff — verified correct throughout

Source: `analysis/olhoff_stabilization_audit/olhoffOptStabilized.m`, LP subproblem
`Matlab/reproduction2007/algo/innerLoopLP.m`, profile
`olhoff_fig3a_s1_native_bimodal_p100_move005_0025_fixed1600_v1`.

### Outer iteration

One pass of `for outer = 1:cfg.maxOuter`:

1. `assemble2D` then `eigSolve(K, M, Jcalc, cfg.solver)` — one generalized eigensolve;
   multiplicity detection `while n+N<=Jcalc-1 && abs(w(n+N)-w(n))/w(n) < cfg.tolMult`;
2. `genGrad` for the clustered modes and for mode `J`, plus `applyFilter` on each — the
   generalized-gradient construction;
3. one `innerLoopLP` call;
4. `rho = min(1, max(cfg.rhomin, rho + drho))` — one accepted update.

`N_outer` counts successful updates; `hist` retains `lpFlag`, `finiteOk`, `nInner`,
`cumInner`, `policyStage`, `trigger`, `gap12`, `N`, `dxOuter`, `dRms`, `vol`,
`volumeResidual`, `tEig`, `tGrad`, `tInner`. All present in the frozen artifacts.

**Outer iterations are the correct method-level unit.** It is the level at which the design
is updated and at which the method's own literature counts.

### `nInner` — the protocol's claim is exactly right

`innerLoopLP.m` makes exactly one call:

```matlab
opts = optimoptions('linprog','Display','none','Algorithm','dual-simplex-highs');
[x, ~, flag] = linprog(f, A, b, Aeq, beq, lb, ub, opts);
st = struct('nInner',1,'degenHits',0,'conv',flag==1,'dxHist',[],'relHist',[],'lpFlag',flag);
```

`nInner` is hard-coded to 1, independent of solver effort. Confirmed in frozen data at
160x20: `unique(hist.nInner) = [1]`, `sum(hist.nInner) = 1600` for `nOuter = 1600`.
**`sum(nInner) == N_outer` exactly.**

Note the third output of `linprog` is `exitflag`; the `output` struct carrying
`.iterations` is **not** captured on the production path. So solver-internal iterations
genuinely require new instrumentation, as the evidence matrix says.

**The distinction is proven by frozen evidence.**
`analysis/performance_campaign_targeted_replays/olhoff_640_failure_diagnostics.csv`:
`linprog_exitflag = 0`, `lp_iterations = 38`, `solver_message = "Solver stopped
prematurely"`, `reproduction_verdict = FAILURE_REPRODUCED`, `causal_classification =
GENERIC_LP_ITERATION_LIMIT_ONLY`. One LP call, 38 solver iterations, `nInner` still 1.
Three distinct levels, demonstrated on real data.

**Verdict: `nInner = 1` is one `linprog` call and must never be reported as a HiGHS/simplex
iteration. The protocol states this correctly in four places. No correction needed.**

### Does LP-call count add information?

Essentially none while one call occurs per successful update. The protocol's recommendation
— footnote rather than a main column, promoted only if a retry policy ever makes them
differ — is correct. Main Table 1 currently carries an `Olhoff LP calls enter/cert` column;
given the exact redundancy, that column is the natural thing to sacrifice to make room for
the achieved-quality columns (Finding M3).

### Are solver-internal LP iterations meaningful work metrics?

**No.** They are diagnostics of MATLAB's `dual-simplex-highs` on a degenerate LP with 51 201
variables, 4 inequality rows and 1 equality row (frozen 640x80 diagnostics), not a property
of the Du–Olhoff formulation. Supplementary placement is correct. The spec's insistence on
capturing them only from genuine solver output, writing `NA` otherwise, and never inferring
them from `st.nInner`, is correct and should be kept verbatim.

### Method-specific gate imposes a floor on `k_enter` (Finding M8)

The Olhoff acceptance condition requires `policyStage == 2`, which requires the native
bimodal trigger (`N == 2 && gap12 <= 0.01`) to have held for 100 consecutive iterations. In
the frozen 160x20 run, `res.trigger_iterations = 245`. So **Olhoff's `k_enter` at 160x20
cannot be below ≈245 regardless of design quality** — the gate, not maturation, sets the
floor.

The protocol's defence (F11, F13) — that this is part of the intended reproduced Olhoff
result and that imposing artificial symmetry would be less fair — is correct, and it is the
best-argued point in the package. But the magnitude must be visible: report each method's
own gate-satisfaction iteration beside `k_enter`, so a reader can see how much of the
headline number is gate-imposed.

## 4. Cross-method comparison — can they share one plot?

**Yes, with an on-figure qualification.**

What the plot measures: **method-level design-update count to the accepted endpoint.**
Well-defined and honest for all three.

What it does not measure: computational work. Quantified from the frozen campaign at
800x100:

| method | s per method-level update | contains an eigensolve? | contains an LP solve? |
|---|---:|:--:|:--:|
| Yuksel | 0.489 | no | no |
| Proposed | 0.814 | **no** (frozen solid reference, never refreshed) | no |
| Olhoff | 1.844 | **yes** (75 % of its cost) | **yes** |

A **3.8× spread**, with qualitatively different content per update.

**Required qualification:** F1's panels must carry an in-axes note of the form
"method-level update counts; per-update cost differs by up to 3.8× at `N_e` = 80 000 — see
F2", and F2 must be adjacent, not merely cross-referenced. The protocol places F2 in the
mandatory main set, which is right; it does not require the in-axes note or the adjacency,
which it should.

**Demanding false algorithmic equivalence would be worse than the current design.** The
protocol is correct to refuse it, correct to list the seven non-equivalent quantities
explicitly, and correct to say they "may appear in adjacent columns but must never be
described as interchangeable units". That section needs no change.

## 5. Summary of corrections to this spec

| ref | correction | new science? |
|---|---|---|
| M7 | Fix the Yuksel Stage-1 handoff sentence; restate the exclusion on objective-mismatch grounds | no |
| M6 | Disclose the Stage-1 cap truncation at three meshes and its effect on the Stage-2 start | no |
| M8 | Report each method's gate-satisfaction iteration beside `k_enter` | no |
| Mi5 | Record that a Proposed iteration contains no eigensolve; put the 75 % Olhoff figure in the footnote | no |
| §4 | Add the in-axes per-update-cost qualifier to F1 and require F2 adjacency | no |
| M3 | Sacrifice the redundant `Olhoff LP calls` column to make room for achieved quality | no |
