# WP1 — Exact executable route of BASE_mma_160x20
NESTED-MMA ROUTE AUDIT — READ-ONLY — traced from source, not from printed labels

## Source identity

The run was produced by the clean-room tree at
`/Volumes/HP911Pro/Combobulating/Olhoff/`. That tree is **byte-identical** to the
repository's `Matlab/reproduction2007/`: 46 of 46 `.m` files under `algo/`, `fem/`,
`filter/`, `mma/`, `runs/` hash-match, as do `setpaths.m` and `top88.m`. The repository copy
is therefore the exact code audited here.

## Call graph

    run_case('BASE_mma_160x20', overrides)                       runs/run_case.m
      |- cfg = defaultCfg()  <- overrides                        algo/defaultCfg.m
      |- disp(cfg)                        <-- prints the PRE-OVERRIDE echo (rminEl=3)
      |- olhoffOpt(cfg)                                          algo/olhoffOpt.m
      |    |- model2D(cfg)                                       fem/model2D.m
      |    |- cfg.rminEl <- rminPhys/(b/nely) = 0.06/0.05 = 1.2  <-- ACTUAL filter radius
      |    |- prepFilter(160, 20, 1.2)                           filter/prepFilter.m
      |    |- for outer = 1 : maxOuter(=800)
      |    |    |- assemble2D(mdl, rho, p=3, massInterp='4')     fem/assemble2D.m
      |    |    |     \- massScale(rho,'4')  =  Eq. (4): rho^6 below 0.1, rho above
      |    |    |- eigSolve(K, M, Jcalc=n+Nmax=5, 'eigs')        fem/eigSolve.m
      |    |    |- multiplicity: while n+N<=Jcalc-1 && |w(n+N)-w(n)|/w(n) < tolMult(0.05)
      |    |    |     -> FREQUENCY-relative, NO hysteresis
      |    |    |- genGrad(...) -> F (NE x N x N);  FJ -> fJJ    algo/genGrad.m
      |    |    |- filterMode='diag':  applyFilter on F(:,j,j) and fJJ ONLY
      |    |    |     -> off-diagonal f_sk are NOT filtered
      |    |    |- innerSolver='mma' -> innerLoop(ctx)           algo/innerLoop.m
      |    |    |    \- for it = 1 : maxInner(=300)
      |    |    |         |- offDiag=1 -> deltaLambda(F, drho)   algo/deltaLambda.m
      |    |    |         |     \- solves det|f_sk'drho - delta_sk*DELTA| = 0  (Eq. 25d, erratum form)
      |    |    |         |- m = N+2 constraints: (25c)xN, (25b), (25e)
      |    |    |         |- mmasub(...)                         mma/mmasub.m -> subsolv.m
      |    |    |         \- converged if it>=minInner(5) and dx/max|xmma| < tolInner(0.01)
      |    |    |- rho <- min(1, max(rhomin, rho + drho))
      |    |    \- break if max|drho| < tolOuter(1e-3)
      |    \- final assemble + eigSolve + classifyModes
      \- save results/BASE_mma_160x20.mat        <-- NEVER REACHED (see WP11)

## Verified answers to the WP1 questions

| item | finding |
|---|---|
| outer solver | `algo/olhoffOpt.m` |
| inner MMA solver | `algo/innerLoop.m`, calling `mma/mmasub.m` |
| generalized gradients | `algo/genGrad.m`, full `NE x N x N` tensor |
| off-diagonal treatment | `offDiag=1` → `deltaLambda` solves the N x N subeigenvalue problem and returns `ddlam(e,j) = sum_{s,k} v_js v_jk (f_sk)_e`. **This is genuine full coupling**, the erratum form of Eq. (25d). It is not a diagonal approximation with a different label. |
| filter path | `prepFilter` (top88 weights, element units) + `applyFilter` (top88 sensitivity filter with the `max(1e-3,x)` normalisation). `filterMode='diag'` filters only the diagonal `f_jj` and `f_JJ`. |
| multiplicity detector | frequency-relative, `tolMult = 0.05`, **no hysteresis** |
| mass interpolation | `massInterp='4'` → Du–Olhoff Eq. (4), the discontinuous `rho^6` branch |
| move-limit handling | `lo = max(rhomin-rho, -move)`, `hi = min(1-rho, +move)`, `move = 0.01`, applied as MMA variable bounds |
| density bounds | `rho <- min(1, max(rhomin=1e-3, rho+drho))` after the inner solve |
| inner stopping test | `dx / max(|xmma(1:NE)|) < tolInner` with `dx = max|xmma - x|` over the design block, plus `it >= minInner=5`. Relative to the **accumulated** increment — explicitly a reconstruction, documented as such in the source header. |
| outer stopping test | `max|drho| < tolOuter = 1e-3` |

## Is any LP fallback invoked?

**No.** `olhoffOpt.m` selects the inner solver by

    if strcmpi(cfg.innerSolver,'lp')
        [drho, st] = innerLoopLP(ctx);
    else
        [drho, st] = innerLoop(ctx);
    end

There is no `N`-dependent branch, no failure fallback, and no LP call anywhere in the MMA
path. `innerLoopLP` is unreachable for this configuration at every outer iteration and at
both `N = 1` and `N = 2`. The 752 recorded iterations are pure nested MMA.

## Two reconstruction choices that are not paper-literal

1. **Mixed filtering.** With `filterMode='diag'` the diagonal `f_jj` are filtered but the
   off-diagonal `f_sk` fed to `deltaLambda` are **not**. The subeigenvalue problem therefore
   mixes filtered and unfiltered tensor entries. The paper does not specify what is filtered;
   this is flagged as `material/unknown` in the existing post-mortem and remains so.
2. **Inner stopping criterion.** The paper states only "Increments drho_e converged?" and
   gives no criterion. The implemented relative test is declared in the source as
   "RECONSTRUCTION, not the authors'."

Neither is a defect. Both are undocumented degrees of freedom that must be reported as such
in any paper-fidelity claim.
