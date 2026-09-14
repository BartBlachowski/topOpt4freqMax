# CODE TRACE — what beta is, from the source

Traced, not inferred from the variable name. Every claim below cites a file and
line at `+impl` tree `c1455374…` (74/74 verified).

## The call path

```
olhoffSolve.m
  267   [mvNow, mvState] = olh.move.limit(cfg, outer, hist, mvState)   <-- DECIDES the move
  298   ctx = struct(..., 'move', mvNow, ...)                          <-- move enters the box
  313   [drho, st] = innerLoop(ctx)                                    <-- subproblem solved
  326   rho = min(1, max(rhomin, rho + drho))
  341   dxOuter = max(abs(drho))
  368   hist.beta(outer) = st.beta                                     <-- beta RECORDED
  381   hist.move(outer) = mvNow
```

`olh.move.limit` is called with `hist` holding iterations `1..outer-1`, so the
stall decision at iteration `outer` uses **completed iterations only**.

## Where beta is defined: `+impl/algo/innerLoop.m`

`innerLoop` solves Du–Olhoff sub-problem (25). Its header states the formulation
outright (line 5): *"Independent variables : beta and drho_e (paper sec. 3.5.2)"*.

```
 46   nvar = NE + 1;                    % x = [drho ; beta_scaled]
 47   lo = max(ctx.rhomin - ctx.rho, -ctx.move);
 48   hi = min(1          - ctx.rho,  ctx.move);
 49   xmin = [lo; 0];   xmax = [hi; 5];
 52   x = [zeros(NE,1); 1];             % drho = 0, beta = lambda_n
 ...
 93   % (25c)  beta - [omega_j^2 + Delta(omega_j^2)] <= 0
 95   fval(j)        = bs - (ctx.lam(j) + dlam(j))/lamref;
 96   dfdx(j,1:NE)   = -ddlam(:,j).'/lamref;
 97   dfdx(j,nvar)   = 1;
 99   % (25b)  beta - [omega_J^2 + f_JJ' drho] <= 0
101   fval(N+1)      = bs - (ctx.lamJ + ctx.fJJ.'*drho)/lamref;
 ...
133   f0val = -bs;                      % MMA MINIMISES, so this MAXIMISES beta
134   df0dx = zeros(nvar,1); df0dx(nvar) = -1;
137   [xmma,...] = mmasub(m, nvar, it, x, xmin, xmax, ...)
153   st.beta = x(end)*lamref;          % returned in ABSOLUTE eigenvalue units
```

with `lamref = ctx.lam(1)`, the current `lambda_n`.

### What that makes beta

| question | answer |
|---|---|
| what is it | the **bound variable of the Du–Olhoff bound formulation (25a)** |
| primal or dual | **primal** — literally the `(NE+1)`-th entry of MMA's design vector `x` |
| slack / artificial | **no**. MMA's own artificial variables are `y` and `z`; its duals are `lam, xsi, eta, mu, zet, s`. beta is none of them |
| scalar or vector | **scalar** |
| units | **eigenvalue**, `lambda = omega^2` (stored internally scaled by `lambda_ref`, unscaled on return) |
| value at the subproblem optimum | the largest lower bound on all tracked `lambda_j` achievable by one linearised step inside the move box |
| depends on NE | **no explicit dependence.** NE sets the number of *other* variables, but beta is an eigenvalue-valued scalar |
| depends on move | **yes**, through the box `lo/hi = ±move`, which bounds `drho` and hence `Delta lambda` |
| depends on MMA asymptotes | **yes, indirectly** — see below |
| depends on objective/constraint scaling | yes, through `lamref` scaling, which is divided out again on return |
| depends on multiplicity treatment | yes — the number of `(25c)` rows is `N`, the multiplicity |

## A genuine name collision, worth flagging

**There are two unrelated quantities called `beta` in this codebase.**

| | `hist.beta` / `st.beta` | MMA's internal `beta` |
|---|---|---|
| defined at | `innerLoop.m:153` | `mmasub.m:108`, `beta = min(zzz,xmax)` |
| what it is | Du–Olhoff bound variable (25a) | **upper trust bound vector** on the design variables |
| shape | scalar | `nvar × 1` vector |
| units | eigenvalue | design-variable increment |

The stall detector watches the **first**. The brief's instruction not to infer
meaning from the name is well founded: reading `mmasub` alone would give exactly
the wrong answer.

## How beta becomes a move decision: `+impl/architecture/+olh/+move/limit.m`

```
case 'ladder'
    W   = cfg.move.continuation.window;      % 10
    tol = cfg.move.continuation.tolerance;   % 5e-3
    case 'boundVariable'
        b = hist.beta;  wantDrop = false;    % beta INCREASES when useful
    if numel(b) >= 2*W && (outer - state.lastStage) > W
        w2  = mean(b(end-W+1:end));          % last 10 completed iterations
        w1  = mean(b(end-2*W+1:end-W));      % the 10 before those
        rel = (w2-w1)/max(abs(w1),eps);      % RELATIVE INCREASE
        if rel < tol
            state.stage    = min(state.stage+1, numel(cfg.move.levels));
            state.lastStage = outer;
        end
    end
    mv = cfg.move.levels(state.stage);
```

So the exact descent predicate is:

> **descend one rung when the mean of beta over the last 10 completed iterations
> exceeds the mean over the preceding 10 by less than 0.5% — provided at least
> 11 iterations have passed since the previous rung change.**

`(outer - lastStage) > W` is a **dwell guard**: after a change at iteration `L`
the earliest possible next descent is `L + W + 1 = L + 11`.

## The move → beta coupling, in code

```
innerLoop.m:47-49   lo/hi = ±move             -> xmax-xmin = 2*move (interior elements)
mmasub.m:80-81      low = xval - 0.5*(xmax-xmin)   -> asymptote spread ∝ move   (iter 1,2)
mmasub.m:102,106    zzz2 = xval ± 0.5*(xmax-xmin)  -> MMA trust bounds ∝ move
                    (mmasub's own `move` constant is 0.5, distinct from the Olhoff move)
```

The Olhoff move therefore sets the box, the initial MMA asymptote spread and the
MMA trust bounds — all proportional to it. This coupling is **real and measured**
(see `MATHEMATICAL_ANALYSIS.md`: `g ≈ 2·move` early, at all three meshes).

## Replay check

`scripts/btm_common.py` re-implements the predicate above and replays it against
recorded beta histories. It reproduces **every actual production descent at all
three meshes**: `[79, 90, 101]` at 160x20, `[130, 141, 152]` at 320x40, `[138]`
at 400x50. The audit is therefore analysing the rule production actually applied,
not a paraphrase of it.
