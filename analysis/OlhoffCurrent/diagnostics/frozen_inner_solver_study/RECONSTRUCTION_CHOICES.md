# Reconstruction inventory (before experiments)

Evidence: `+impl/algo/innerLoop.m`, `innerLoopRho.m`, `deltaLambda.m`,
`defaultCfg.m`, `olhoffOpt.m`, `useMMA.m`, active `+impl/mma_published/`,
reference `state_identity.json`, and Svanberg's official distribution downloaded
from https://www.smoptit.se/GCMMA-MMA-code-1.5.zip. The authoritative runtime
configuration overrides defaultCfg: inner tolerance .05, min 5, max 500.
Paper: Du & Olhoff (2007), local `references/Du2007_Topological.pdf`, pp.97-99;
its determinant uses the erratum described in the reference audit.

| Choice | Actual frozen production | Paper specifies / silent | Origin | Repository alternative |
|---|---|---|---|---|
| Independent variables | drho, bs=beta/lam1 | drho and beta specified; numeric scaling silent | scaling reconstruction | innerLoopRho density coordinates |
| Spectral / volume scaling | divide by lam1 / Vtot | silent | reconstruction | none relevant |
| bs box | [0,5] | silent | reconstruction | LP paths |
| History between inner calls | xold2=xold1; xold1=x, persists | silent | standard MMA | already persistent |
| Asymptotes between inner calls | returned low/upp passed on | silent | standard MMA | already persistent |
| Counter between inner calls | it=1,2,... | silent | standard MMA | global counter in innerLoopRho |
| State across outer problems | reset x=[0;1], old=x, low/upp=box | silent | reconstruction | innerLoopRho preserves across outer |
| Newton primal/dual warm starts | subsolv initializes afresh every approximate solve | silent | inherited subsolv behavior | none |
| Asymptote initialization | .5 width at first two calls | silent | canonical MMA | inactive .01 |
| Asymptote factors | 1.2 increase, .7 decrease, signed consecutive steps | silent | canonical MMA | no relevant difference |
| Asymptote distance clamp | [.01,.2] widths from third call | silent | inherited local modification | official [.01,10] |
| Physical move box | max(rhomin-rho,-.01), min(1-rho,.01) | (25f) density bounds; move silent | reconstruction | not varied |
| MMA local bound geometry | albefa=.1; local move=.5 width | silent | canonical MMA | inactive move=1 |
| Regularization | .001*abs(gradient) + 1e-5/width | silent | MMA defaults | GCMMA adaptive raa |
| Approximate solve accuracy | epsimin=1e-7, barrier resets to 1 each call | silent | MMA default, absolute units | commented historical formula only |
| tolInner | .05 | unspecified increment convergence diamond | reconstruction | defaultCfg .01 |
| Relative step | max(abs(newdrho-drho))/max(max(abs(newdrho)),1e-12) | silent | reconstruction | same in innerLoopRho |
| minInner / maxInner | 5 / 500 | silent | reconstruction | defaultCfg max=300 |
| Nonlinear relinearization | recompute same fixed deltaLambda problem every inner call | nonlinear dependent sub-eigenproblem specified | coherent sequential approximation | LP when off-diagonal equality constraints enforced |
| MMA / GCMMA | MMA | MMA cited; no GCMMA prescription | chosen solver | GCMMA absent before audit download |
| Damping / line search | no fixed-NLP wrapper line search; subsolv has residual backtracking (up to 50 halvings) | silent | inherited Newton safeguard inside each approximation | no NLP line search |
| Conservative check | none | silent | plain MMA | official GCMMA downloaded audit-only |
| Offset / full coupling | dOff=lam-lam1, offDiag=true | exact multiplicity determinant; offsets silent | reconstruction | innerLoopRho omits dOff |
| Volume | affine, equal element volumes | (25e) | problem definition | optional nonlinear volFun absent here |

The local mma_published README says everything except move and asyinit matches
published code. Direct source inspection contradicts that statement: the two
maximum-distance constants remain .2, whereas official code has 10. The README
and production files are left untouched. This discrepancy motivates S3.

## Semantic diff: innerLoopRho versus innerLoop

1. Density coordinates instead of increments, with subtractive conversion of
   trial variables back to drho and different floating-point arithmetic.
2. State struct retains itG, low/upp, xold1/xold2 across outer calls; original
   retains the same quantities across all calls in ONE frozen problem.
3. deltaLambda(F,drho) omits dOff. On this frozen state dOff(2)=7428.636632,
   so it changes the spectral constraints, not just a solver choice.
4. Optional volFun handling is absent. Irrelevant here but a semantic difference.
5. Allows move=inf; forbidden here. The actual finite move must remain fixed.

Therefore it is not run. S1 merely serializes/deserializes the already persistent
state at the production stopping boundary, preserving all expressions. A
cross-outer persistence effect is not identifiable from one zero-update state.

No solver-history discard occurs between nonlinear relinearizations. The
subsolv Newton workspace is discarded, as in the official method; that is
separate from MMA approximation history. No claim of a primary reset failure
can be made from the present source code.

Additional solver-level limits: subsolv allows at most 200 Newton steps at
each barrier level and up to 50 residual-backtracking trials. These are
inherited defaults, not altered here. The production copy omits the upstream
print when the 200-step cap is approached, so absence of that print does not
prove the requested accuracy was achieved. Exact NLP residuals remain decisive.
The degen flag is set before offsets are added in deltaLambda; its initial
hit is only a diagnostic and does not select the eigenvectors/gradients used.

The initial inventory preceded solver experiments. The internal Newton
backtracking/cap clarification above was added during analysis; it introduces
no new variant and is not an independent causal intervention.
