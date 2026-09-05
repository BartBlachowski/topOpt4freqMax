# `analysis/OlhoffM4Reconstruction` — the conference Du–Olhoff reconstruction (M4)

This is the **current, conference-active** realization of the Du & Olhoff (2007)
eigenfrequency-maximization method in this repository. It was imported from the
scientifically audited development repository

    /Users/piotrek/Programming/Matlab/Olhoff

It is **not** one of the older `analysis/Olhoff*` implementations, and it is not
`Matlab/reproduction2007`. Those are superseded or historical; see
[`../OLHOFF_IMPLEMENTATION_STATUS.md`](../OLHOFF_IMPLEMENTATION_STATUS.md).

## The label, and the caveat that goes with it

Report this method as **“Du–Olhoff reconstruction (M4)”**. Never as
“Olhoff 2007”.

> Du–Olhoff timings and iteration counts refer to the frozen reconstruction used
> in this study. Some continuation and inner-solver details are not uniquely
> specified by the original publication; therefore these values should be
> interpreted as representative measurements of this reconstruction rather than
> exact historical implementation timings.

M4 is a **reconstruction** (class C in the source audits): an internally
coherent multiplicity treatment assembled only from the paper's own equations
(19), (24), (25c) and (25d). It is not a claim about what Du & Olhoff coded.

## Layout

```
olhoffm4_root.m                 where this import lives
olhoffm4_paths.m                fail-closed path guard  -> hold the returned onCleanup
olhoffm4_owned_names.m          the functions this import owns
olhoffm4_assert_dispatch.m      proves which .m file each of them resolves to
olhoffm4_forbidden_paths.m      the superseded implementations that must never run
olhoffm4_config.m               THE FROZEN CONFERENCE CONFIGURATION
olhoffm4_caveat.m               the caveat text, in one place
olhoffm4_run.m                  benchmark wrapper: runs, and accounts for the cost
olhoffm4_verify_import.m        integrity + attestation + reconstruction proofs
olhoffm4_apply_unified_diff.m   applies the declared diff in memory, strictly
olhoffm4_equivalence_160x20.m   source-vs-import bitwise equivalence at 160x20
olhoffm4_sha256_file.m          raw-file SHA-256, matching `shasum -a 256`
olhoffm4_sha256_bytes.m         the same digest over bytes held in memory
olhoffm4_read_bytes.m           raw uint8 file read, no encoding translation

+frozen/                        THE AUDITED SOLVER CORE -- do not edit
  algo/   olhoffOpt defaultCfg multRule moveControl genGrad deltaLambda
          innerLoop innerLoopLP innerLoopRho useMMA
  fem/    model2D elemMats2D assemble2D massScale eigSolve classifyModes
  filter/ prepFilter applyFilter
  mma_published/  mmasub subsolv   (Svanberg's published Sept-2007 constants)
  mma/            mmasub subsolv   (the 'asfound' variant; NOT used)

patches/    the one declared modification, as a diff plus the verbatim source
evidence/   equivalence results
IMPORT_MANIFEST.json / .md      provenance, hashes, source->destination mapping
SOURCE_SHA256.txt               hash of every file in the source solver tree
```

### Why `+frozen/`

`genpath` skips `+`-prefixed folders and everything beneath them, and several
repository scripts call `addpath(genpath(<repo>/analysis))`. Putting the core in
a plain subfolder would shadow `tools/Matlab/mmasub.m` and break the isolation
guarantee that `Matlab/reproduction2007/runner/repro2007_verify_isolation.m`
checks. Under `+frozen/` the core is reachable **only** through
`olhoffm4_paths()`, which asserts the implementation's identity first.

## Usage

```matlab
addpath('<repo>/analysis/OlhoffM4Reconstruction');
maxNumCompThreads(1);

out = olhoffm4_run(160, 20);      % installs the guard, runs, returns accounting
out.accounting.outer_iterations
out.accounting.inner_iterations_total
out.accounting.outer_time_excluding_inner_s
out.accounting.inner_time_total_s
out.accounting.inner_time_share_pct
```

Never report `outer_iterations + inner_iterations_total` as a single iteration
count: they are different objects.

## The frozen realization

Read `olhoffm4_config.m` — every field is there in full, with its provenance in
the comments. In summary:

| group | setting |
|---|---|
| problem | `a=8, b=1, t=1`, `E=1e7`, `ν=0.3`, `ρ_m=1`; simply supported at mid height, both axial DOFs; maximize `ω₁`; 50 % volume; `ρ⁰=0.5`, `ρ_min=1e−3` |
| FE | Q4 plane stress, consistent mass, `eigs` with a fixed deterministic start vector, `J = n + N_max = 5` modes |
| stiffness | SIMP, `p = 3` constant, no continuation |
| mass | eq. (4b), C¹, `c₁=6e5`, `c₂=−5e6`, `q=1`, `r=6` |
| filter | Sigmund (1997) sensitivity filter, top88 weights, `filterMode='all'`, **fixed physical radius `R = 0.06·b`**, `rminEl = R/(b/nely)` derived at run time |
| multiplicity | **M4**: `multRule='subspace'`, `subN = 2`, no classifier |
| step control | S2 staged move, `move₀ = 0.04`, ladder `[0.04 0.02 0.01 0.005]`, window 10, tol 5e−3, legacy `beta` stall signal |
| inner loop | **genuine nested MMA**, `innerVar='drho'`, published constants, full coupling, `maxInner = 500`, **`tolInner = 0.05`**, `minInner = 5` |
| outer stop | `‖Δρ‖₂ < 0.05·√(N_e/3200)` — a mesh-independent per-element RMS tolerance of `8.838835e−04` — with `outerGuard='settledmove'` |
| safety cap | `maxOuter = 400`. **A run that reaches it is not converged.** |
| numerics | `threads = 1` |

Derived filter radii: 160×20 → 1.2, 240×30 → 1.8, 320×40 → 2.4, 400×50 → 3.0,
480×60 → 3.6, 560×70 → 4.2, 640×80 → 4.8, 720×90 → 5.4, 800×100 → 6.0, all at
`r_phys = 0.06`.

### The one benchmark-vs-audit difference

The audited runs set `diag = true` (the per-iteration diagnostic recorder). The
benchmark sets `diag = false`, because that recorder costs measurable time per
outer iteration and is not part of the method. It is **purely additive** and
proved bitwise inert — `olhoffm4_equivalence_160x20.m` re-proves it at 160×20,
comparing raw IEEE-754 bytes of the final density field.

### One thing the source audits say, that the benchmark does not hide

The most recent source audit (`audit_s2_design_continuation/REPORT.md`, verdict
`S2_LADDER_ITSELF_DEFECTIVE`) established that the **outer iteration count of
this method is an artifact of the move-limit continuation trigger**: the same
solver, filter, M4 and outer tolerance give 91/104/131 outer iterations under
one trigger and 86/54/59 under another, with no change to the physics. The
design-driven `drms` trigger was measured and **not adopted**; `s2Signal` is
absent from the frozen configuration, so the legacy `beta` signal is what runs.

That is exactly what the caveat above is for. The measured numbers are
published; they are labelled as measurements of *this reconstruction*.
