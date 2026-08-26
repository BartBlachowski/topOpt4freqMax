# Provenance — Du–Olhoff 2007 clean-room benchmark reproduction (Eq. 22 LP route)

This directory holds an **independent clean-room reproduction** of the algorithm of

> Jianbin Du and Niels Olhoff, *Topological design of freely vibrating continuum
> structures for maximum values of simple and multiple eigenfrequencies and
> frequency gaps*, **Struct Multidisc Optim 34:91–110 (2007)**,
> DOI `10.1007/s00158-007-0101-y`

implemented against its **Publisher's Erratum**

> **Struct Multidisc Optim 34:545 (2007)**, DOI `10.1007/s00158-007-0167-6`,
> published online 6 September 2007.

The erratum is mandatory. As printed in the main article, equations **(25d)**,
**(26f)** and **(26g)** omit the `Δ` symbol; the erratum restores it:

```
printed:   det | f_sk^T Δρ  −  δ_sk (ω²)  | = 0      WRONG
erratum:   det | f_sk^T Δρ  −  δ_sk Δ(ω²) | = 0      IMPLEMENTED HERE
```

The erratum also corrects the Fig. 2 caption symbols (`ω^φ` → `ω⁰`), which is
what fixes the reading of the published initial frequencies.

A second, independent confirmation of the corrected form is
Olhoff & Du (2014), *Structural Topology Optimization with Respect to
Eigenfrequencies of Vibration*, whose eq. (19d)/(20f,g) print the corrected
equations directly. That paper closes no other gap — it contains zero
occurrences of the word "filter".

Local copies of all three documents already exist in this repository and were
verified byte-identical to the copies held in the source repository:

| document | repository path | SHA256 (first 16) |
|---|---|---|
| Du & Olhoff (2007) main article | `references/Du2007_Topological.pdf` | `b4dc8153ddc505b5` |
| Olhoff & Du (2014) | `references/Olhoff2014_Structural.pdf` | `1567fc37f64b11be` |
| Publisher's Erratum, SMO 34:545 | *source repo only*, `docs/Du and Olhoff - Topological design of freely vibrating continuum s.pdf` | `e3cb33390ccbabe1` |

> The erratum PDF is **not** imported here: this repository's `.gitignore`
> excludes `*.pdf`, and the erratum's content that matters to the
> implementation is transcribed verbatim above. It remains available at the
> source path recorded below.

---

## Source repository

| field | value |
|---|---|
| **Source path** | `/Users/piotrek/Programming/Matlab/Olhoff` |
| **Source Git branch** | *(none — the source directory is **not** a Git repository)* |
| **Source Git HEAD** | *(none — see below)* |
| **Source dirty/clean status** | *not applicable; no version control present* |
| **Migration date** | 2026-08-26 |
| **Migrated into** | `Matlab/reproduction2007/` of `topOpt4freqMax` |
| **Target branch at import** | `benchmark-methodology-r2` |
| **Target HEAD at import** | `cf290fc7f9daf9da27bc8224f9585a0e1657bff1` |
| **Target working tree at import** | clean (no modified, no untracked files) |

`git rev-parse --is-inside-work-tree` in the source directory returns
`fatal: not a git repository`. There is therefore **no upstream branch or
commit to cite**, and the SHA256 manifest in `SOURCE_SHA256.txt` is the *only*
provenance anchor for the imported bytes. That manifest was computed before the
copy and re-verified after it; all 61 files matched.

File timestamps in the source directory place the reproduction work on
**2026-08-25** (implementation `10:47`–`11:57`, result artifacts through
`15:52`).

---

## What this implementation is

**Du–Olhoff 2007 clean-room benchmark reproduction (Eq. 22 LP route).**

It is the implementation that produced the successful Fig. 3a / Fig. 4
benchmark family for the simply supported beam (paper Fig. 2a, target mode
`n = 1`):

| quantity | paper | this implementation | configuration |
|---|---|---|---|
| ω₁⁰ (initial, case a) | 68.7 | 68.3209 | 240×30 |
| ω₁ at the optimum | 174.7 | **170.4709** (−2.4 %) | `fig3a_best` |
| ω₂ at the optimum | 174.7 (bimodal) | **170.8659** (gap 0.23 %) | `fig3a_best` |
| ω₃ at the optimum | 284.9 | **285.1939** (+0.1 %) | `fig3a_best` |
| multiplicity at the optimum | bimodal | bimodal, `N = 2` | `fig3a_best` |
| Fig. 4 ω₁/ω₂ coalescence | ≈ iteration 20 | iteration 26 | `fig4_history` |

Frozen artifacts of both runs are stored under `baseline/` and are the
comparison target of the migration regression (`runner/repro2007_regression.m`).

### What it is NOT

This is **not** "the correct implementation", and it is not a recovered copy of
the authors' own code. The 2007 paper leaves the following quantities
**unstated**, and every one of them had to be reconstructed:

- the filter radius `r_min` (this is the parameter that decides whether the
  optimum is bimodal at all — see `NOTES.md` §6);
- the move limit (it sets the pace of Fig. 4 — `NOTES.md` §8c);
- the multiplicity tolerance;
- the mesh `NE`;
- the inner-loop convergence criterion (Fig. 1 asks only "Increments Δρ_e
  converged?" and never gives a test — `NOTES.md` §4);
- the support idealization, which the paper *draws* one way (bottom corners)
  and *numbers* another (mid-height, axially restrained both ends —
  `NOTES.md` §2).

The reproduction therefore establishes that **the published benchmark is
reproducible**. It does **not** establish that these undocumented numerical
choices are the ones Du and Olhoff actually made. Both statements are needed;
neither implies the other.

---

## Relationship to the other two implementations in this repository

Code duplication between this directory and `analysis/OlhoffApproachExact/` is
**intentional and load-bearing**. The three implementations are kept as
independent executable realizations so that a disagreement between them is
evidence rather than a merge artifact. They must not be refactored onto shared
numerical code while the paper revision is open.

See `Matlab/README.md` for the three-way map.

`OLHOFFEXACT_FAILURE_POSTMORTEM.md` (imported here unchanged) is the forensic
comparison of all three against the paper. Its §2.1 is the origin of the
`legacy` / `rebuilt` / `clean-room` vocabulary used throughout.

---

## Import integrity

- **61 files** imported; all SHA256-verified against `SOURCE_SHA256.txt`
  after the copy.
- **Zero source files were modified.** No algorithm, filter semantic,
  multiplicity semantic, Eq. (22) formulation, LP formulation, mass
  interpolation, move handling, eigensolver setting, or numerical constant was
  changed, simplified, or "cleaned up".
- Two layout changes were applied, **content-identical**:
  - `CLAUDE.md` → `SOURCE_CLAUDE.md`, so that this repository's Claude Code
    tooling does not read the source repository's instruction file as its own;
  - `results/{FIG4_definitive,FINAL_lp_240x30,lp240_rmin1.3}.*` →
    `baseline/`, to separate frozen regression targets from the run-time
    output directory `results/`.
- Everything under `runner/` is **new integration code written for this
  repository**. It calls the imported implementation and never reimplements
  any part of it. It is excluded from the SHA256 manifest by construction.

## Deliberately not imported

- The remaining ~80 exploratory artifacts in the source `results/` tree
  (≈ 84 MB of sweep `.mat`/`.log`/`.png` files). They are evidence for
  `NOTES.md`, not inputs to the implementation, and remain at the source path.
- The three PDFs under the source `docs/` — two already exist under
  `references/`, and `.gitignore` excludes `*.pdf`.
