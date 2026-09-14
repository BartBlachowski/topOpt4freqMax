# OLHOFF_IMPLEMENTATION_MAP

Every Du–Olhoff-related tree discovered on this machine, classified.

**There is exactly one `PRODUCTION` entry.**

This map is the central record required by §13 of the promotion brief. It exists
so that historical directories do not have to be modified to be labelled:
several of them are covered by frozen hash manifests, and editing them would
invalidate the evidence they carry.

| Classification | Meaning |
|---|---|
| `PRODUCTION` | production scripts may execute it — **exactly one tree** |
| `DEVELOPMENT_UPSTREAM` | where development happens; never a production dependency |
| `FROZEN_EVIDENCE` | immutable scientific record, hash-pinned; never modify |
| `EXPERIMENTAL` | experimental code; never production |
| `HISTORICAL` | a superseded realization kept for provenance |
| `AUDIT_ONLY` | audit runners/evidence around another implementation |
| `UNKNOWN` | unclassified — permitted only with justification, reported as debt |

---

## The map

| Tree | Class | Production executable? | Mutable? | Notes |
|---|---|---|---|---|
| **`analysis/OlhoffCurrent`** | **PRODUCTION** | **YES — the only one** | via promotion only, never edited in place | Promoted from upstream `695f03b`. Core under `+impl/`. Gate: `olhoffcurrent_paths`. |
| `/Users/piotrek/Programming/Matlab/Olhoff` | `DEVELOPMENT_UPSTREAM` | no — **blocked by absolute path** | yes, freely | git, branch `architecture/canonical-config`, HEAD `695f03b`, clean at promotion. Research/development tree. |
| `analysis/OlhoffM4Reconstruction` | `FROZEN_EVIDENCE` | no — blocked | **no** — hash-pinned by `IMPORT_MANIFEST.json` | The frozen conference reconstruction. Production **before** this promotion. Core under `+frozen/`. |
| `Matlab/reproduction2007` | `HISTORICAL` | no — blocked | no | Clean-room Du–Olhoff 2007 reproduction (Eq. 22 LP + paper-literal MMA). Still dispatched by `run_topopt_from_json` key `OlhoffDu2007Repro` for **non-Olhoff-column** historical work. Shares **49 bare names** with upstream. |
| `analysis/OlhoffApproach` | `HISTORICAL` | no — blocked | no | Original bound-formulation MMA implementation + Python port. **Untracked in git** (88 files). |
| `analysis/OlhoffApproachExact` | `HISTORICAL` | no — blocked | no | The "exact Olhoff 2014" line: own FE, multiplicity, generalized gradients. **Untracked** (836 files). |
| `analysis/OlhoffRegularized` | `HISTORICAL` | no — blocked | no | Globalized variant on the `reproduction2007` primitives. **Untracked** (35 files). |
| `analysis/OlhoffReproduced2007` | `HISTORICAL` | no — blocked | no | Thin runner exposing `reproduction2007` on Yuksel geometries. **Untracked** (5 files). |
| `analysis/OlhoffApproachExactOpus` | `AUDIT_ONLY` | no — blocked | no | Clean-room re-derivation. **No `.m` or `.py` remains on disk** — only `.png`/`.csv`/`.mat` artifacts, so not MATLAB-executable today. Blocked anyway. |
| `analysis/olhoff_stabilization_audit` | `HISTORICAL` | no — blocked | no | Superseded stabilization profile; contains `olhoffOptStabilized.m`. What the *previous* benchmark driver dispatched. |
| `analysis/olhoff_native_convergence` | `AUDIT_ONLY` | no — blocked | no | `olhoffOptTelemetry.m`, `nativeConvergenceDetector.m`. |
| `analysis/olhoff_fixed_budget_audit` | `AUDIT_ONLY` | no — blocked | no | Fixed-budget audit runners. |
| `analysis/olhoff_practical_convergence_audit` | `AUDIT_ONLY` | no — blocked | no | One offline audit runner. |
| `analysis/olhoff_nested_mma_route_audit` | `AUDIT_ONLY` | no — blocked | no | Python + reports only; no `.m`. |
| `analysis/OlhoffExperiments` | `EXPERIMENTAL` | no — blocked | n/a | **Does not exist yet.** Blocked pre-emptively so it cannot become a production dependency by accident when it is created. |

**`UNKNOWN` entries: none.** Every discovered tree is classified.

---

## Enforcement

Blocking is not documentary. `olhoffcurrent_forbidden_paths` names every tree
above, and `olhoffcurrent_assert_dispatch` refuses to return if any of them is
on the MATLAB path or supplies **any** candidate — winning or shadowed — for a
symbol `analysis/OlhoffCurrent` owns.

The blacklist is **not** the whole gate. A bare `.m` file carrying one of our 32
owned names, outside this tree and outside the MATLAB installation, is a blocker
**even when nothing here declares it** — so an Olhoff tree added tomorrow still
fails closed without anyone remembering to update this file.

Negative tests: `analysis/OlhoffCurrent/tests/test_path_isolation.m`.

---

## Technical debt

1. **Four executable historical trees are untracked in git** —
   `OlhoffApproach`, `OlhoffApproachExact`, `OlhoffRegularized`,
   `OlhoffReproduced2007`: **959 files on disk, 0 in version control.** They are
   not `.gitignore`d, merely never added. Their contents are unversioned, and no
   commit pins what they held when the results citing them were produced.
   Resolving this means committing or deliberately archiving them — historical
   reorganization, **deferred by §15 of the promotion brief**.

2. **`analysis/OLHOFF_IMPLEMENTATION_STATUS.md`** predates this promotion and
   still names `analysis/OlhoffM4Reconstruction` as "conference-active". It
   remains accurate as a description of the conference campaign that produced
   the frozen results, and is deliberately left unedited: it is referenced by
   the frozen audit trail. **This map supersedes it for the question "what may
   production execute?"**

3. Deferred by §15 and not attempted here: moving any historical tree,
   consolidating audits, creating an archive, deleting duplicates, or populating
   `analysis/OlhoffExperiments`.
