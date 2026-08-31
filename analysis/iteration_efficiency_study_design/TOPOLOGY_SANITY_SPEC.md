# Phase 1C quantitative topology-sanity specification

## 1. Purpose and C1 delta

The topology gate rejects gross/pathological structural invalidity. It does not rank
cleanliness, truss elegance, fragmentation aesthetics, or numerical speck count.

Phase 1C accepts Critical Finding C1. The superseded T1 gate required both each detached
component and their aggregate to contain fewer than five elements. The aggregate clause
had no resolution derivation and was reported by the original audit to admit no frozen
Olhoff state at 800x100. That case is now unavailable/RUN_ERROR/N/A and is not inferred;
the aggregate clause is deleted on the available evidence. The former constant five-element
rule, derived from Olhoff's native `rmin=1.3`,
is also not described or retained as a common scale.

## 2. Required physical connectivity

The benchmark is the fixed 8-by-1 domain with both translational degrees of freedom fixed
at the mid-height node of each end face. All production meshes must have even `nely`; this
makes the floor/round support indices used by the three implementations coincide.

For each support node, form the footprint of incident Q4 elements. On the exact-count
binary field, `C_required=1` iff one four-neighbor component intersects both footprints.
This support-to-support path is the only common hard path:

- Proposed uses a distributed semi-harmonic load;
- Yuksel changes from a point load to a design-dependent inertial load;
- Olhoff's eigenproblem has no external loaded region.

There is therefore no common support-to-load rule to impose. Column-to-column
`left_right_connected` is a compatibility diagnostic only.

## 3. Exact-count binary projection

For raw physical densities `x_e` and target `Vf`:

1. `nSolid=round(Vf*Ne)`;
2. order by decreasing density;
3. break exact ties by increasing global element index;
4. set exactly the first `nSolid` elements to one.

Raw density remains the spectral field and raw relative volume remains the volume gate.
Binary density is used only for structural connectivity/speck classification and binary
spectral diagnostics.

## 4. Method-neutral significant-component scale

Native filter radii are method properties, not a neutral yardstick: Olhoff, Proposed, and
Yuksel use 1.3, 2.0, and 2.5 element widths (footprints 5, 9, and 21). Phase 1C instead
uses the fixed FE geometry and preregistered mesh family.

All elements are square because `8/nelx=1/nely`. Let the coarsest production mesh be
`160x20`, with element area

\[
A_{e,0}=8/(160\cdot20)=0.0025.
\]

Define the smallest **physically significant detached component** as the area of a 2x2
Q4 patch on that mesh:

\[
A_{sig}=4A_{e,0}=0.01.
\]

A 2x2 patch is the smallest two-dimensional Q4 patch containing an interior shared node;
single cells and one-cell-wide numerical remnants do not define the common physical
feature scale. At mesh `j`, a detached component is significant iff

\[
n_c A_e(j)\ge A_{sig},\qquad
a_{sig}(j)=\lceil A_{sig}/A_e(j)\rceil.
\]

For the nine meshes, `a_sig` is exactly:

| mesh | 160x20 | 240x30 | 320x40 | 400x50 | 480x60 | 560x70 | 640x80 | 720x90 | 800x100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| significant at area >= | 4 | 9 | 16 | 25 | 36 | 49 | 64 | 81 | 100 elements |

Thus the physical threshold is constant while its element count grows with refinement.
A constant element count was rejected because its physical area and solid-volume share
shrink 25-fold over this mesh family.

## 5. Phase-1C baseline gate

The hard baseline is

\[
H_T(k)=[C_{required}(k)=1]\land
[\max_c A_c^{detached}(k)<A_{sig}].
\]

Equivalently, `C_required==1 && n_islands_significant==0`, using `a_sig(mesh)` above.
There is **no aggregate detached-area veto**. Many individually sub-resolution specks do
not become a single physical component merely because their areas sum. Aggregate area,
component count, and LCC remain mandatory diagnostics.

## 6. Frozen-evidence feasibility check

Phase 1C performed read-only exact-count/four-connectivity recomputation on the available
frozen Olhoff trajectories. No optimizer or MATLAB production calculation was run.

| mesh | states | `a_sig` | support pass % | repaired-gate pass % | longest repaired run | final max detached | final pass |
|---|---:|---:|---:|---:|---:|---:|:--:|
| 160x20 | 1601 | 4 | 98.88 | 66.52 | 957 | 0 | yes |
| 240x30 | 1601 | 9 | 98.88 | 93.07 | 1319 | 0 | yes |
| 320x40 | 1601 | 16 | 98.88 | 97.69 | 1492 | 12 | yes |
| 400x50 | 1601 | 25 | 99.00 | 96.63 | 1476 | 2 | yes |
| 480x60 | 358 | 36 | 95.53 | 90.78 | 272 | 13 | yes* |
| 560x70 | 400 | 49 | 96.00 | 88.00 | 307 | 10 | yes* |
| 640x80 | 1067 | 64 | 98.50 | 95.03 | 925 | 4 | yes* |
| 720x90 | 1601 | 81 | 99.00 | 97.81 | 1517 | 8 | yes |
| 800x100 | N/A | 100 | N/A | UNVERIFIABLE_AT_PRESENT | N/A | N/A | N/A |

`*` These trajectories later terminate in the LP solver; topology is not the reported
solver status. The 800x100 trajectory artifact is zero bytes, and the frozen endpoint is
`RUN_ERROR` with E1 `N/A`. Its topology evidence is therefore
`UNVERIFIABLE_AT_PRESENT`: no state count, pass fraction, persistence run, or final topology
measurement is inferred from the missing case. Regeneration is not required for protocol
freeze.

Other available frozen fields are limited. Proposed's 160x20 diagnostic final field
passes (`C_required=1`, no detached component). Yuksel's 800x100 diagnostic final field
passes (`C_required=1`, largest detached component 1, aggregate detached area 4, against
`a_sig=100`). Final-campaign Proposed/Yuksel trajectories were not stored, so no honest
all-state pass fraction exists for their other meshes. This absence is classified as
requiring new trajectory instrumentation, not silently inferred from endpoint CSVs.

The repaired rule fails early unformed states, does not let aggregate speck count dominate,
and no longer structurally excludes the fine Olhoff cases. These diagnostics establish
logical feasibility, not a desired method ranking.

## 7. Diagnostics and sensitivities

Record every state:

- `C_required`, component count, largest-component fraction;
- all detached component element counts and physical areas;
- largest and aggregate detached area, `n_islands_all`, `n_islands_significant`;
- the mesh-specific implied LCC bound and detached fraction;
- raw `x>=0.5` counterparts, grayness, and binary turnover.

T0 (literally zero islands) is a **known strict diagnostic**, not a sensitivity: frozen
evidence already shows it censors fine Olhoff trajectories. The permissiveness sensitivity
uses method-neutral FE patch scales of 1x1 and 3x3 coarsest-mesh cells around the 2x2
baseline, with physical areas `A_e0` and `9A_e0`. It is OAT, cannot rescue the baseline,
and is labelled `TOPOLOGY_SCALE_SENSITIVE` if status/order changes. A fixed fractional
LCC threshold remains rejected because no neutral percentage exists and its absolute
allowance grows with mesh size.

## 8. Persistence and images

The repaired gate must pass at every state in the common P-window. Images do not decide
acceptance. Show paired raw/binary fields at `k_enter` and `k_cert` where distinct, common
orientation/support markers and `[0,1]` color scale, no smoothing or pseudo-load. A
`NOT_REACHED` cell shows the last observed base-valid state with its exact limiting status.
