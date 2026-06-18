# Exp-1 result — CC frequency maximization at 80×20 vs 40×5 (unchanged settings, mesh only)

**What was run:** the existing `OlhoffApproachExact` MATLAB solver (`topopt_freq_exact.m`), **verbatim**, via `experiments/run_cc_meshcompare*.m`. No fixes, heuristics, continuation, projection, or optimizer changes were introduced. The *only* change between the two runs is `nelx,nely`: 40×5 → 80×20. `rmin_elem=2.5` is held fixed **in element units** (the literal "unchanged setting"). MATLAB R2025b. Designs/freqs cross-checked bit-for-bit with the independent clean-room FE.

Two setting regimes were exercised because the "disconnected branch" under study only exists in one of them:

- **Regime A — literal `run_clamped_clamped_exact.m`** (`move_lim=Inf, alpha=1, outer=300`): collapses to a ~0 Hz mechanism at **both** meshes (40×5 and 80×20 → ω₁≈0.015, N=1). This is the documented paper-literal divergence; it is uninformative about connectivity. *(Not the basis of the disconnection finding.)*
- **Regime B — the audit baseline that produced the analyzed disconnected design** (`move_lim=0.2, outer_move=0.2, alpha=0.5, outer=80, mult_tol=1e-3`): this is the regime whose 40×5 design (disconnected, ω₁≈462/414) the Phase-5 analysis dissected. Holding it fixed and changing **only the mesh** is the controlled experiment. **All results below are Regime B.**

Validation: Regime-B 40×5 reproduces the on-record baseline **exactly** — final ω₁=413.869 vs `baseline_trace` 413.87. So the MATLAB run is faithful.

---

## Direct answers to the questions

| Question | 40×5 (control) | **80×20 (this experiment)** |
|---|---|---|
| **Topology connected?** | **No** — 2 disconnected blocks (2 components at 4- and 8-connectivity) | **Yes** — a single **8-connected** component (diagonal-braced); continuous solid load path across the whole span |
| **Central span / brace present?** | **No** — center emptied (center-third mean ρ=0.069, some columns ρ≈0.002) | **Yes** — center-third mean ρ=**0.322**, every column has a cell with ρ≥0.585; clear **X-bracing** spanning the domain |
| **Final ω₁ / ω₂** | 413.9 / 450.1 | **314.0 / 423.8** |
| **Best-iter ω₁ / ω₂** | 593.7 (iter 30) | **330.3 / 333.0 (iter 14)** |
| **N (multiplicity)** | 1 (not bimodal) | **1 (not bimodal)** |
| **Volume** | 0.499 (active) | **0.500 (active)** |
| **vs Fig. 3c (456.4, bimodal, connected)** | wrong topology (disconnected), ω too high via massless void | **right topology class (connected X-braced full-span truss); ω₁=314 < 456.4; N=1, not yet bimodal** |

### Topology maps (density, bottom row = y=0)
```
40×5  : @@@@@%##*+==-:.         .::-++#%@@@@@@@@      <- two solid blocks
        ... (empty center) ...                          (DISCONNECTED)

80×20 : @@@@@@@@@@@@@*-     .:+%@@@...@@%#+:.    :+@@@@@@@   <- end blocks +
        @@@@@@%##%@@@%+:  .:+#*=:..    :-=+=:  .-+%@@@@@      X-braced
        :=#@@%#**+- :+%%%#+*#+:.   ...   -*#*#%@@#=:.         diagonal
        ... symmetric cross-braces span the full length ...  truss
                                                              (CONNECTED)
```
The 80×20 design is the classic connected, symmetric, cross-braced clamped-beam vibration topology — **the same morphological class as the paper's Fig. 3c.**

---

## What this confirms — and the important caveat

**Hypothesis G (mesh coarseness) is strongly confirmed.** Changing *only* the mesh flips the outcome from a disconnected two-block design (40×5) to a connected, full-span, X-braced truss (80×20) — the Fig. 3c topology class. The 40×5 mesh (5 rows) literally cannot represent thin diagonal braces, so the only ω₁-ascent direction there is emptying→disconnecting; the 80×20 mesh can form braces, giving a *connected* ascent direction the optimizer follows. This is direct, controlled evidence that **the reconstruction's disconnection at 40×5 is substantially a discretization artifact.**

**Caveat (hypothesis A/H persist):** the connected 80×20 design is **not** the global optimum even at 80×20. Under the du2007_c1 near-massless-void model, disconnected designs at 80×20 still score **higher** than the connected one the optimizer found:

| 80×20 design (vol 0.5) | components (8-conn) | center ρ | ω₁ |
|---|---|---|---|
| optimizer result (connected) | 1 | 0.322 | **314** |
| same design, center emptied | 2 | 0.001 | 522 |
| clean two-block | 2 | 0.001 | **689** |

So disconnection remains the higher-frequency branch under the massless-void mass model at *both* meshes — the finer mesh did **not** make connected the *global* optimum. What it changed is the **basin the move-limited optimizer settles into**: at 40×5 the path runs to the disconnected branch; at 80×20 the path settles into the connected braced basin (and, in 80 iterations, does not leave it). This is exactly the behavior expected if **the paper's Fig. 3c is the connected braced local optimum that a sufficiently resolved optimizer naturally lands on**, while the disconnected design is a competing higher-ω branch that the coarse mesh exposes.

**Not yet bimodal:** the 80×20 run ends at N=1, ω₁=314, still oscillating — it reached the connected Fig-3c *topology* but not the converged *bimodal* 456.4 optimum (80 iterations, move-limited, did not converge). Reaching 456.4 would require running to convergence (more iterations) — deliberately **not** done here, since that is a setting change beyond "mesh only."

---

## One-line conclusion

**Changing only the mesh from 40×5 to 80×20 turns the disconnected two-block design into a connected, full-span, X-braced truss (the Fig. 3c topology class) with a continuous central span — confirming that coarse-mesh discretization was a primary driver of the disconnection — while ω₁ (314, N=1) remains below the paper's converged bimodal 456.4, and disconnected designs remain higher-frequency under the du2007_c1 void-mass model, so the connected outcome reflects basin/resolution selection (as in the paper), not a change of the underlying global optimum.**

Artifacts: `experiments/results/cc_b_40x5.mat`, `cc_b_80x20.mat`, `rho_b_*.csv`; runners `run_cc_meshcompare.m` (regime A), `run_cc_meshcompare_b.m` (regime B).
