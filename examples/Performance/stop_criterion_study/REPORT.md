# One stopping criterion for the three methods: why per-iteration design-change norms cannot be made universal, and what to use instead (2026-10-09)

**Question.** The nine-mesh Table 1 campaign uses a different stopping rule per method (max|Δx_e| for Yuksel and the proposed approach, ‖Δx‖₂ < 0.05√(n_e/3200) for Du–Olhoff). A common rule is needed, but every candidate tried so far (max|Δx|, ‖Δx‖₂/‖x‖₂) has the same defect: a tolerance that lets 800×100 converge cleanly produces long flat tails on the coarse meshes, and a tolerance that stops 240×30 at ~50 iterations leaves 800×100 gray.

**Answer in one paragraph.** This is not a tolerance-selection problem; it is a metric problem. All per-iteration design-change norms (max, L2, RMS, relative L2) measure how fast the optimizer is still *moving elements*, not how far the design is from its final state. Once ω₁ and the discreteness M_nd have settled, all three optimizers keep moving boundary elements by 0.01–0.10 per iteration (OC and the adaptive-box MMA both oscillate at the boundary), so the norms sit on a *noise floor* that is 3–10× above the thresholds in use, and then decay slowly and non-monotonically (Table B, Fig. 1–3). A stop therefore happens when the noise happens to dip below ε: with ε near the floor the dip comes early and may be a mid-run plateau (the gray 800×100 Olhoff design at c = 0.2 stopped at outer iteration 37), with ε below the floor the dip comes hundreds of iterations after stagnation (240×30 Proposed at ‖Δx‖/‖x‖ < 10⁻³: iteration 271 versus stagnation at ~45). Both failure modes are properties of the metric, so no ε fixes both, and the "universal" ε does not exist for any of these norms. The remedy is to stop on *stagnation of the quantities the paper actually reports*, measured over a window: **stop at the first iteration k ≥ W at which, over the last W = 10 iterations, the method's own objective varied by less than 0.1 % and M_nd varied by less than 0.005** (rule R1 below). R1 is dimensionless, contains no n_e, is identical for all three methods, and in an offline replay on 23 recorded histories (three methods, 160×20 … 800×100) it stops within 0.48–2.68× of the measured stagnation iteration with an ω₁ deficit of at most 0.91 % relative to running 300 (Proposed, Olhoff) or 600 more (Yuksel stage 2) iterations. A windowed max|Δx| rule (R2) is a defensible second choice, but it is not universal: it never fires for the proposed method at 800×100 because single OC elements keep jumping by more than 0.04.

No code was changed for this study. Everything below was produced with the existing solvers and drivers through recording runs placed in a scratch directory; scripts, per-iteration histories (CSV) and figures are in `data/`, `fig/` and `scripts/` next to this report.

---

## 1. What was measured

**Recording runs.** Each method was run on the simply supported beam of Table 1 with its production configuration (frozen profiles `proposed_practical_move02_tol001`, `yuksel_practical_move01_tol001`, production Olhoff preset `duOlhoffPedersenAdaptiveBoxSensitivityFiltered`) but with the native stop *disabled or extended*, so that the whole trajectory past the usual stop is on record:

| method | meshes | horizon | recorded per iteration |
|---|---|---|---|
| Proposed | 160×20 … 800×100 (9) | 300 SIMP iterations (production stop at max|Δx| ≤ 0.01 recorded but not acted on) | x_phys; max|Δx|, ‖Δx‖₂/‖x‖₂, RMS(Δx), M_nd, objective, E1 structural ω₁ (every iteration up to 320×40, every 5th from 400×50) |
| Yuksel | 160×20 … 800×100 (9) | stage 1 on its native rule (max|Δx| < 0.01, budget 2000); stage 2 extended to 600 iterations | same, E1 ω₁ every 5th iteration |
| Du–Olhoff | 160×20, 240×30, 320×40, 400×50, 800×100 | 300 outer iterations, stop disabled | ρ replayed from the recorded increments; same metrics, native ω₁ |

The stopping rule of every method is a pure *observer* of the trajectory (it changes when the loop exits, not what the loop does), with one exception: Yuksel's stage-1 tolerance decides the design handed to stage 2. Hence every candidate rule can be replayed exactly on these records (checked: the replay reproduces the production stop iterations 107/236/207/182 for Proposed and 121/111/101/93 for Olhoff at c = 0.05, and 58/52/67/62 at c = 0.2 — the numbers of the `campaign_mac_convergence_*` campaigns).

**Ground truth for "the design has stopped changing".** For each record, k_stag is the first iteration after which ω₁ (E1 structural ω₁ for Proposed/Yuksel, native for Olhoff) stays within 0.5 % and M_nd within 0.01 of the end of the record. A rule is judged by where it stops relative to k_stag and by the ω₁ it leaves on the table relative to the end state.

**Caveats of the record.** (i) Yuksel's stage 2 is still improving slowly at the 600-iteration cap on the finer meshes, so its "end state" is a cap, not a converged state; the losses quoted for Yuksel are relative to that cap. (ii) For Proposed and Yuksel the late trajectory *lowers* ω₁ (Table B: Proposed 800×100 peaks at 163.4 rad/s and ends at 161.4; Yuksel 160×20 peaks at 158.3 and ends at 156.9), because their objective is a static surrogate, not ω₁; negative "losses" in the tables are this effect. (iii) One load case, one starting design, one filter radius per method; thresholds were chosen after seeing these data, so they are calibrated on this benchmark and should be preregistered, not re-tuned, for any other case.

## 2. Diagnosis: why no ε works for a per-iteration change norm

Figures 1–3 (`fig/traces_<method>.png`) show the four quantities per iteration for every mesh; Table B gives their levels at stagnation and at the end of the record.

1. **The norms do not go to zero when the design stops changing.** At k_stag the median max|Δx| is 0.014–0.074 (Proposed), 0.03–0.10 (Yuksel) and 0.05–0.07 (Olhoff); the median ‖Δx‖₂/‖x‖₂ is 2.5–7.5·10⁻³, 0.9–4.8·10⁻³ and 5.3–6.4·10⁻³. These are the optimizer's boundary oscillation (OC with a move limit flips boundary elements back and forth; the Olhoff adaptive box does the same at the 0.002–0.1 scale — the period-2 behaviour already documented for 160×20). They carry no information about ω₁ or M_nd, both of which are flat.
2. **A threshold below the floor is reached by chance.** With max|Δx| ≤ 0.01 the proposed method stops at 107, 236, 207, 182, 219, 256, never, 297, never (160×20 … 800×100) although k_stag is 29–95: the stop is the first iteration at which no element happened to jump. The waiting time for that event grows with the number of boundary elements, i.e. with the mesh. This is the "flat tail" the user sees on 240×30 at ‖Δx‖/‖x‖ < 10⁻³ (stop at 271). The earlier finding that 0.019 and 0.020 give 98 and 47 iterations at 320×40 is the same lottery.
3. **A threshold near the floor catches mid-run plateaus.** Du–Olhoff's RMS(Δx) dips during the adaptive-box transitions well before the design is finished; at c = 0.2 (RMS < 3.5·10⁻³) the campaign stopped 640×80/720×90/800×100 at outer 38/33/37 with ω₁ = 156/154/153 rad/s, 7 % below the converged 165. The 800×100 record (Fig. 3) shows what that run was interrupted in: Du–Olhoff on this mesh de-grays in two slow phases, M_nd = 0.53 → 0.30 (iterations 25 → 100) and 0.26 → 0.16 (175 → 240), with ω₁ flat at 163.6–163.8 between iterations 125 and 175 while the second phase is being prepared; k_stag = 232, against 34–59 on 160×20–400×50. The gray design at "iteration 125" in the current figure set (ω₁ = 163.6, M_nd = 0.275) is therefore an unfinished run, not a mesh artefact, and every rule that fired on that plateau — single-iteration RMS/relative-L2 at 5·10⁻³ (37), max|Δx| ≤ 0.04 with persistence (105), the objective-only window (122) — produced it. The rules that waited for the second phase (relative L2 < 10⁻³: 256; c = 0.05: 246; R1: 241) are the same rules that produce the 100-iteration flat tails on 160×20–240×30, which is the user's dilemma restated: the dip statistics are different on the two meshes, the physics is not.
4. **Scaling the tolerance with n_e cannot repair this.** ‖Δx‖₂ < 0.05√(n_e/3200) is literally RMS(Δx) < 8.84·10⁻⁴, a mesh-intensive quantity (so the "tailored to 3200" objection is cosmetic: write it as RMS). The relative norm ‖Δx‖₂/‖x‖₂ ≈ RMS(Δx)/√(mean x²) is the same quantity up to a factor ≈ 1.4–1.5. Both inherit the floor and the lottery. Table A shows the RMS rule stopping Proposed at 209–289 and the relative-L2 rule at 219–294, for designs that stagnated at 30–95.
5. **Mesh dependence is therefore not a scaling law to be fitted; it is the statistics of dips.** Across meshes the floor itself is nearly flat (Table B), while the time to the first dip below a given ε is erratic (Proposed at ε = 0.01: from 107 to never). No monotone ε(n_e) maps onto that.

## 3. Candidate common rules, replayed

Table C and `fig/stop_rules_summary.png` give, per method and mesh, the stop iteration and the ω₁ deficit of each candidate; Table D summarises over all records.

**R1 (recommended) — windowed stagnation of objective and discreteness.** With f_k the method's own objective (Du–Olhoff: ω₁ or λ₁; Yuksel and Proposed: their compliance-type objective) and M_nd,k = 4·mean(x_phys(1−x_phys)), stop at the first k ≥ W with

    (max_{k−W≤j≤k} f_j − min_{k−W≤j≤k} f_j) / |f_k| < 10⁻³   and   max_{k−W≤j≤k} M_nd,j − min_{k−W≤j≤k} M_nd,j < 5·10⁻³ ,   W = 10.

Properties: dimensionless; no n_e, no move limit, no filter radius in it; both quantities are already computed or trivially available in every loop (no extra eigensolve); it is a statement about the outcome the paper reports (frequency and black-and-whiteness), which is what a reviewer wants to be sure has converged. The objective window alone stops too early on the Olhoff plateau at 800×100 (iteration 122, 1.1 % of ω₁ and half of the de-graying still to come); the M_nd window alone stops too early when the design is discrete but ω₁ is still moving (Yuksel); the conjunction is what makes it universal. It is also the only candidate that resolves the user's dilemma as stated: it stops Du–Olhoff 240×30 at 57 (stagnation 34) and 800×100 at 241 (stagnation 232) with the same two numbers in the rule. Replay: R1 fired on all 23 of 23 histories. Stop iteration relative to k_stag: 0.48–2.68 (Proposed), 0.81–1.23 (Yuksel stage 2), 1.04–1.68 (Du–Olhoff); ω₁ at the stop relative to the end state: -0.24 … +0.33 % (Proposed), -0.46 … +0.91 % (Yuksel), -0.13 … +0.27 % (Du–Olhoff); the worst case is yuksel 480x60 (+0.91 %). M_nd at the stop exceeds its end value by at most 0.010. Per-mesh numbers: Table C.

**R2 — persistence version of the current rule.** max|Δx_e| ≤ 0.04 for W = 10 consecutive iterations. Same ω₁ behaviour as R1 on the coarse meshes, but it never fires for Proposed at 800×100 within 300 iterations (single elements keep jumping by 0.05–0.1), it stops Du–Olhoff 800×100 at 105 with M_nd = 0.29 (gray, 1.3 % of ω₁ missing), it never fires for Yuksel at 640×80–800×100 because some element moves by the full move limit 0.1 at every stage-2 iteration (Table B: median max|Δx| = 0.100 at the end of the record), and its threshold is 20 % of the move limit of one method and 40 % of another's. Acceptable as a documented fallback, not as the common rule.

**Rules not recommended.** Single-iteration max|Δx| ≤ 0.025 (the 2026-10-07 retune) is tied for ω₁ but is the dip lottery with a larger ε; the two objective-only windows are fine for Proposed and Yuksel but let Du–Olhoff stop on its ω₁ plateau at 240×30 with 0.5 % left; element flip-fraction and M_nd-only rules miss up to 2.6 % of ω₁.

## 4. What this changes in the paper and the campaign

1. **One sentence replaces three.** "All three implementations, and both stages of the Yuksel–Yilmaz method, terminate on the same rule: the first iteration at which, over the preceding ten iterations, the method's objective has varied by less than 0.1 % and the discreteness measure M_nd by less than 0.005." Add that the rule is a stagnation test, not an optimality test, and that it was verified offline against 300-iteration extended runs (ω₁ within x % at every mesh; cite the supplementary table).
2. **Yuksel's stage-1 tolerance is part of the algorithm, not of the benchmark.** Changing it from the published 0.01 to 0.04 (the current driver) changes which design stage 2 starts from, hence the method. Either keep the native stage-1 rule and apply R1 to stage 2 only, or apply R1 to both stages and say so; the current mix is the one thing a reviewer can legitimately call unfair.
3. **Fix the inconsistency already in the text.** Section 4.1 states max|Δx_e| < 10⁻³ for Yuksel and the proposed method, the frozen profiles use 0.01, and the current driver uses 0.04; the Table 1 figures in the text (345 outer iterations, 1083 and 263 iterations at 400×50) belong to an older campaign. Whatever rule is adopted, the text, the manifest and the table must quote the same one.
4. **The √(n_e/3200) form should go** even if the Olhoff native rule were kept for a sensitivity row: present it as RMS(Δx) < 8.8·10⁻⁴.
5. **Regenerating Table 1 is cheap.** Because the rules are observers, the iteration counts and the designs at the new stop follow from the recorded histories without rerunning anything (discovery pass); only the clean timing needs the two-pass protocol already written in the benchmark plan (run to the discovered k* with history and diagnostics off). Expected effect on the counts: Du–Olhoff 57–78 instead of 93–122 outer iterations on 160×20–400×50 (its cost is dominated by the tail of nested MMA solves) and ≈ 240 at 800×100, where the method genuinely needs them; Proposed 44–98 instead of 107–297; Yuksel stage 2 unchanged in order of magnitude. The nine-mesh scaling exponent of Du–Olhoff will steepen accordingly: its iteration count is flat up to 400×50 and then grows, and that is a property of the method on fine meshes, not of the stop rule.
6. **Say what the tail does.** For the two static-surrogate methods the extended tail lowers ω₁ by up to 1.2 % while making the design slightly more discrete; for Du–Olhoff the tail adds ≤ 0.3 %. This supports stopping at stagnation and is worth one sentence in the response to Reviewer 3's point 12 (gray regions / incomplete convergence): the gray that remains at the stop is the sensitivity-filter band, not an unfinished run (shown by the extended records: M_nd changes by < 0.01 after k_stag on every mesh).

## 5. Open points

* Du–Olhoff at 480×60–720×90 was not recorded (each stop-disabled run costs 1–3 h); the replay covers 160×20–400×50 and 800×100 for that method.
* The thresholds 10⁻³ / 5·10⁻³ / W = 10 were chosen on these data. The sensitivity to W (10 vs 20) and to the objective tolerance (10⁻³ vs 5·10⁻⁴) is in Table D: tighter settings cost 10–50 % more iterations for ≤ 0.1 % of ω₁. They should be frozen before the other examples (clamped beam, building) are rerun and reported as such.
* Yuksel's stage-2 at the fine meshes is the one place where "stagnation" and "converged" differ by up to 0.9 % in ω₁ within 600 iterations; if that matters, the honest statement is that the method has a slow tail, not that the rule is wrong.

## Appendix: tables from the replay

### Table A. Where the current rules stop, versus where the design has actually stagnated

k_stag = first iteration after which omega_1 stays within 0.5 % and M_nd within 0.01 of the end of the extended record (300 iterations for Proposed and Du-Olhoff, stage-2 cap 600 for Yuksel). Entries: stop iteration (omega_1 loss vs end state, %). "none" = the rule never fired within the record.


**Proposed**

| mesh | record | k_stag | max\|dx\| <= 0.01 | RMS(dx) < 8.8e-4 (= 0.05*sqrt(n_e/3200)) | \|\|dx\|\|/\|\|x\|\| < 1e-3 | max\|dx\| <= 0.04 | RMS(dx) < 3.5e-3 (c = 0.2) | \|\|dx\|\|/\|\|x\|\| < 5e-3 |
|---|---|---|---|---|---|---|---|---|
| 160x20 | 299 | 29 | 107 (-0.12) | none | none | 32 (-0.13) | 43 (-0.24) | 45 (-0.22) |
| 240x30 | 299 | 95 | 236 (-0.03) | 266 (-0.01) | 271 (-0.00) | 36 (+0.31) | 42 (+0.33) | 43 (+0.33) |
| 320x40 | 299 | 30 | 207 (-0.05) | 209 (-0.05) | 219 (-0.04) | 35 (-0.26) | 40 (-0.17) | 41 (-0.16) |
| 400x50 | 299 | 34 | 182 (-0.01) | 221 (+0.01) | 249 (+0.01) | 32 (-0.66) | 38 (-0.42) | 38 (-0.42) |
| 480x60 | 299 | 30 | 219 (+0.03) | 257 (+0.01) | 277 (+0.00) | 50 (+0.01) | 50 (+0.01) | 51 (+0.01) |
| 560x70 | 299 | 39 | 256 (-0.02) | 254 (-0.02) | 280 (-0.01) | 57 (-0.06) | 52 (-0.09) | 52 (-0.09) |
| 640x80 | 299 | 39 | none | 289 (+0.00) | none | 47 (-0.18) | 46 (-0.18) | 47 (-0.18) |
| 720x90 | 299 | 39 | 297 (+0.00) | 234 (+0.02) | 285 (+0.01) | 45 (-0.31) | 50 (-0.19) | 50 (-0.19) |
| 800x100 | 299 | 44 | none | 263 (+0.02) | 294 (+0.01) | 56 (-0.18) | 55 (-0.18) | 55 (-0.18) |

**Yuksel (stage 2)**

| mesh | record | k_stag | max\|dx\| <= 0.01 | RMS(dx) < 8.8e-4 (= 0.05*sqrt(n_e/3200)) | \|\|dx\|\|/\|\|x\|\| < 1e-3 | max\|dx\| <= 0.04 | RMS(dx) < 3.5e-3 (c = 0.2) | \|\|dx\|\|/\|\|x\|\| < 5e-3 |
|---|---|---|---|---|---|---|---|---|
| 160x20 | 720 | 179 | 244 (-0.17) | 245 (-0.16) | 251 (-0.15) | 151 (-0.77) | 149 (-0.88) | 149 (-0.88) |
| 240x30 | 767 | 201 | 320 (-0.07) | 270 (-0.09) | 312 (-0.07) | 220 (-0.40) | 199 (-0.39) | 199 (-0.39) |
| 320x40 | 851 | 394 | 572 (-0.02) | 507 (+0.03) | 531 (+0.00) | 311 (+1.05) | 282 (+1.77) | 283 (+1.77) |
| 400x50 | 914 | 414 | 732 (+0.03) | 669 (+0.06) | 682 (+0.06) | 400 (+0.56) | 351 (+1.42) | 351 (+1.42) |
| 480x60 | 1185 | 884 | none | 781 (+0.72) | 782 (+0.72) | 781 (+0.72) | 639 (+1.58) | 639 (+1.58) |
| 560x70 | 1432 | 1119 | none | 1045 (+0.70) | 1068 (+0.63) | 1073 (+0.62) | 882 (+1.72) | 883 (+1.72) |
| 640x80 | 2141 | 1749 | none | 1853 (+0.28) | 1854 (+0.28) | 2006 (+0.09) | 1583 (+1.78) | 1583 (+1.78) |
| 720x90 | 1761 | 1374 | none | 1383 (+0.48) | 1491 (+0.31) | 1491 (+0.31) | 1212 (+1.51) | 1212 (+1.51) |
| 800x100 | 1749 | 1329 | none | 1424 (+0.33) | 1509 (+0.19) | 1604 (+0.12) | 1203 (+1.42) | 1203 (+1.42) |

**Du-Olhoff**

| mesh | record | k_stag | max\|dx\| <= 0.01 | RMS(dx) < 8.8e-4 (= 0.05*sqrt(n_e/3200)) | \|\|dx\|\|/\|\|x\|\| < 1e-3 | max\|dx\| <= 0.04 | RMS(dx) < 3.5e-3 (c = 0.2) | \|\|dx\|\|/\|\|x\|\| < 5e-3 |
|---|---|---|---|---|---|---|---|---|
| 160x20 | 300 | 53 | 122 (-0.01) | 121 (-0.01) | 122 (-0.01) | 67 (+0.25) | 58 (+0.32) | 58 (+0.32) |
| 240x30 | 300 | 34 | 89 (-0.04) | 111 (-0.02) | 114 (-0.03) | 61 (-0.04) | 52 (-0.07) | 52 (-0.07) |
| 320x40 | 300 | 59 | 85 (-0.10) | 101 (-0.08) | 128 (-0.06) | 67 (-0.22) | 67 (-0.22) | 67 (-0.22) |
| 400x50 | 300 | 52 | 79 (-0.13) | 93 (-0.08) | 105 (-0.06) | 60 (-0.33) | 62 (-0.31) | 62 (-0.31) |
| 800x100 | 300 | 232 | 245 (-0.04) | 246 (-0.04) | 256 (-0.04) | 18 (+9.27) | 37 (+7.35) | 37 (+7.35) |

### Table B. Level of the per-iteration change metrics once the design has stagnated

Median over the 20 iterations after k_stag (left) and over the last 20 recorded iterations (right). These are the values a threshold must exceed to stop at stagnation, and the floor it must stay above to stop at all.

| method | mesh | k_stag | max\|dx\| @stag | rel-L2 @stag | RMS @stag | max\|dx\| end | rel-L2 end | RMS end | omega_1 @stag | omega_1 max | omega_1 end | M_nd @stag | M_nd end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| proposed | 160x20 | 29 | 0.021 | 0.0061 | 0.0040 | 0.006 | 0.0021 | 0.0014 | 153.6 | 153.9 | 153.5 | 0.261 | 0.252 |
| proposed | 240x30 | 95 | 0.014 | 0.0025 | 0.0017 | 0.003 | 0.0005 | 0.0004 | 157.1 | 157.7 | 157.6 | 0.171 | 0.161 |
| proposed | 320x40 | 30 | 0.025 | 0.0050 | 0.0034 | 0.002 | 0.0004 | 0.0002 | 159.2 | 159.2 | 158.7 | 0.129 | 0.121 |
| proposed | 400x50 | 34 | 0.021 | 0.0037 | 0.0026 | 0.008 | 0.0007 | 0.0005 | 160.2 | 160.8 | 159.5 | 0.099 | 0.097 |
| proposed | 480x60 | 30 | 0.074 | 0.0075 | 0.0052 | 0.041 | 0.0013 | 0.0009 | 160.9 | 160.9 | 160.3 | 0.100 | 0.091 |
| proposed | 560x70 | 39 | 0.055 | 0.0055 | 0.0038 | 0.017 | 0.0010 | 0.0007 | 161.1 | 161.5 | 160.7 | 0.081 | 0.077 |
| proposed | 640x80 | 39 | 0.033 | 0.0041 | 0.0028 | 0.052 | 0.0013 | 0.0009 | 161.4 | 162.3 | 160.9 | 0.072 | 0.070 |
| proposed | 720x90 | 39 | 0.041 | 0.0050 | 0.0035 | 0.035 | 0.0010 | 0.0007 | 161.9 | 163.3 | 161.1 | 0.065 | 0.063 |
| proposed | 800x100 | 44 | 0.073 | 0.0050 | 0.0035 | 0.041 | 0.0011 | 0.0008 | 162.0 | 163.4 | 161.4 | 0.058 | 0.055 |
| yuksel | 160x20 | 179 | 0.030 | 0.0033 | 0.0023 | 0.001 | 0.0001 | 0.0001 | 157.7 | 158.3 | 156.9 | 0.077 | 0.080 |
| yuksel | 240x30 | 201 | 0.100 | 0.0048 | 0.0034 | 0.001 | 0.0000 | 0.0000 | 160.0 | 160.1 | 159.3 | 0.036 | 0.046 |
| yuksel | 320x40 | 394 | 0.091 | 0.0040 | 0.0028 | 0.021 | 0.0007 | 0.0005 | 160.9 | 160.9 | 160.7 | 0.026 | 0.035 |
| yuksel | 400x50 | 414 | 0.086 | 0.0027 | 0.0019 | 0.013 | 0.0006 | 0.0004 | 159.2 | 160.0 | 160.0 | 0.029 | 0.037 |
| yuksel | 480x60 | 884 | 0.031 | 0.0009 | 0.0006 | 0.016 | 0.0008 | 0.0005 | 159.7 | 160.5 | 160.5 | 0.024 | 0.029 |
| yuksel | 560x70 | 1119 | 0.100 | 0.0019 | 0.0014 | 0.049 | 0.0012 | 0.0009 | 159.5 | 160.3 | 160.3 | 0.019 | 0.023 |
| yuksel | 640x80 | 1749 | 0.100 | 0.0019 | 0.0013 | 0.078 | 0.0012 | 0.0009 | 158.9 | 159.7 | 159.7 | 0.012 | 0.014 |
| yuksel | 720x90 | 1374 | 0.100 | 0.0018 | 0.0013 | 0.100 | 0.0020 | 0.0014 | 159.5 | 160.3 | 160.3 | 0.012 | 0.015 |
| yuksel | 800x100 | 1329 | 0.100 | 0.0018 | 0.0012 | 0.100 | 0.0014 | 0.0010 | 159.5 | 160.3 | 160.3 | 0.009 | 0.013 |
| olhoff | 160x20 | 53 | 0.056 | 0.0055 | 0.0038 | 0.013 | 0.0007 | 0.0005 | 168.6 | 169.2 | 169.2 | 0.126 | 0.116 |
| olhoff | 240x30 | 34 | 0.073 | 0.0064 | 0.0044 | 0.007 | 0.0009 | 0.0006 | 167.5 | 167.7 | 167.3 | 0.127 | 0.123 |
| olhoff | 320x40 | 59 | 0.049 | 0.0053 | 0.0036 | 0.006 | 0.0008 | 0.0005 | 166.3 | 166.3 | 165.7 | 0.151 | 0.142 |
| olhoff | 400x50 | 52 | 0.028 | 0.0043 | 0.0030 | 0.003 | 0.0005 | 0.0003 | 167.0 | 167.0 | 166.3 | 0.132 | 0.123 |
| olhoff | 800x100 | 232 | 0.011 | 0.0015 | 0.0010 | 0.004 | 0.0006 | 0.0004 | 165.4 | 165.4 | 165.4 | 0.169 | 0.159 |

### Table C. Candidate common rules replayed on every recorded history

Entries: stop iteration (omega_1 loss vs end state, %). Yuksel: rule applied to stage 2 only (stage-1 handoff kept native).


**Proposed**

| mesh | k_stag | R1: objective range < 1e-3 AND M_nd range < 5e-3 over 10 it. | R1 tight: 5e-4 / 5e-3 / 10 it. | objective range < 1e-3 over 10 it. only | R2: max\|dx\| <= 0.04 for 10 consecutive it. | max\|dx\| <= 0.025 (single it.) | M_nd range < 5e-3 over 10 it. only |
|---|---|---|---|---|---|---|---|
| 160x20 | 29 | 44 (-0.24) | 48 (-0.24) | 44 (-0.24) | 41 (-0.22) | 36 (-0.21) | 41 (-0.22) |
| 240x30 | 95 | 46 (+0.33) | 48 (+0.33) | 46 (+0.33) | 45 (+0.33) | 42 (+0.33) | 39 (+0.32) |
| 320x40 | 30 | 54 (-0.13) | 56 (-0.13) | 54 (-0.13) | 44 (-0.14) | 41 (-0.16) | 41 (-0.16) |
| 400x50 | 34 | 91 (-0.09) | 132 (-0.04) | 91 (-0.09) | 41 (-0.32) | 37 (-0.42) | 36 (-0.42) |
| 480x60 | 30 | 62 (+0.03) | 71 (+0.03) | 62 (+0.03) | 59 (+0.02) | 59 (+0.02) | 42 (-0.09) |
| 560x70 | 39 | 64 (-0.05) | 68 (-0.04) | 64 (-0.05) | 76 (-0.05) | 69 (-0.04) | 42 (-0.26) |
| 640x80 | 39 | 67 (-0.03) | 72 (-0.03) | 67 (-0.03) | 56 (-0.06) | 55 (-0.06) | 38 (-0.57) |
| 720x90 | 39 | 82 (+0.02) | 96 (+0.04) | 82 (+0.02) | 68 (-0.03) | 60 (-0.06) | 36 (-0.76) |
| 800x100 | 44 | 98 (+0.08) | 106 (+0.09) | 98 (+0.08) | none | 126 (+0.09) | 42 (-0.56) |

**Yuksel (stage 2)**

| mesh | k_stag | R1: objective range < 1e-3 AND M_nd range < 5e-3 over 10 it. | R1 tight: 5e-4 / 5e-3 / 10 it. | objective range < 1e-3 over 10 it. only | R2: max\|dx\| <= 0.04 for 10 consecutive it. | max\|dx\| <= 0.025 (single it.) | M_nd range < 5e-3 over 10 it. only |
|---|---|---|---|---|---|---|---|
| 160x20 | 179 | 221 (-0.22) | 225 (-0.21) | 221 (-0.22) | 176 (-0.53) | 208 (-0.26) | 147 (-0.88) |
| 240x30 | 201 | 208 (-0.46) | 263 (-0.10) | 208 (-0.46) | 229 (-0.39) | 223 (-0.40) | 193 (-0.26) |
| 320x40 | 394 | 364 (-0.15) | 364 (-0.15) | 364 (-0.15) | 449 (+0.13) | 473 (+0.11) | 273 (+2.36) |
| 400x50 | 414 | 447 (+0.33) | 533 (+0.15) | 447 (+0.33) | 409 (+0.53) | 671 (+0.06) | 335 (+2.11) |
| 480x60 | 884 | 718 (+0.91) | 788 (+0.72) | 718 (+0.91) | 790 (+0.71) | 784 (+0.72) | 607 (+2.62) |
| 560x70 | 1119 | 1010 (+0.81) | 1235 (+0.23) | 1010 (+0.81) | 1291 (+0.13) | 1191 (+0.30) | 853 (+2.71) |
| 640x80 | 1749 | 1719 (+0.61) | 1770 (+0.45) | 1719 (+0.61) | none | none | 1563 (+2.40) |
| 720x90 | 1374 | 1325 (+0.65) | 1389 (+0.47) | 1325 (+0.65) | none | none | 1182 (+2.42) |
| 800x100 | 1329 | 1292 (+0.64) | 1342 (+0.47) | 1292 (+0.64) | none | none | 1171 (+2.28) |

**Du-Olhoff**

| mesh | k_stag | R1: objective range < 1e-3 AND M_nd range < 5e-3 over 10 it. | R1 tight: 5e-4 / 5e-3 / 10 it. | objective range < 1e-3 over 10 it. only | R2: max\|dx\| <= 0.04 for 10 consecutive it. | max\|dx\| <= 0.025 (single it.) | M_nd range < 5e-3 over 10 it. only |
|---|---|---|---|---|---|---|---|
| 160x20 | 53 | 66 (+0.27) | 77 (+0.22) | 66 (+0.27) | 81 (+0.23) | 72 (+0.21) | 62 (+0.29) |
| 240x30 | 34 | 57 (-0.05) | 60 (-0.05) | 46 (-0.16) | 70 (-0.04) | 64 (-0.05) | 47 (-0.12) |
| 320x40 | 59 | 78 (-0.10) | 81 (-0.10) | 65 (-0.28) | 90 (-0.10) | 69 (-0.18) | 71 (-0.15) |
| 400x50 | 52 | 78 (-0.13) | 84 (-0.11) | 59 (-0.35) | 69 (-0.21) | 64 (-0.30) | 67 (-0.25) |
| 800x100 | 232 | 241 (-0.04) | 241 (-0.04) | 122 (+1.09) | 105 (+1.31) | 130 (+1.05) | 241 (-0.04) |

### Table D. Summary over all recorded histories (methods x meshes)

worst loss = largest omega_1 deficit at the stop relative to the end state; ratio = k*/k_stag (1 = stops exactly at stagnation, <1 = before, >1 = after).

| rule | histories | never fired | worst omega_1 loss [%] | worst M_nd excess | k*/k_stag min / median / max |
|---|---|---|---|---|---|
| max\|dx\|<=0.01 | 23 | 7 | 0.03 | +0.005 | 1.06 / 2.39 / 7.62 |
| obj<5e-4 & Mnd<5e-3 W20 | 23 | 0 | 0.33 | +0.011 | 0.71 / 1.58 / 5.79 |
| obj<5e-4 & Mnd<5e-3 W10 | 23 | 0 | 0.72 | +0.010 | 0.51 / 1.37 / 3.88 |
| relL2<0.001 | 23 | 2 | 0.72 | +0.003 | 0.88 / 2.02 / 9.23 |
| rms<8.8e-4 (Olh c=.05) | 23 | 1 | 0.72 | +0.005 | 0.88 / 1.75 / 8.57 |
| obj<1e-3 & Mnd<5e-3 W10 | 23 | 0 | 0.91 | +0.010 | 0.48 / 1.25 / 2.68 |
| max\|dx\|<=0.02 W10 | 23 | 6 | 0.98 | +0.094 | 0.58 / 1.71 / 3.53 |
| obj rel W20 <1e-3 | 23 | 0 | 1.01 | +0.104 | 0.60 / 1.49 / 4.15 |
| obj rel W10 <5e-4 | 23 | 0 | 1.03 | +0.108 | 0.51 / 1.37 / 3.88 |
| max\|dx\|<=0.025 | 23 | 3 | 1.05 | +0.111 | 0.44 / 1.24 / 2.86 |
| obj rel W10 <1e-3 | 23 | 0 | 1.09 | +0.118 | 0.48 / 1.13 / 2.68 |
| obj<1e-3 & Mnd<1e-2 W10 | 23 | 0 | 1.09 | +0.118 | 0.48 / 1.25 / 2.68 |
| relL2<0.002 | 23 | 0 | 1.22 | +0.009 | 0.84 / 1.42 / 8.07 |
| max\|dx\|<=0.04 W10 | 23 | 4 | 1.31 | +0.134 | 0.45 / 1.33 / 2.06 |
| w1 rel W10 <1e-3 | 23 | 0 | 1.51 | +0.118 | 0.43 / 1.10 / 1.67 |
| Mnd abs W10 <5e-3 | 23 | 0 | 2.71 | +0.008 | 0.41 / 0.96 / 1.41 |
| flip<1e-3 W5 | 23 | 0 | 2.82 | +0.084 | 0.60 / 1.27 / 2.93 |
| flip<1e-3 | 23 | 0 | 3.29 | +0.102 | 0.36 / 0.94 / 1.57 |
| relL2<0.005 | 23 | 0 | 7.35 | +0.328 | 0.16 / 1.09 / 1.70 |
| rms<3.5e-3 (c=.2) | 23 | 0 | 7.35 | +0.328 | 0.16 / 1.09 / 1.67 |
| max\|dx\|<=0.04 | 23 | 0 | 9.27 | +0.405 | 0.08 / 1.14 / 1.79 |
| relL2<0.01 | 23 | 0 | 9.67 | +0.418 | 0.07 / 0.91 / 1.27 |


## Figures

* `fig/traces_proposed.png`, `fig/traces_yuksel.png`, `fig/traces_olhoff.png` — per-iteration max|Δx| (solver-native), ‖Δx‖₂/‖x‖₂, M_nd and ω₁ for every recorded mesh (Fig. 1–3).
* `fig/stop_rules_summary.png` — stop iteration and ω₁ deficit of the current and candidate rules per method and mesh, with k_stag.

## Files

`scripts/stopstudy_{proposed,yuksel,olhoff}.m` (recording runners; they only set existing configuration fields and run the production code paths), `scripts/stopstudy_metrics.m`, `scripts/replay.py` (rule replay and k_stag), `scripts/make_tables.py`, `scripts/plots.py`, `scripts/plot_summary.py`; `data/<method>_<mesh>.csv` (per-iteration metrics), `data/replay_all.csv`, `data/tables.md`.
