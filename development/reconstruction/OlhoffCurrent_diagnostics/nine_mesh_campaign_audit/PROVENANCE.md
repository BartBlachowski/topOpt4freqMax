# Provenance and campaign identity

The intended promoted three-rung campaign is absent. The most recent nine-mesh campaign is `examples/Performance/conference_benchmark/campaign_9mesh_r2`, generated 2026-09-11 at 23:51:04+02:00 (results JSON 23:51:05). It contains 27 observations, nine each for three methods; this audit selects `method_key=olhoff`. Other similarly named directories contain earlier campaigns, including a different fixed-work stabilization formulation; they are not pooled.

Initial branch `benchmark-methodology-r2`; HEAD `013cc48451d33bed61c5c4eea174bbd898d548a2`; initial tracked and untracked status clean. Audit changes are confined to this directory. No AGENTS.md was found. The user-request attachment and repository evidence policy were read; no optimization-capable tests were run.

## Verified source identity

All **21/21** manifest source hashes match the current files. All **75/75** production implementation files match `SOURCE_MANIFEST.json`, with no extra source files. Independently reconstructed `+impl` tree hash:

`edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb`

Both dirty campaign driver files are now the exact HEAD versions. The campaign recorded parent commit `bba45e72ea18eca7615315fcc543572d42a436bc-dirty`. The subsequently committed changes add common record fields and a warm-up record-schema check; inspection finds no controller promotion in that diff. A git subject such as “Nine test passed” is not scientific evidence.

`olhoffcurrent_config.m` SHA-256: `16b431039ec47a9131a5b9365d1e4e28628507cc88e986684bf982c28e462b50`. The effective per-mesh hashes were independently recreated using the 81-row schema and MATLAB-compatible value formatting. **9/9 match**. For C320, production is `2a5b500991ff5931eef4919e768503d41fd558fe56ffbffe40c382c74d676cad`; the validated three-rung candidate is `afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab`. Config hashes correctly differ across meshes because mesh and derived tolerance belong in them.

## Exact effective formulation

The archive stores canonical configs in `records.effective_config`; the manifest independently stores their canonical form. They agree field by field excluding provenance metadata. [effective_configs.json](effective_configs.json) and `config_<mesh>.txt` retain every value.

Domain 8×1, thickness 1, Q4 plane stress, consistent mass; mid-height simple supports with both ends axially restrained (four constrained DOFs). E=10^7, nu=0.3, solid density=1; uniform initial density 0.5, minimum 0.001, volume fraction 0.5. SIMP p=3 fixed; mass eq4b, q=1, low-density exponent 6, cutoff 0.1. Sensitivity filter on all generalized gradients, physical R=0.06; no projection or material continuation. Fixed subspace size 2, diagonal offsets retained and off-diagonal gradients active. Eigensolver `eigs`, tolerance 10^-12, cap 5000, deterministic sine-based start vector. Published MMA on increments, reset for each outer solve, tolerance 0.05, min/max inner 5/500.

**Actual ladder [0.04,0.02,0.01,0.005]; continuation `boundVariable` (legacy alias beta), window 10, tolerance 0.005; stop `designChange`, L2, tolerance 0.05 sqrt(NE/3200), settledMove guard only; cap 400.** No stageExhaustion authority. No runtime fallback occurred: legacy is the configured default.

All nine share this formulation. Only mesh dimensions, derived L2 tolerance, runtime name and provenance metadata differ; physical radius and scientific settings do not. There are no mesh-specific cap or solver overrides. The other method's Yuksel cap override is not an Olhoff change.

## Machine, time and retention

Campaign host `PMS.local`, MATLAB `25.2.0.2998904 (R2025b)`, MACA64, Apple Accelerate ILP64 BLAS, requested and reported computation threads=1. This is one sequential driver invocation with a discarded 48×6, five-outer warm-up. CPU model, RAM capacity, macOS version, competing-load record, thermal state, and peak memory were **not recorded**. Current machine values would not prove campaign conditions, so none are substituted.

Manifest configuration-resolution times are around 20:12; those are not exact job starts. Per-run canonical `resolvedAt` values are retained in effective_configs.json; they locate pre-solve configuration resolution, not a complete start/end timing log. Export at 23:51:04–05 is known; exact solver start/end times are unavailable. Sum of the nine Olhoff solver times is 2.6681 h. Model preparation and final modal evaluation are in solver wall time; config/path setup, export and common evaluator are outside it.

The driver has no load/resume route for these records. No resume is declared, and nothing in the stored terminal logs indicates one. This does not prove the absence of discarded earlier attempts. The run-time files were dirty but are hash-identified. No append-only execution log or original output checksum seal exists. JSON/MAT agreement is internal consistency, not independent execution authentication.

Existing topology/export ancillary files include older September 6 timestamps beside September 11 results; they cannot establish current topology identity. All audit plots were rebuilt from the MAT density vectors. Original output files may have been overwritten by the campaign; subsequent regeneration cannot be fully excluded. We do not repair or rewrite any source or historical artifact.

## Promotion history and retained historical defects

The pre-campaign `three_rung_promotion_closure/REPORT.md` explicitly says promotion was blocked and production remained legacy. The September 11 “ready” commit added reports and validation artifacts, but changed no production policy. Campaign `manifest_role` expressly says OUTPUT, not preregistration. No prospective nine-mesh three-rung preregistration or successful production-promotion seal was found.

Historical four-rung C160/C240/C320/C400 trajectories are present and match their own current EVIDENCE declarations; independent A/B replay succeeds. The separate C320 three-rung candidate MAT is missing. Its 352-row CSV and scalar record remain hash-valid, and all 52 scientific CSV columns match the retained C320 oracle prefix (timing and two declared shadow columns excluded). A historical cross-study baseline digest is stale; the candidate's optional old C320 container digest differs from the current original-oracle declaration. P400/F400 raw containers differ from their declarations. See [historical_evidence_checks.json](historical_evidence_checks.json) and [historical_seal_checks.json](historical_seal_checks.json). These defects narrow claims; they do not justify deleting the surviving positive evidence.

Reproduction of this audit: run `scripts/audit.py`, `scripts/supplement.py`, then `scripts/write_reports.py` with repository `.venv/bin/python`. These are read-only postprocessors of source data. Run `scripts/seal.py` last. No MATLAB invocation is needed.
