# Evidence identity

GRAY_FORENSICS_EVIDENCE_PASS

| mesh | authority | iteration | endpoint status | rho SHA-256 | saved config SHA-256 | three-rung effective config SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| 400x50 | three_rung_architecture | 466 | CONVERGED_EXACT_S3_COUNTERFACTUAL | b3d388a576fe939cc701988797f036cbe0dac12581217f48916462577d1f8211 | 0afb0d4daa11b52aac8f184c856f71626bfd11b0a581e66d5aaacfa7f78f3a78 | cb907ad79a95c961740bc6d5d4ecd64855776ad8bd4528379ba018d939997008 |
| 480x60 | three_rung_canary_preflight | 386 | CONVERGED | 0a498a7d6ab0565b29c15ff9364060d937d4df10aa661fc90d02c038ce6e4a60 | 03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e | 03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e |
| 800x100 | three_rung_canary_preflight | 468 | CONVERGED | 50e4e625c330e1aadddf75ea121966213173b5817af7337d2ef3603e287bb9c7 | 7724af5ed26786e16132c30fbae840f5d8679d5ea798fc75d5c48eefd0e50ffe | 7724af5ed26786e16132c30fbae840f5d8679d5ea798fc75d5c48eefd0e50ffe |

The 400 endpoint is **RHO(:,466)** of the causal-controller four-rung parent, authenticated by the three_rung_architecture S3 density hash and exact-prefix proof. RHO(:,505) is explicitly excluded. 480 and 800 use the actual last trajectory columns, bitwise equal to the separate saved states and record hashes. No legacy beta endpoint is substituted.

All five input MAT containers match their expected SHA-256 and byte count. The three saved configs independently reproduce their recorded schema-ordered config hashes using the prior validated hash implementation. For 400 the effective three-rung hash changes only move.levels; it is labelled a counterfactual config, not the original stored config. Full configurations, multiplicity/MMA fields, trajectory paths, MATLAB metadata and controller status are in evaluations/identity.json and config_*.json.

All endpoints are stage 3 / move 0.01, persistent branch B under E=A OR B, stageExhaustion for move/stop, beta without controller authority. Shared p=3, q=1, eq4b mass, sensitivity/all filter, physical R=0.06, fixed N=2 subspace with offsets/offdiagonals, published increment MMA, min/max inner=5/500, relative tolerance=0.05. Element radii 3/3.6/6. No projected density or continuation.
