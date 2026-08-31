# Precision requalification plan

Status: **required; not executed**.

Compare lossless double trajectories with every proposed stored precision on shared
states for all meshes, methods, E1/E2/E3 modes, diagnostic values, selected ordinals,
selected frequencies, Q ratios, hard-gate decisions, `k_enter`, `k_cert`, and reference
freeze. Include Olhoff LP and, if selected, nested MMA independently. Acceptance requires
identical discrete decisions and ordinals plus a preregistered numeric error bound.

The pass artifact must use schema `candidate_c_precision_qualification_v1` and bind the
Candidate C classifier version, evaluator SHA-256, frozen contract SHA-256, scope, route,
input provenance hashes, exact acceptance limits, results, and `pass=true`. Prior
omega1/binary or evaluator-version evidence cannot qualify Candidate C.

