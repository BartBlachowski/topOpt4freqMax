# Phase 2B — Olhoff representation-fidelity qualification

This isolated directory contains precision-qualification evidence only. It does not contain
production scientific results and must not be used in scaling or method-comparison tables.

Genuine pairs are obtained by deterministic prefix reruns of the unmodified source. A run
whose cap is `k` returns accepted state `k` as double in `res.rho` and simultaneously returns
the source logging cast in `res.rho_snapshots(:,end)`. Each pair is accepted only when its
single state and every non-timing history field match the same prefix of a longer baseline.
