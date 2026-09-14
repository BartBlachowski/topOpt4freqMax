# Olhoff variant selector test

Verdict: **PASS**.

| selector | LP executed | MMA executed | result identity |
|---|---:|---:|---|
| `lp` | yes | no | `Olhoff-LP`, principal |
| `mma` | no | yes | `Olhoff-MMA`, secondary paper-native |
| `both` | yes | yes | two independent records/directories |

The MMA smoke used full off-diagonal coupling, the post-call effective physical
filter-radius rule, and no LP fallback. Its three outer updates required
70/46/63 inner MMA iterations; all converged, none hit the 300-iteration cap,
and recorded LP calls were zero. LP recorded three outer updates, three LP calls,
zero failed calls, and three genuine backend iterations exposed by HiGHS.

The LP/MMA comparison remains explicitly paper-native, not move-controlled:
LP uses the accepted approximately .005 S1 route; MMA uses the audited .010 move.
