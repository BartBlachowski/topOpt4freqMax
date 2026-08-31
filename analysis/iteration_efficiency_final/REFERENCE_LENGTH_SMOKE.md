# Reference-length smoke

Verdict: **PASS**.

The final workflow replayed the lossless 96x12 Olhoff-LP trajectory containing
state 0 plus 3,200 post-updates. Candidate-C and hard-gate evidence reproduced:

- trajectory dtype: `double`;
- reference status: PASS;
- `b_ref = 2100`;
- `Q_ref = [162.6600903758, 162.9977569655, 162.9977566494]`;
- `B0 = 3200`, `B_meas = 3200`, tail not truncated.

Primary P=100 endpoints were:

| q | k_enter | k_cert |
|---:|---:|---:|
| .98 | 229 | 328 |
| .99 | 309 | 408 |
| .995 | 453 | 552 |

P=50 and P=200 sensitivities also reproduced all nine expected endpoint pairs.
No cap fallback or post-horizon information was used.
