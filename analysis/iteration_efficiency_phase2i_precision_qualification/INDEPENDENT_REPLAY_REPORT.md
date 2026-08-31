# Independent replay report

The independent path uses Python, SciPy `eigsh`, and a separately implemented exact-count
topology/reference/persistence engine. It replays 84 representative
single/double spectra covering ordinary, rho≈0.1-heavy, high-ordinal, near-q, endpoint,
and all hard-gate-mismatch states.

- Selected-ordinal mismatches versus MATLAB: **0**.
- Maximum Python/MATLAB selected-frequency relative difference:
  **7.898e-12**.
- Full 3,200-state hard gates match MATLAB for both representations.
- Independently reproduced hard-gate mismatch states: **[41, 45, 48, 99]**.
- Independently reproduced `b_ref`: **2100 / 2100**.
- P=50/100/200 endpoints reproduce exactly.

Independent replay result: **PASS**, supporting the negative qualification conclusion.
