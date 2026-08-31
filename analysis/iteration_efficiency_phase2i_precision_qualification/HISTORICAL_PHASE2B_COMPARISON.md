# Historical Phase 2B comparison

Phase 2B remains a valid negative result under the old evaluator. Its discontinuous Eq. (4)
mass law changed branch when `0.099999999999999645` rounded to float32 above 0.1.
The measured maximum relative errors were E2 `0.0226523752185`
and E3 `0.0226523756441`; `b_ref` moved 2200→2100 and
`k_cert(q=.995)` moved 708→623.

Candidate C's continuous Eq. (4a) removes that spectral pathology: the new maxima are E2
`5.59553986329e-08` and E3
`5.59482294612e-08`, with identical modal selections,
`b_ref`, and persistence endpoints. This does not make Phase 2B erroneous.

The new qualification nevertheless fails for a different reason: float32-created cutoff
ties alter the exact-count binary topology at 95 states and flip the hard gate at k=41, 45,
48, and 99. Thus the rho=.1 *spectral* discontinuity is gone, while an exact-count topology
precision sensitivity remains.
