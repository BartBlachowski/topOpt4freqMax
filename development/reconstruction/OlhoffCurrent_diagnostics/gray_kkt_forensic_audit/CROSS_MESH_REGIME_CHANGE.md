# Cross-mesh regime change

| mesh | Mnd % | gray % / area | mid % / area | broad core % / area | max depth / R |
| --- | --- | --- | --- | --- | --- |
| 400 | 15.3732 | 17.960% / 1.43680 | 3.920% / 0.31360 | 0.000% / 0.00000 | 0.05657 / 0.943 |
| 480 | 26.3416 | 28.729% / 2.29833 | 11.861% / 0.94889 | 13.014% / 1.04111 | 0.36667 / 6.111 |
| 800 | 34.4123 | 37.762% / 3.02100 | 16.090% / 1.28720 | 20.870% / 1.66960 | 0.38000 / 6.333 |

| mesh | gray RMS, all-interior dual | gray RMS, best gray dual | gray p95, best gray dual | broad RMS, best gray dual | filtered gray RMS, best gray dual |
| --- | --- | --- | --- | --- | --- |
| 400 | 0.555345 | 0.354769 | 0.781865 | empty | 0.0222383 |
| 480 | 0.444567 | 0.334039 | 0.754716 | 0.171565 | 0.0489931 |
| 800 | 1.16312 | 1.16254 | 1.69269 | 0.311465 | 1.15835 |

400→480 is qualitative: the domain acquires broad end-region gray patches rather than just more elements in a fixed-width diffuse boundary. Mid area triples (.3136→.94889); broad core grows from zero to 1.04111. At 800 broad core expands to 1.6696 and a connected gray network spans most of the beam. Maximum depth largely saturates after 480 even as area continues growing.

The transition is not a matching jump in cancellation strength, which weakens for the exact active branch. Raw-gray stationarity is already deficient at 400, so the nonstationarity metric does **not** uniquely identify why the topology changes at 480. Filtering sharply flattens gray gradients at both 400 and 480. At 800 the close first pair introduces a further local-optimality regime, seen in the full-tensor dual reconstruction. No claim of a proven topology bifurcation, filter-support artifact or mesh-convergence limit follows from three endpoints alone.


### Local-core qualification

A separate core-only multiplier fit gives raw broad-core normalized RMS **.06063 at 480** and **.18259 at 800**. At 480 the fitted raw derivative threshold 1.35315 closely matches the filtered gray threshold 1.35847; using that filtered-derived dual gives core RMS .06064. This is positive evidence of **approximate local balance inside much of the 480 broad patch**. It does not eliminate raw residuals in the surrounding gray field: all-gray RMS is .37060 with the same core-fitted multiplier. At 800 even the core-only fit remains above .1; using its filtered gray dual gives core RMS .37089.

Thus the primary nonstationarity category applies to the **complete constrained design and its substantial unresolved gray regions**, not every gray element. Some gray locations are locally balanced, and the 480 core could participate in a nearby stationary gray design. This audit cannot rule that out. The mixed-stationarity category is not issued as a certification of those patches because no common admissible physical multiplier makes the surrounding free design stationary. Local fits cannot be assigned independently to different parts of one volume-constrained problem. These results narrow the conclusion: “the whole endpoint is a genuine relaxed optimum” is unsupported; “every broad patch is itself necessarily nonstationary” is also unsupported.
