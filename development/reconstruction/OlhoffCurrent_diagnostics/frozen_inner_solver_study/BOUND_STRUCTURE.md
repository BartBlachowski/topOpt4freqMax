# Exact active-bound structure

| Category | Count |
|---|---:|
| interior | 16 |
| lower density | 9404 |
| lower move | 4906 |
| upper move | 4788 |
| upper density | 9686 |
| coincident | 0 |


At tolerance 1e-6 times local box width, 99.944% of
variables are at some bound, while 33.660% are at
specifically ±move. Density-floor and density-ceiling bounds account for 19,090
variables. Thus the earlier ~33.7% move statistic and ~99.94% total saturation
measure different sets; neither is a contradictory count.

Coincidence uses equality of raw physical and move bounds to 1e-14 and has zero
members here. Interior has 16 members. Of 8,272 strict-gray elements, 12 are
interior at the declared tolerance; of 3,422 mid-gray elements, 6 are interior.
The smallest absolute gray oracle increment is 0.0009775965, so the task's
statement that EVERY gray increment is at full move is not literally supported
by the retained oracle. Its hash and certificate still pass. Threshold-dependent
activity is a description, never a substitute for KKT complementarity.
