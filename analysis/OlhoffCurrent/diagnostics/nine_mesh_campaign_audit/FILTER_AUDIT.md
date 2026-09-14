# Filter and physical length scale

R/b=0.06 is verified in every canonical config, actual MAT config, and solver radius-conversion source. With b=1 and square elements h=1/nely, rminEl=R/h gives the intended 1.2,1.8,2.4,3.0,3.6,4.2,4.8,5.4,6.0. A manifest legacy-view `rminEl=null` is not a missing filter: nonempty physical radius takes precedence and is converted inside the solver.

`prepFilter.m` uses cone weights max(0,rminEl−distance) with row-sum normalization and boundary truncation. Positive interior support includes:

| Mesh | rminEl | Positive stencil entries | Center fraction | Weighted RMS radius/b |
| --- | --- | --- | --- | --- |
| 160x20 | 1.2 | 5 | 0.6 | 0.0316228 |
| 240x30 | 1.8 | 9 | 0.275097 | 0.0326725 |
| 320x40 | 2.4 | 21 | 0.161566 | 0.0333557 |
| 400x50 | 3 | 25 | 0.106606 | 0.0321536 |
| 480x60 | 3.6 | 37 | 0.0734055 | 0.0327489 |
| 560x70 | 4.2 | 57 | 0.0542542 | 0.0326364 |
| 640x80 | 4.8 | 69 | 0.0412138 | 0.0329369 |
| 720x90 | 5.4 | 97 | 0.0326892 | 0.032872 |
| 800x100 | 6 | 109 | 0.0265349 | 0.0327886 |


The continuum cone's weighted RMS radius is sqrt(3/10)R≈0.03286b. The discrete values cluster near this, including the fine 720→800 step (0.032872→0.032789), while that step loses 3.64% of omega1 and 0.207 of adjacent IoU relative to the preceding pair. There is no comparably large discrete length-scale jump there. At 160, rminEl<sqrt(2) excludes diagonal neighbours: five positive weights and a 60% center weight make it special. The 160→240 stencil expands to nine entries and center weight falls to 27.5%; the topology and gap also change. That is an association, not isolated causality.

Every mesh changes discrete support, so merely matching a kink to a stencil change is weak evidence. The largest grayness jumps also align with premature stopping regimes (320's terminal stage changes; 800 stops unusually early). The evidence ranks the proven policy/stopping defect above an alleged physical-radius inconsistency. A sensitivity filter does not itself guarantee a minimum feature size or binary morphology. No filter change or new controlled experiment is recommended from these data alone.
