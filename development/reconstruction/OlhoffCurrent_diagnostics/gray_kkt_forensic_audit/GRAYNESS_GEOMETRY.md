# Grayness geometry

| mesh | Mnd % | gray % / area | mid % / area | broad core % / area | max depth / R |
| --- | --- | --- | --- | --- | --- |
| 400 | 15.3732 | 17.960% / 1.43680 | 3.920% / 0.31360 | 0.000% / 0.00000 | 0.05657 / 0.943 |
| 480 | 26.3416 | 28.729% / 2.29833 | 11.861% / 0.94889 | 13.014% / 1.04111 | 0.36667 / 6.111 |
| 800 | 34.4123 | 37.762% / 3.02100 | 16.090% / 1.28720 | 20.870% / 1.66960 | 0.38000 / 6.333 |

Physical area units are domain-coordinate units squared; full area=8. Gray is 0.1<rho<0.9; mid is 0.4<=rho<=0.6. Broad core means gray distance to the nearest non-gray element centre exceeds R=0.06. It is a conservative core measure, not the area of every entire component that contains a core. Domain edges are not treated as density interfaces. Distances have finite-grid centre uncertainty of order h. A separate rho=0.5 interface distance distribution is retained; there is no direct solid/void contact through a diffuse band without defining a threshold.

400: no gray core deeper than R; the gray zones are resolved interface/member bands. 480: two leading gray components have areas 0.79444 and 0.79056, each bounding box 1.91667×0.76667. 800: one leading connected gray network has area 2.8676 and spans 7.8×0.76; this connectivity includes thin bridges and does not mean the whole box is gray. Core maps and snapshots identify broad end patches. The maximum-depth diameter proxies 2d are 0.11314,0.73333,0.76; these are not exact member widths. The transition is primarily 400→480, with only a small further increase in maximum depth at 800 but a large increase in broad area.

Every component size/bounding box/depth, four/eight-neighbour component count, density quantiles, fractions below .01 and above .99, gray depth and threshold-interface distances is in evaluations/geometry.json. F01–F04 use identical physical coordinates; histograms use identical bins.
