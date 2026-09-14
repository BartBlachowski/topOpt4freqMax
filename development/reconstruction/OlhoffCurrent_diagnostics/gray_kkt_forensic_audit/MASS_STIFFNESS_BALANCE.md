# Mass versus stiffness balance

| mesh | class | median |gK| | median |gM| | median |net| | median C | C<0.1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 400 | gray | 1.5855 | 1.42276 | 1.2656 | 0.398052 | 11.1359 |
| 400 | mid | 1.44162 | 1.38454 | 0.676772 | 0.24169 | 20.4082 |
| 400 | solid | 7.16806 | 2.6456 | 4.4765 | 0.536207 | 0.294551 |
| 400 | void | 3.92371e-05 | 3.56055e-07 | 3.88075e-05 | 0.981454 | 0.677966 |
| 480 | gray | 1.79185 | 0.78241 | 1.26514 | 0.490345 | 6.42978 |
| 480 | mid | 1.80403 | 0.551178 | 1.29002 | 0.541044 | 5.09368 |
| 480 | solid | 5.30959 | 2.00163 | 3.30314 | 0.518782 | 0.272851 |
| 480 | void | 3.58644e-05 | 4.07538e-07 | 3.52926e-05 | 0.972763 | 0.681995 |
| 480 | broad | 1.88747 | 0.590242 | 1.34594 | 0.540117 | 0 |
| 800 | gray | 0.458048 | 0.0487329 | 0.252676 | 0.959786 | 7.3949 |
| 800 | mid | 0.515285 | 0.0940167 | 0.199894 | 0.919072 | 9.05842 |
| 800 | solid | 0.948603 | 0.196847 | 0.759794 | 0.826284 | 4.71468 |
| 800 | void | 0.00104883 | 4.34989e-08 | 0.00107614 | 0.999698 | 0.508614 |
| 800 | broad | 0.426158 | 0.0598894 | 0.207111 | 0.733963 | 8.54097 |

C=|gK+gM|/(|gK|+|gM|+machine epsilon), with gM signed. Small C indicates cancellation; large C does not. Medians of individual terms cannot be subtracted to recover the median net.

There is **no measured strengthening of cancellation** as the gray area grows. Gray median C increases from .398 to .490 to .960 for the exact active branch. This rejects the proposed explanation based on a mesh-growing population of almost-zero net spectral derivatives. The 800 active mode is strongly mode-dependent: an exploratory PSD trace-one mixed dual gives raw gray median C=.469 (filtered-fit weights give .464). That also does not establish increasing cancellation relative to 400; it prevents overinterpreting .960 as an invariant property of the whole subspace.

Constrained stationarity requires balance against a **volume multiplier**, not cancellation of stiffness and mass to zero. Gray rho>.1 lies on the linear mass branch; the special low-density polynomial acts outside the gray class. This audit does not establish that changing the mass law or p would improve scientific validity. No such perturbation was made.
