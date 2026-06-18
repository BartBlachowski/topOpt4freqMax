# CC square-element mesh resolution study

Configuration: audit-baseline `run_cc_meshcompare_b.m`; only `nelx,nely` changed. Domain remains L=8, H=1. `rmin_elem=2.5` is held as the existing solver setting.

## Measurements

### Mesh 80x10

- Runtime: 22.4 s; outer iterations: 80
- Initial omega: 145.968, 364.269, 622.685
- Best design: iter 43, omega1=325.432, omega2=340.434, N=1, volume=0.4998
- Final design: omega1=292.481, omega2=384.353, N=1, volume=0.4999
- Morphology: disconnected blocks; raw components 4/8=5/5; effective 8-components=2 (ignore islands <10 elems); ccSolid=0.033; central density=0.209; diagonal braces=True (score=0.680)
- Fig. 3c similarity: low (0.359)

### Mesh 160x20

- Runtime: 50.8 s; outer iterations: 80
- Initial omega: 145.569, 363.049, 622.558
- Best design: iter 34, omega1=329.695, omega2=420.396, N=1, volume=0.4998
- Final design: omega1=327.041, omega2=432.049, N=1, volume=0.4998
- Morphology: connected cross-braced truss; raw components 4/8=5/3; effective 8-components=1 (ignore islands <16 elems); ccSolid=0.194; central density=0.322; diagonal braces=True (score=0.650)
- Fig. 3c similarity: high (0.875)

### Mesh 320x40

- Runtime: 184.9 s; outer iterations: 80
- Initial omega: 145.458, 362.715, 622.513
- Best design: iter 54, omega1=386.738, omega2=459.371, N=1, volume=0.4999
- Final design: omega1=383.412, omega2=455.548, N=1, volume=0.4999
- Morphology: connected cross-braced truss; raw components 4/8=11/5; effective 8-components=1 (ignore islands <64 elems); ccSolid=0.249; central density=0.300; diagonal braces=True (score=0.689)
- Fig. 3c similarity: high (0.891)

## Convergence table

| mesh | connected? | ccSolid | omega1best | omega1final | Nfinal |
|---|---:|---:|---:|---:|---:|
| 80x10 | no | 0.033 | 325.432 | 292.481 | 1 |
| 160x20 | yes | 0.194 | 329.695 | 327.041 | 1 |
| 320x40 | yes | 0.249 | 386.738 | 383.412 | 1 |
