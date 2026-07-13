# Topology optimization for frequency maximization

This repository contains code for three different local approaches to topology optimization for frequency maximization. All of them are based on SIMP, but they differ in how the fundamental frequency objective is approximated.

`analysis/OlhoffApproach` is a local Olhoff-inspired comparison implementation; it is not claimed to be a canonical or paper-faithful Du--Olhoff reconstruction. The second local implementation is inspired by the two-stage static approximation of Yuksel and Yilmaz (2025). The third is the proposed one-level quasi-static approximation.

The unsuccessful paper-reconstruction campaign under `analysis/OlhoffApproachExact*` is retained only as an archived diagnostic record. It is not production code and is not reviewer evidence. Revision experiments that retain an Olhoff comparison use `analysis/OlhoffApproach` only.

## JSON-driven MATLAB runner

You can run the optimization from a JSON task file using:

```matlab
addpath("examples");
[x, omega, tIter, nIter] = run_topopt_from_json("examples/case1.json");
```

Example JSON included in this repository:

```matlab
[x, omega, tIter, nIter] = run_topopt_from_json("examples/BeamTopOptFreq.json");
```

You can also pass an already decoded JSON struct:

```matlab
cfg = jsondecode(fileread("examples/BeamTopOptFreq.json"));
[x, omega, tIter, nIter] = run_topopt_from_json(cfg);
```
