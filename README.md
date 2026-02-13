# Topology optimization for frequency maximization

This repository contains code for three diffrent approaches to topology optimization for frequency maximization. All of them are based on SIMP method, but differ in selected way for fundamental frequency maximization.

The first approach uses classical double-loop approach proposed by Du and Olhoff (2007). The second one uses two level static approximation proposed by Yuksel and Yilmaz (2025). Finally, the third approach is newly proposed one-level quasi-static approximation.

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
