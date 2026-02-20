clear; clc;
close all;

jsonPath = 'BeamTopOptFreq.json';
data = jsondecode(fileread(jsonPath));

data.optimisation.approach = 'Olhoff';
[x1, omega1, tIter1, nIter1, mem1] = run_topopt_from_json(data);

data.optimisation.approach = 'Yuksel';
[x2, omega2, tIter2, nIter2, mem2] = run_topopt_from_json(data);

data.optimisation.approach = 'OurApproach';
[x3, omega3, tIter3, nIter3, mem3] = run_topopt_from_json(data);

mem1
mem2
mem3