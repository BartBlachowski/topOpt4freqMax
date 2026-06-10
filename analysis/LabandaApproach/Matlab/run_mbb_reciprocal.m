% run_mbb_reciprocal.m - MBB beam benchmark for the reciprocal optimizer
%
% This is not the Labanda TopIQP reference run.  It exercises the
% experimental reciprocal-variable scalar-QP path in topReciprocalSQP.m.

clear; clc;

opts = struct();
opts.maxIter = 500;
opts.move = 0.10;
opts.displayEvery = 25;
opts.tolRelObj = 2e-5;
opts.tolChange = 1e-3;

fprintf('=== Reciprocal MBB 40x80, volfrac=0.3 ===\n');
nelx1 = 160; nely1 = 80;
tic;
r1 = topReciprocalSQP(nelx1, nely1, 0.3, 3, 0.04*nelx1, opts);
t1 = toc;
fprintf('Iterations: %d, compliance: %.8g, time: %.3f s\n\n', ...
        r1.nIter, r1.compliance, t1);

figure('Name','Reciprocal MBB 40x80 V=0.3','Color','white');
imagesc(1 - r1.xPhys); colormap(gray); axis equal off;
title(sprintf('Reciprocal MBB 40\\times80, V=0.3, c=%.4g, iter=%d', ...
      r1.compliance, r1.nIter));

fprintf('=== Reciprocal MBB 80x40, volfrac=0.2 ===\n');
nelx2 = 40; nely2 = 80;
tic;
r2 = topReciprocalSQP(nelx2, nely2, 0.2, 3, 0.04*nelx2, opts);
t2 = toc;
fprintf('Iterations: %d, compliance: %.8g, time: %.3f s\n\n', ...
        r2.nIter, r2.compliance, t2);

figure('Name','Reciprocal MBB 80x40 V=0.2','Color','white');
imagesc(1 - r2.xPhys); colormap(gray); axis equal off;
title(sprintf('Reciprocal MBB 80\\times40, V=0.2, c=%.4g, iter=%d', ...
      r2.compliance, r2.nIter));
