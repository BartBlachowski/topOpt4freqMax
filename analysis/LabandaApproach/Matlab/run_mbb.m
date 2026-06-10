% run_mbb.m — MBB beam benchmark for TopIQP
%
% Paper Table 1 reference results (TopIQP column):
%   40x80,  V=0.3 -> 88 iterations
%   80x40,  V=0.2 -> 169 iterations
%
% Call from the Matlab command window (with this folder on the path):
%   run_mbb

clear; clc;

% Paper reports meshes as nely x nelx.  topIQP expects (nelx,nely).
% Table 5 uses rmin = 0.04*Lx; with unit square elements this is 0.04*nelx.

% --- Case 1: 40x80, V=0.3 (paper Table 1 row 5) ---
fprintf('=== MBB 40x80, volfrac=0.3 ===\n');
nelx1 = 80; nely1 = 40;
r1 = topIQP(nelx1, nely1, 0.3, 3, 0.04*nelx1);
fprintf('Iterations: %d  (paper ref: 88)\n\n', r1.nIter);

figure('Name','MBB 40x80 V=0.3','Color','white');
imagesc(1 - r1.xPhys); colormap(gray); axis equal off;
title(sprintf('MBB 40\\times80,  V=0.3,  c=%.4g,  iter=%d', r1.compliance, r1.nIter));

% --- Case 2: 80x40, V=0.2 (paper Table 1 row 6) ---
fprintf('=== MBB 80x40, volfrac=0.2 ===\n');
nelx2 = 40; nely2 = 80;
r2 = topIQP(nelx2, nely2, 0.2, 3, 0.04*nelx2);
fprintf('Iterations: %d  (paper ref: 169)\n\n', r2.nIter);

figure('Name','MBB 80x40 V=0.2','Color','white');
imagesc(1 - r2.xPhys); colormap(gray); axis equal off;
title(sprintf('MBB 80\\times40,  V=0.2,  c=%.4g,  iter=%d', r2.compliance, r2.nIter));

% % --- Case 1: 40x80, V=0.3 (paper Table 1 row 5) ---
% fprintf('=== MBB 80x160, volfrac=0.3 ===\n');
% nelx1 = 160; nely1 = 80;
% r1 = topIQP(nelx1, nely1, 0.3, 3, 0.04*nelx1);
% fprintf('Iterations: %d  (paper ref: 88)\n\n', r1.nIter);
