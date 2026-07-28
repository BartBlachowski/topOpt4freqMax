% run_mbb.m — MBB beam benchmark for TopIQP
%
% Paper Table 1 reference results (TopIQP column):
%   40x80,  V=0.3 -> 88 iterations
%   80x40,  V=0.2 -> 169 iterations
%
% Call from the Matlab command window (with this folder on the path):
%   run_mbb

clear; clc;

% Choose which LabandaApproach solver to run: 'IQP' (topIQP.m, IQP-only) or
% 'SQP' (topSQP.m, full IQP+EQP -- see topSQP.m docstring).
method = 'IQP';

% Paper reports meshes as nely x nelx.  topIQP/topSQP expect (nelx,nely).
% Table 5 uses rmin = 0.04*Lx; with unit square elements this is 0.04*nelx.

% --- Case 1: 40x80, V=0.3 (paper Table 1 row 5) ---
fprintf('=== MBB 40x80, volfrac=0.3 (method=%s) ===\n', method);
nelx1 = 80; nely1 = 40;
r1 = runLabandaSolver(method, nelx1, nely1, 0.3, 3, 0.04*nelx1, struct());
fprintf('Iterations: %d  (paper ref: 88)\n\n', r1.nIter);

figure('Name','MBB 40x80 V=0.3','Color','white');
imagesc(1 - r1.xPhys); colormap(gray); axis equal off;
title(sprintf('MBB 40\\times80,  V=0.3,  c=%.4g,  iter=%d', r1.compliance, r1.nIter));

% --- Case 2: 80x40, V=0.2 (paper Table 1 row 6) ---
fprintf('=== MBB 80x40, volfrac=0.2 (method=%s) ===\n', method);
nelx2 = 40; nely2 = 80;
r2 = runLabandaSolver(method, nelx2, nely2, 0.2, 3, 0.04*nelx2, struct());
fprintf('Iterations: %d  (paper ref: 169)\n\n', r2.nIter);

figure('Name','MBB 80x40 V=0.2','Color','white');
imagesc(1 - r2.xPhys); colormap(gray); axis equal off;
title(sprintf('MBB 80\\times40,  V=0.2,  c=%.4g,  iter=%d', r2.compliance, r2.nIter));

% % --- Case 1: 40x80, V=0.3 (paper Table 1 row 5) ---
% fprintf('=== MBB 80x160, volfrac=0.3 ===\n');
% nelx1 = 160; nely1 = 80;
% r1 = runLabandaSolver(method, nelx1, nely1, 0.3, 3, 0.04*nelx1, struct());
% fprintf('Iterations: %d  (paper ref: 88)\n\n', r1.nIter);

function r = runLabandaSolver(method, nelx, nely, volfrac, penal, rmin, opts)
%RUNLABANDASOLVER Dispatch to topIQP.m or topSQP.m by name.
    if strcmpi(method, 'SQP')
        r = topSQP(nelx, nely, volfrac, penal, rmin, opts);
    else
        r = topIQP(nelx, nely, volfrac, penal, rmin, opts);
    end
end
