% run_cantilever.m — Cantilever benchmark for TopIQP
%
% Paper Table 1 reference result (TopIQP column):
%   120x60, V=0.5 -> 81 iterations (TopIQP) / 78 iterations (TopSQP)
%
% Boundary conditions:
%   Left edge fully clamped (ux=uy=0)
%   Point load fy=-1 at mid-right node
%
% Call from the Matlab command window (with this folder on the path):
%   run_cantilever

clear; clc;

% Choose which LabandaApproach solver to run: 'IQP' (topIQP.m, IQP-only) or
% 'SQP' (topSQP.m, full IQP+EQP -- see topSQP.m docstring).
method = 'IQP';

fprintf('=== Cantilever 120x60, volfrac=0.5 (method=%s) ===\n', method);

% Paper reports meshes as nely x nelx.  topIQP/topSQP expect (nelx,nely).
% Table 5 uses rmin = 0.04*Lx; with unit square elements this is 0.04*nelx.
nelx = 240; nely = 120;
opts = struct('bcType', 'cantilever');
if strcmpi(method, 'SQP')
    r = topSQP(nelx, nely, 0.5, 3, 0.04*nelx, opts);
else
    r = topIQP(nelx, nely, 0.5, 3, 0.04*nelx, opts);
end
fprintf('Iterations: %d  (paper ref: 81 TopIQP / 78 TopSQP)\n\n', r.nIter);

figure('Name','Cantilever 120x60 V=0.5','Color','white');
imagesc(1 - r.xPhys); colormap(gray); axis equal off;
title(sprintf('Cantilever 120\\times60,  V=0.5,  c=%.4g,  iter=%d', r.compliance, r.nIter));
