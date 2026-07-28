% run_simply_simply.m — Simply-simply beam benchmark for TopIQP
%
% Geometry and boundary conditions are ported from the Olhoff-inspired
% frequency-maximization example:
%   analysis/OlhoffApproach/Matlab/run_simply_simply.m
%   analysis/OlhoffApproach/Matlab/topFreqOptimization_MMA.m (buildSupports)
%
% That source problem is an unloaded eigenvalue problem (L=8, H=1, both
% edges pinned at mid-height). The source script's active mesh is 800x100
% (with a 240x30 fallback commented out); this script uses 240x30, since
% TopIQP's dual-QP IQP subproblem (solved with quadprog every outer
% iteration) does not scale to an 800x100 mesh. TopIQP minimises
% compliance, so there is no load to reuse; a top-edge, mid-span point
% load fy=-1 (three-point bending) is added here — see topIQP.m
% bcType='ss'.
%
% Call from the Matlab command window (with this folder on the path):
%   run_simply_simply

clear; clc;

% Choose which LabandaApproach solver to run: 'IQP' (topIQP.m, IQP-only) or
% 'SQP' (topSQP.m, full IQP+EQP -- see topSQP.m docstring).
method = 'IQP';

fprintf('=== Simply-Simply 240x30, volfrac=0.5 (method=%s) ===\n', method);

% L=8, H=1 in the Olhoff source; topIQP/topSQP work in unit-element-size
% mesh coordinates, so only the aspect ratio (nelx:nely = 240:30 = 8:1)
% carries over. rmin=2 element-widths matches the source's rmin = 2*dx.
nelx = 240; nely = 30;
opts = struct('bcType', 'ss');
if strcmpi(method, 'SQP')
    r = topSQP(nelx, nely, 0.5, 3, 2, opts);
else
    r = topIQP(nelx, nely, 0.5, 3, 2, opts);
end
fprintf('Iterations: %d,  compliance: %.6g\n\n', r.nIter, r.compliance);

figure('Name','Simply-Simply 240x30 V=0.5','Color','white');
imagesc(1 - r.xPhys); colormap(gray); axis equal off;
title(sprintf('Simply-Simply 240\\times30,  V=0.5,  c=%.4g,  iter=%d', r.compliance, r.nIter));
