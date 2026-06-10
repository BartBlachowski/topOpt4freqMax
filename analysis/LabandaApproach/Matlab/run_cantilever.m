% run_cantilever.m — Cantilever benchmark for TopIQP
%
% Paper Table 1 reference result (TopIQP column):
%   120x60, V=0.5 -> 81 iterations
%
% Boundary conditions:
%   Left edge fully clamped (ux=uy=0)
%   Point load fy=-1 at mid-right node

clear; clc;

fprintf('=== Cantilever 120x60, volfrac=0.5 ===\n');

% topIQP uses MBB BCs by default; cantilever needs different BCs.
% This script calls the extended topIQP_bc that accepts a bcType argument.
% For now, patch by modifying the call — see note below.

% NOTE: topIQP.m currently hardcodes MBB boundary conditions.
% To run the cantilever, either:
%   (a) add a bcType parameter to topIQP (recommended next step), or
%   (b) duplicate topIQP.m with cantilever BCs.
%
% The MBB run_mbb.m validates the algorithm first.  Extend BCs after that.

warning('Cantilever BCs not yet wired into topIQP.m. Run run_mbb.m first.');
fprintf('  -> Add bcType parameter to topIQP.m, then re-run this script.\n');
fprintf('     Expected: 81 iterations (TopIQP) / 78 iterations (TopSQP)\n');
