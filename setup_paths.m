%SETUP_PATHS  Put the solvers and configurations of this repository on the MATLAB path.
%   Run once per MATLAB session from any folder:  run('<repo>/setup_paths.m')
repoRoot_ = fileparts(mfilename('fullpath'));
addpath(fullfile(repoRoot_, 'configs'), fullfile(repoRoot_, 'solvers', 'common'), ...
    fullfile(repoRoot_, 'solvers', 'proposed'), fullfile(repoRoot_, 'solvers', 'yuksel'), ...
    fullfile(repoRoot_, 'solvers', 'du_olhoff'), fullfile(repoRoot_, 'paper'));
clear repoRoot_
% All results of the paper were computed single-threaded.  Multithreaded BLAS changes
% the order of floating-point sums; over tens of iterations this can move a design to a
% slightly different local optimum (e.g. 0.2% in omega_1 of the Du-Olhoff clamped beam).
maxNumCompThreads(1);
