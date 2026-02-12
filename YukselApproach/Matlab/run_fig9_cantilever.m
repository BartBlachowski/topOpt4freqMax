%% Yuksel & Yilmaz (2025) - Figure 9 benchmark (cantilever with tip mass)
% Reproduces the setup reported in Figure 9:
% E0 = 1e7 Pa, rho0 = 1 kg/m^3, nu = 0.3, L:H = 15:10, volfrac = 0.5, mesh = 150x100.
% The concentrated mass at right-edge mid-height is handled internally in top99neo_inertial_freq.
clear; clc; close all;

addpath(fileparts(mfilename('fullpath')));

% Figure 9 geometry/mesh (L = 15, H = 10 -> nelx:nely = 150:100)
nelx = 150;
nely = 100;

% Optimization setup (paper-consistent)
volfrac = 0.5;
penal = 3;
rmin = 2.3;
ft = 1;           % sensitivity filter only
ftBC = 'N';       % symmetric filter boundary
eta = 0.5;        % not used when ft = 1
beta = 1;         % not used when ft = 1
move = 0.2;
maxit = 200;      % upper bound; solver also stops at change < 0.01
stage1_maxit = 200;
bcType = "cantilever";
plotDynamicHistory = false; % set true to also run/plot Section-3 dynamic-code history
dynMove = 0.01;             % dynamic-code move limit (paper dynamic setting)
dynMaxIt = 200;

[xPhysStage2, ~, info] = top99neo_inertial_freq( ...
    nelx, nely, volfrac, penal, rmin, ft, ftBC, eta, beta, move, maxit, stage1_maxit, bcType, 3);

figure('Name','Yuksel Figure 9 benchmark','Color','w');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile;
imagesc(1 - reshape(info.stage1.xFinal, nely, nelx));
axis equal off;
caxis([0 1]);
colormap(gray);
title(sprintf('Figure 9(b): \\omega_1 = %.1f rad/s', info.stage1.omega1), 'Interpreter', 'tex');

nexttile;
imagesc(1 - reshape(xPhysStage2, nely, nelx));
axis equal off;
caxis([0 1]);
colormap(gray);
title(sprintf('Figure 9(c): \\omega_1 = %.1f rad/s', info.stage2.omega1), 'Interpreter', 'tex');


fprintf('\nFigure 9 material constants used internally by top99neo_inertial_freq:\n');
fprintf('  E0 = 1e7 Pa, nu = 0.3, rho0 = 1 kg/m^3\n');
fprintf('  Emin = 1e-9*E0, rho_min = 1e-9*rho0, d = 6, x_cut = 0.1\n');
fprintf('  Concentrated mass (internal): 20%% of permitted material mass at right mid-edge\n');
fprintf('  Stage 1 frequency: omega1 = %.4f rad/s (paper: 94.1)\n', info.stage1.omega1);
fprintf('  Stage 2 frequency: omega1 = %.4f rad/s (paper: 101.5)\n', info.stage2.omega1);

