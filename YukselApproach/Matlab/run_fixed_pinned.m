%% Yuksel & Yilmaz (2025) - Figure 8 benchmark (fixed-pinned beam)
% Reproduces the setup reported in Figure 8:
% E0 = 1e7 Pa, rho0 = 1 kg/m^3, nu = 0.3, L:H = 8:1, volfrac = 0.5, mesh = 320x40.
% The solver uses sensitivity filtering and the OC move limit 0.2.
clear; clc; close all;

addpath(fileparts(mfilename('fullpath')));

% Figure 8 geometry/mesh (L = 8, H = 1 -> nelx:nely = 320:40)
nelx = 320;
nely = 40;

% Optimization setup (paper-consistent)
volfrac = 0.5;
penal = 3;
rmin = 2.0;
ft = 1;           % sensitivity filter only
ftBC = 'N';       % symmetric filter boundary
eta = 0.5;        % not used when ft = 1
beta = 1;         % not used when ft = 1
move = 0.2;
maxit = 400;      % tighter stage-2 tolerance is used for fixed-pinned; allow more iterations
stage1_maxit = 200;
bcType = "fixedPinned";
plotDynamicHistory = false; % set true to also run/plot Section-3 dynamic-code history
dynMove = 0.01;             % dynamic-code move limit (paper dynamic setting)
dynMaxIt = 200;

[xPhysStage2, ~, info] = top99neo_inertial_freq( ...
    nelx, nely, volfrac, penal, rmin, ft, ftBC, eta, beta, move, maxit, stage1_maxit, bcType, 3);

figure('Name','Yuksel Figure 8 benchmark','Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

nexttile;
imagesc(1 - reshape(info.stage1.xFinal, nely, nelx));
axis equal off;
caxis([0 1]);
colormap(gray);
title(sprintf('Figure 8(b): \\omega_1 = %.1f rad/s', info.stage1.omega1), 'Interpreter', 'tex');

nexttile;
imagesc(1 - reshape(xPhysStage2, nely, nelx));
axis equal off;
caxis([0 1]);
colormap(gray);
title(sprintf('Figure 8(c): \\omega_1 = %.1f rad/s', info.stage2.omega1), 'Interpreter', 'tex');

fprintf('\nFigure 8 material constants used internally by top99neo_inertial_freq:\n');
fprintf('  E0 = 1e7 Pa, nu = 0.3, rho0 = 1 kg/m^3\n');
fprintf('  Emin = 1e-9*E0, rho_min = 1e-9*rho0, d = 6, x_cut = 0.1\n');
fprintf('  Stage 1 load DOF (auto-selected from full-solid mode 1): %d\n', info.stage1.loadDof);
fprintf('  Stage 1 frequency: omega1 = %.4f rad/s (paper: 224.6)\n', info.stage1.omega1);
fprintf('  Stage 2 frequency: omega1 = %.4f rad/s (paper: 255.6)\n', info.stage2.omega1);

