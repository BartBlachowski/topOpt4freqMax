%% Yuksel & Yilmaz (2025) - Figure 4 benchmark (simply supported beam)
% Reproduces the setup reported in Figure 4:
% E0 = 1e7 Pa, rho0 = 1 kg/m^3, nu = 0.3, L:H = 8:1, volfrac = 0.5, mesh = 320x40.
% The solver uses sensitivity filtering and the OC move limit 0.2.
clear; clc; close all;

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

% Figure 4 geometry/mesh (L = 8, H = 1 -> nelx:nely = 320:40)
nelx = 400;
nely = 50;

% Optimization setup (paper-consistent)
volfrac = 0.5;
penal = 3;
rmin = 2.5;
ft = 1;           % sensitivity filter only
ftBC = 'N';       % symmetric filter boundary
eta = 0.5;        % not used when ft = 1
beta = 1;         % not used when ft = 1
move = 0.2;
maxit = 1000;      % upper bound; solver also stops at change < 0.01
stage1_maxit = 1000;
bcType = "simply";
visualizationQuality = 'regular'; % set 'smooth' for high-quality final display

[xPhysStage2, ~, info] = top99neo_inertial_freq( ...
    nelx, nely, volfrac, penal, rmin, ft, ftBC, eta, beta, move, maxit, stage1_maxit, bcType);

figure('Name','Yuksel Figure 4 benchmark','Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

nexttile;
imgStage1 = buildTopologyDisplayImage(info.stage1.xFinal, nelx, nely, visualizationQuality, true);
imagesc(1 - imgStage1);
set(gca, 'YDir', 'normal');
axis equal off;
caxis([0 1]);
colormap(gray);
title(sprintf('Figure 4(b): \\omega_1 = %.1f rad/s', info.stage1.omega1), 'Interpreter', 'tex');

nexttile;
imgStage2 = buildTopologyDisplayImage(xPhysStage2, nelx, nely, visualizationQuality, true);
imagesc(1 - imgStage2);
set(gca, 'YDir', 'normal');
axis equal off;
caxis([0 1]);
colormap(gray);
title(sprintf('Figure 4(c): \\omega_1 = %.1f rad/s', info.stage2.omega1), 'Interpreter', 'tex');

% Second figure: Figure-6 dynamic-code frequency convergence history (Section 3).
dynMaxIt = 200;
dynMove = 0.01; % paper setting for the dynamic run
[~, infoDyn] = top99neo_dynamic_freq( ...
    nelx, nely, volfrac, penal, rmin, ft, ftBC, dynMove, dynMaxIt, bcType);
if isfield(infoDyn, 'omegaHist') && ~isempty(infoDyn.omegaHist)
    plot_dynamic_frequency_convergence(infoDyn.omegaHist, 'Figure 6 (simply supported)');
    dynHistLen = size(infoDyn.omegaHist, 1);
    omegaDyn = infoDyn.omega1;
else
    warning('Dynamic omega history is unavailable; Figure 6 convergence plot was skipped.');
    dynHistLen = 0;
    omegaDyn = NaN;
end

fprintf('\nFigure 4 material constants used internally by top99neo_inertial_freq:\n');
fprintf('  E0 = 1e7 Pa, nu = 0.3, rho0 = 1 kg/m^3\n');
fprintf('  Emin = 1e-9*E0, rho_min = 1e-9*rho0, d = 6, x_cut = 0.1\n');
fprintf('  Stage 1 frequency: omega1 = %.4f rad/s\n', info.stage1.omega1);
fprintf('  Stage 2 frequency: omega1 = %.4f rad/s\n', info.stage2.omega1);
fprintf('  Dynamic code frequency (200 iters): omega1 = %.4f rad/s\n', omegaDyn);
fprintf('  Dynamic code history points stored: %d\n', dynHistLen);
