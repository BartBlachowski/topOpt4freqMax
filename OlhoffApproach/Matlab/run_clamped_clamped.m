%% ========================================================================
%  Olhoff & Du (2014) - Clamped-Clamped beam (CC)
%  Natural frequency maximization with SIMP + MMA
%  ========================================================================
clear; clc; close all;

cfg = struct();

% Geometry / mesh
cfg.L    = 8;
cfg.H    = 1;
cfg.nelx = 240;
cfg.nely = 30;

% Material
cfg.E0      = 1e7;
cfg.Emin    = max(1e-6*cfg.E0, 1e-3);
cfg.rho0    = 1.0;
cfg.rho_min = 1e-6;
cfg.nu      = 0.3;
cfg.t       = 1.0;

% Optimization controls
cfg.volfrac  = 0.5;
cfg.penal    = 3.0;
cfg.rmin     = 2 * cfg.L / cfg.nelx;
cfg.maxiter  = 300;
cfg.J        = 3;
cfg.supportType = "CC";

% Projection / continuation (optional tweaks)
cfg.beta_schedule = [1 2 4 8 16 32 64];
cfg.beta_interval = 40;

opts = struct('doDiagnostic', true, 'diagnosticOnly', false, 'diagModes', 5, ...
    'visualization_quality', 'regular');
if isfield(cfg, 'visualization_quality') && ~isempty(cfg.visualization_quality)
    opts.visualization_quality = cfg.visualization_quality;
end
paper = struct('init', 146.1, 'opt', 456.4);

[omega_best, xPhys_best, diag_out] = topFreqOptimization_MMA(cfg, opts);

fprintf('CC case: omega1 initial=%.1f (paper %.1f) | optimized=%.1f (paper %.1f)\n', ...
    diag_out.initial.omega(1), paper.init, omega_best, paper.opt);

figure('Name', 'Olhoff CC topology');
theme("light");
hold on;
imgDisp = buildTopologyDisplayImage(xPhys_best, cfg.nelx, cfg.nely, opts.visualization_quality, true);
imagesc(1 - imgDisp);
set(gca, 'YDir', 'normal');
axis equal tight off; colormap(gray(256));
title(sprintf('CC: omega1=%.1f (paper: %.1f)', omega_best, paper.opt));
