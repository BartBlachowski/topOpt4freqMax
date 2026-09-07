%% Du--Olhoff 2007 reproduction - Yuksel Figure 9 cantilever problem
clearvars -except optimizer; clc; close all;

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

% Match analysis/YukselApproach/Matlab/run_cantilever.m.
nelx = 150;
nely = 100;
volfrac = 0.5;
penal = 3;
rmin = 2.3;

% Inner-loop optimizer (step 3 of the Du--Olhoff Fig. 1 loop):
%   "lp"  -- Eq. (22) LP route after Krog & Olhoff (1999); one linprog solve
%            per outer iteration.  This is the reproduction's working route
%            and the setting every result in this directory was produced with.
%   "mma" -- the paper-literal MMA inner loop on problem (25) with the full
%            Eq. (25d) coupling.  It is the clean-room study's labelled
%            BASELINE, not its successful configuration (NOTES.md 7 records
%            that it does not converge once N >= 2), and it costs up to
%            max_inner = 300 MMA sub-iterates per outer iteration, so the
%            iteration cap set below is far more expensive on that route.
% Set the variable before running to override without editing this file:
%   optimizer = "mma"; run_olhoff_cantilever
if ~exist('optimizer','var') || isempty(optimizer)
    optimizer = "lp";
end

% Olhoff reproduction-specific optimization controls. The default 0.20 point
% mass fraction matches Yuksel: 20% of permitted material mass, placed at the
% right-edge midpoint on both translational DOFs.
move = 0.005;
maxit = 200;
bcType = "cantilever";
runCfg = struct('verbose',true,'tol_mult',0.05,'tip_mass_fraction',0.20, ...
    'optimizer',optimizer);

[rho,omega,info] = topopt_olhoff_reproduced2007( ...
    nelx,nely,volfrac,penal,rmin,move,maxit,bcType,runCfg);

figure('Name','Olhoff reproduced 2007 - cantilever','Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');
nexttile;
imagesc(1-reshape(rho,nely,nelx));
set(gca,'YDir','normal'); axis equal tight off; colormap(gray(256)); clim([0 1]);
title(sprintf('Cantilever with tip mass: \\omega_1 = %.2f rad/s',omega(1)), ...
    'Interpreter','tex');
nexttile;
omegaHistory = [info.history.omega(1:3,:) omega(1:3)];
plot(0:info.nOuter,omegaHistory','LineWidth',1.2); grid on;
xlabel('Outer iteration'); ylabel('\omega [rad/s]');
legend('\omega_1','\omega_2','\omega_3','Location','best');
title(sprintf('Du--Olhoff %s trajectory',info.route.label));

fprintf('\nOlhoff reproduced 2007 -- cantilever Yuksel problem\n');
fprintf('  mesh: %dx%d, LxH: 15x10, rmin: %.2f elements\n',nelx,nely,rmin);
fprintf('  concentrated mass per translational DOF: %.6f\n',info.model.tipMassValue);
fprintf('  optimizer: %s\n',info.route.description);
fprintf('  status: %s, iterations: %d\n',info.status,info.nOuter);
fprintf('  omega1/2/3: %.6f / %.6f / %.6f rad/s\n',omega(1),omega(2),omega(3));
