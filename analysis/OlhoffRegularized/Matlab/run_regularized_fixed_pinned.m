%% OlhoffRegularized - fixed--pinned 8 x 1 beam
clearvars -except optimizer formulation; clc; close all;

thisDir=fileparts(mfilename('fullpath'));addpath(thisDir);
if ~exist('optimizer','var')||isempty(optimizer),optimizer="lp";end
if ~exist('formulation','var')||isempty(formulation),formulation="olhoff";end

nelx=160;nely=20;volfrac=.5;penal=3;rmin=1.5;move=.005;
bcType="fixedPinned";
optimizer="mma";
runCfg=struct('optimizer',optimizer,'formulation',formulation,'verbose',true, ...
    'tol_mult',.05,'max_outer_iterations',1000,'max_inner_iterations',500, ...
    'max_trial_steps',8,'move_min',1e-7,'move_max',move, ...
    'accept_ratio',.10,'grow_ratio',.75,'stationarity_tol',.02, ...
    'density_tol',1e-4,'objective_tol',1e-5,'persistence',20, ...
    'progress_tolerance',1e-4,'progress_persistence',20);

[rho,omega,info]=topopt_olhoff_regularized( ...
    nelx,nely,volfrac,penal,rmin,move,bcType,runCfg);
localPlot(rho,omega,info,nelx,nely);
localReport(omega,info,nelx,nely,rmin);

function localPlot(rho,omega,info,nelx,nely)
figure('Name','OlhoffRegularized - fixed pinned','Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');
nexttile;imagesc(1-reshape(rho,nely,nelx));set(gca,'YDir','normal');
axis equal tight off;colormap(gray(256));clim([0 1]);
title(sprintf('Fixed--pinned: \\omega_1 = %.2f rad/s',omega(1)),'Interpreter','tex');
nexttile;plot(1:info.nOuter,info.history.omega(1:3,:)','LineWidth',1.2);
grid on;xlabel('Outer iteration');ylabel('\omega [rad/s]');
legend('\omega_1','\omega_2','\omega_3','Location','best');title(info.route.description);
end

function localReport(omega,info,nelx,nely,rmin)
fprintf('\nOlhoffRegularized -- fixed-pinned\n');
fprintf('  mesh: %dx%d, rmin: %.3g elements\n',nelx,nely,rmin);
fprintf('  route: %s\n',info.route.description);
fprintf('  status: %s (%s)\n',info.status,info.stop_reason);
fprintf('  outer/accepted/trials/inner: %d / %d / %d / %d\n', ...
    info.iterations.outer,info.iterations.accepted_updates, ...
    info.iterations.trial_total,info.iterations.inner_total);
fprintf('  omega1/2/3: %.6f / %.6f / %.6f rad/s\n',omega(1),omega(2),omega(3));
end
