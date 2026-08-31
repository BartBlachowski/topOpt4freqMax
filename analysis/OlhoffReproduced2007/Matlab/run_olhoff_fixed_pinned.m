%% Du--Olhoff 2007 reproduction - Yuksel Figure 8 fixed--pinned problem
clear; clc; close all;

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

% Match analysis/YukselApproach/Matlab/run_fixed_pinned.m.
nelx = 320;
nely = 40;
volfrac = 0.5;
penal = 3;
rmin = 2.0;

% Olhoff reproduction-specific optimization controls.
move = 0.005;
maxit = 400;
bcType = "fixedPinned";
runCfg = struct('verbose',true,'tol_mult',0.07);

[rho,omega,info] = topopt_olhoff_reproduced2007( ...
    nelx,nely,volfrac,penal,rmin,move,maxit,bcType,runCfg);

figure('Name','Olhoff reproduced 2007 - fixed pinned','Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');
nexttile;
imagesc(1-reshape(rho,nely,nelx));
set(gca,'YDir','normal'); axis equal tight off; colormap(gray(256)); clim([0 1]);
title(sprintf('Fixed--pinned: \\omega_1 = %.2f rad/s',omega(1)),'Interpreter','tex');
nexttile;
omegaHistory = [info.history.omega(1:3,:) omega(1:3)];
plot(0:info.nOuter,omegaHistory','LineWidth',1.2); grid on;
xlabel('Outer iteration'); ylabel('\omega [rad/s]');
legend('\omega_1','\omega_2','\omega_3','Location','best');
title('Du--Olhoff Eq. (22) LP trajectory');

fprintf('\nOlhoff reproduced 2007 -- fixed-pinned Yuksel problem\n');
fprintf('  mesh: %dx%d, LxH: 8x1, rmin: %.2f elements\n',nelx,nely,rmin);
fprintf('  status: %s, iterations: %d\n',info.status,info.nOuter);
fprintf('  omega1/2/3: %.6f / %.6f / %.6f rad/s\n',omega(1),omega(2),omega(3));
