% RUN_SIMPLY_SIMPLY_EXACT  Du & Olhoff (2007) SS beam example.
%
%   Reproduces Table 1 / Fig. 3a of Du & Olhoff (2007):
%     Simply-simply supported beam, 40x5 mesh, volfrac=0.5
%     Paper target: omega_1 -> 174.7 rad/s  (bimodal at optimum)
%
%   Reference: Du & Olhoff (2007), Struct Multidisc Optim 34:91-110.

addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'tools', 'Matlab'));

cfg = struct();
cfg.support_type    = 'SS';
cfg.nelx            = 40;
cfg.nely            = 5;
cfg.volfrac         = 0.5;
cfg.outer_max_iter  = 300;
cfg.outer_tol       = 1e-4;
cfg.move_lim        = 0.2;
cfg.verbose         = true;

fprintf('\n');
fprintf('==========================================================\n');
fprintf('  Du & Olhoff (2007) — Simply-Simply (SS) beam\n');
fprintf('  40x5 mesh, volfrac=0.5\n');
fprintf('  Paper Table 1 target: omega_1 = 174.7 rad/s (bimodal)\n');
fprintf('==========================================================\n\n');

[rho_final, hist] = topopt_freq_exact(cfg);

ni = hist.outer_iters;
fprintf('\n==========================================================\n');
fprintf('  RESULTS\n');
fprintf('==========================================================\n');
fprintf('  Outer iterations: %d\n',     ni);
fprintf('  Final omega_1   : %.4f rad/s\n',  hist.omega(ni, 1));
fprintf('  Final omega_2   : %.4f rad/s\n',  hist.omega(ni, 2));
fprintf('  Final N         : %d\n',          hist.N(ni));
fprintf('  Final volume    : %.4f\n',         hist.volume(ni));
fprintf('  Paper target    : 174.7 rad/s (N=2)\n');
fprintf('==========================================================\n\n');

% Plot topology.
figure('Name','SS optimized topology');
nely = 5; nelx = 40;
rho_img = reshape(rho_final, nely, nelx);
imagesc(1-rho_img);
colormap(gray); axis equal tight off;
title(sprintf('SS: \\omega_1=%.1f rad/s (target 174.7, paper)', hist.omega(ni,1)));

% Plot convergence.
figure('Name','SS convergence');
plot(1:ni, hist.omega(1:ni,1), 'b-o', 'MarkerSize', 3);
hold on;
plot(1:ni, hist.omega(1:ni,2), 'r--s', 'MarkerSize', 3);
yline(174.7, 'k--', 'Paper target');
xlabel('Outer iteration'); ylabel('\omega (rad/s)');
legend('\omega_1','\omega_2','Paper target');
title('SS beam — convergence');
grid on;
