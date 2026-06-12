% RUN_SIMPLY_SIMPLY_EXACT  Du & Olhoff (2007) SS beam example.
%
%   Reproduces Table 1 / Fig. 3a of Du & Olhoff (2007):
%     Simply-simply supported beam, configured mesh, volfrac=0.5
%     Paper target: omega_1 -> 174.7 rad/s  (bimodal at optimum)
%
%   Reference: Du & Olhoff (2007), Struct Multidisc Optim 34:91-110.

addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'tools', 'Matlab'));

cfg = struct();
cfg.support_type    = 'SS';
cfg.nelx            = 160;
cfg.nely            = 20;
cfg.volfrac         = 0.5;
cfg.mass_mode       = 'du2007_c1';
cfg.sensitivity_filter = true;
cfg.rmin_elem       = 2.5;
cfg.outer_max_iter  = 300;
cfg.inner_max_iter  = 30;
cfg.outer_tol       = 1e-4;
cfg.move_lim        = Inf;    % PHASE A: disable extra inner move-limit (not in paper)
cfg.acceptance_check = false;  % PHASE A: disable backtracking acceptance (not in paper)
cfg.verbose         = true;

fprintf('\n');
fprintf('==========================================================\n');
fprintf('  Du & Olhoff (2007) — Simply-Simply (SS) beam\n');
fprintf('  %dx%d mesh, volfrac=%.1f\n', cfg.nelx, cfg.nely, cfg.volfrac);
fprintf('  Paper Table 1 target: omega_1 = 174.7 rad/s (bimodal)\n');
fprintf('==========================================================\n\n');

[rho_final, hist] = topopt_freq_exact(cfg);

ni = hist.outer_iters;
omega_final = hist.final_omega;
if isfield(hist, 'omega_trial')
    omega_hist = hist.omega_trial;
else
    omega_hist = hist.omega;
end
if all(isnan(omega_hist(:))), omega_hist = hist.omega; end

fprintf('\n==========================================================\n');
fprintf('  RESULTS\n');
fprintf('==========================================================\n');
fprintf('  Outer iterations: %d\n',     ni);
fprintf('  Final omega_1   : %.4f rad/s\n',  omega_final(1));
fprintf('  Final omega_2   : %.4f rad/s\n',  omega_final(2));
fprintf('  Final N         : %d\n',          hist.final_N);
fprintf('  Final volume    : %.4f\n',         hist.final_volume);
fprintf('  Paper target    : 174.7 rad/s (N=2)\n');
fprintf('==========================================================\n\n');

% Plot topology.
figure('Name','SS optimized topology');
rho_img = reshape(rho_final, cfg.nely, cfg.nelx);
imagesc(1-rho_img);
colormap(gray); axis equal tight off;
title(sprintf('SS: \\omega_1=%.1f rad/s (target 174.7, paper)', omega_final(1)));

% Plot convergence.
figure('Name','SS convergence');
plot(1:ni, omega_hist(1:ni,1), 'b-o', 'MarkerSize', 3);
hold on;
plot(1:ni, omega_hist(1:ni,2), 'r--s', 'MarkerSize', 3);
yline(174.7, 'k--', 'Paper target');
xlabel('Outer iteration'); ylabel('\omega (rad/s)');
legend('\omega_1','\omega_2','Paper target');
title('SS beam — convergence');
grid on;
