% Exp-1: CC frequency-max at 80x20 vs 40x5, UNCHANGED settings except mesh.
% Uses the existing OlhoffApproachExact solver verbatim (read-only addpath).
% No fixes/heuristics/continuation/projection/optimizer changes are introduced.
% Settings copied from run_clamped_clamped_exact.m; only nelx,nely differ.

root = '/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(root,'analysis','OlhoffApproachExact','Matlab'));
addpath(fullfile(root,'tools','Matlab'));
outdir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results');
if ~exist(outdir,'dir'), mkdir(outdir); end

base = struct();
base.support_type     = 'CC';
base.volfrac          = 0.5;
base.mass_mode        = 'du2007_c1';
base.sensitivity_filter = true;
base.rmin_elem        = 2.5;     % element units -- UNCHANGED (as run_clamped_clamped_exact.m)
base.outer_max_iter   = 300;
base.inner_max_iter   = 30;
base.outer_tol        = 1e-4;
base.move_lim         = Inf;     % PHASE A: unchanged
base.acceptance_check = false;   % PHASE A: unchanged
base.verbose          = false;

meshes = {[40 5],[80 20]};
for mi = 1:numel(meshes)
    cfg = base;
    cfg.nelx = meshes{mi}(1);
    cfg.nely = meshes{mi}(2);
    tag = sprintf('%dx%d', cfg.nelx, cfg.nely);
    fprintf('\n===== CC run mesh %s =====\n', tag);
    t0 = tic;
    [rho_final, hist] = topopt_freq_exact(cfg);
    el = toc(t0);

    ot = hist.omega_trial;                 % post-update freqs per outer iter
    if all(isnan(ot(:))), ot = hist.omega; end
    w1 = ot(:,1);
    [bestw1, bi] = max(w1);
    ni = hist.outer_iters;

    fprintf('--- mesh %s done in %.1fs, outer_iters=%d ---\n', tag, el, ni);
    fprintf('FINAL : omega1=%.3f omega2=%.3f N=%d vol=%.4f\n', ...
        hist.final_omega(1), hist.final_omega(2), hist.final_N, hist.final_volume);
    fprintf('BEST  : iter=%d omega1=%.3f omega2=%.3f Ntrial=%d vol=%.4f\n', ...
        bi, ot(bi,1), ot(bi,2), hist.N_trial(bi), hist.volume(bi));
    fprintf('Paper Fig.3c target: omega1=456.4 (bimodal, CONNECTED)\n');

    save(fullfile(outdir, sprintf('cc_%s.mat', tag)), ...
        'rho_final','hist','cfg','el','-v7');
    % also dump rho as plain CSV for language-agnostic post-processing
    writematrix(rho_final(:), fullfile(outdir, sprintf('rho_%s.csv', tag)));
end
fprintf('\nAll runs complete. Designs saved to %s\n', outdir);
