function run_cc_320x40_lp_vs_nl_restart()
%RUN_CC_320X40_LP_VS_NL_RESTART  Compare nonlinear and LP-reduced embeddings.
%
% Starts from the saved connected 320x40 Case-B local move-control design.
% Both cases use the same restart state and settings; only
% cfg.subproblem_formulation differs.

root = '/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','lp_vs_nl_diagnostic'));
addpath(fullfile(root,'analysis','OlhoffApproachExact','Matlab'));
addpath(fullfile(root,'tools','Matlab'));

indir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results_local_move');
src = fullfile(indir, 'cc_local_B_move005.mat');
outdir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results_lp_vs_nl');
if ~exist(outdir,'dir'), mkdir(outdir); end

S = load(src, 'rho_final', 'cfg');
rho_start = S.rho_final(:);

base = S.cfg;
base.initial_rho = rho_start;
base.outer_max_iter = 100;
base.rho_snapshot_interval = 1;
base.verbose = false;
base.move_lim = 0.05;
base.outer_move = 0.05;

% Use the relaxed multiplicity tolerance from the preceding activation
% diagnostic so the comparison actually enters the N>1 subproblem.
base.mult_tol = 0.05;

cases = struct( ...
    'name', {'NL_nonlinear', 'LP_reduced'}, ...
    'subproblem_formulation', {'nonlinear', 'lp_reduced'});

for ci = 1:numel(cases)
    cfg = base;
    cfg.subproblem_formulation = cases(ci).subproblem_formulation;
    tag = cases(ci).name;

    fprintf('\n===== CC LP-vs-NL restart %s =====\n', tag);
    fprintf('start: Case B move=0.05 final design; mult_tol=%.4g; formulation=%s\n', ...
        cfg.mult_tol, cfg.subproblem_formulation);
    t0 = tic;
    [rho_final, hist] = topopt_freq_exact_lpdiag(cfg);
    el = toc(t0);

    ot = hist.omega_trial;
    if all(isnan(ot(:))), ot = hist.omega; end
    [bestw1, bi] = max(ot(:,1));
    gap_best = (ot(bi,2) - ot(bi,1)) / ot(bi,1);

    fprintf('--- %s done in %.1fs, outer_iters=%d ---\n', tag, el, hist.outer_iters);
    fprintf('BEST  : local_iter=%d omega1=%.3f omega2=%.3f omega3=%.3f gap12=%.4f Ntrial=%d vol=%.4f\n', ...
        bi, ot(bi,1), ot(bi,2), ot(bi,3), gap_best, hist.N_trial(bi), hist.volume(bi));
    fprintf('FINAL : omega1=%.3f omega2=%.3f omega3=%.3f N=%d vol=%.4f\n', ...
        hist.final_omega(1), hist.final_omega(2), hist.final_omega(3), hist.final_N, hist.final_volume);
    fprintf('LAST  : drho_norm=%.3e drho_max=%.3e offdiag=%.3e\n', ...
        hist.drho_norm(end), hist.drho_max(end), hist.inner_offdiag_max(end));

    save(fullfile(outdir, sprintf('cc_embed_%s.mat', tag)), ...
        'rho_start','rho_final','hist','cfg','el','bestw1','bi','src','-v7');
    writematrix(rho_final(:), fullfile(outdir, sprintf('rho_embed_%s.csv', tag)));
end

fprintf('\nLP-vs-NL restart comparison complete. Results saved to %s\n', outdir);
end
