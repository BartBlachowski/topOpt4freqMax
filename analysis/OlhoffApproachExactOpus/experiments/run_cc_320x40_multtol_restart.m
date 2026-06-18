function run_cc_320x40_multtol_restart()
%RUN_CC_320X40_MULTTOL_RESTART  Early multiplicity activation diagnostic.
%
% Starts from the saved Case-B local move-control design and runs three
% multiplicity-tolerance cases for 100 additional outer iterations.

root = '/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(root,'analysis','OlhoffApproachExact','Matlab'));
addpath(fullfile(root,'tools','Matlab'));

indir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results_local_move');
src = fullfile(indir, 'cc_local_B_move005.mat');
outdir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results_multtol');
if ~exist(outdir,'dir'), mkdir(outdir); end

S = load(src, 'rho_final', 'hist', 'cfg');
rho_start = S.rho_final(:);

base = S.cfg;
base.initial_rho = rho_start;
base.outer_max_iter = 100;
base.rho_snapshot_interval = 1;
base.verbose = false;
base.move_lim = 0.05;
base.outer_move = 0.05;

cases = struct( ...
    'name', {'B0_mult001', 'B1_mult005', 'B2_mult010'}, ...
    'mult_tol', {0.01, 0.05, 0.10});

for ci = 1:numel(cases)
    cfg = base;
    cfg.mult_tol = cases(ci).mult_tol;
    tag = cases(ci).name;

    fprintf('\n===== CC multtol restart %s =====\n', tag);
    fprintf('start: Case B move=0.05 final design; mult_tol=%.4g\n', cfg.mult_tol);
    t0 = tic;
    [rho_final, hist] = topopt_freq_exact(cfg);
    el = toc(t0);

    ot = hist.omega_trial;
    if all(isnan(ot(:))), ot = hist.omega; end
    [bestw1, bi] = max(ot(:,1));

    n2 = find(hist.N_trial == 2);
    if isempty(n2)
        firstN2 = NaN;
        maxRunN2 = 0;
    else
        firstN2 = n2(1);
        d = diff([0; n2(:); inf]);
        runStarts = find(d(1:end-1) > 1);
        runEnds = find(d(2:end) > 1);
        maxRunN2 = max(runEnds - runStarts + 1);
    end

    fprintf('--- %s done in %.1fs, outer_iters=%d ---\n', tag, el, hist.outer_iters);
    fprintf('N=2 first=%g max_consecutive=%d\n', firstN2, maxRunN2);
    fprintf('BEST  : local_iter=%d omega1=%.3f omega2=%.3f omega3=%.3f Ntrial=%d vol=%.4f\n', ...
        bi, ot(bi,1), ot(bi,2), ot(bi,3), hist.N_trial(bi), hist.volume(bi));
    fprintf('FINAL : omega1=%.3f omega2=%.3f omega3=%.3f N=%d vol=%.4f\n', ...
        hist.final_omega(1), hist.final_omega(2), hist.final_omega(3), hist.final_N, hist.final_volume);
    fprintf('LAST  : drho_norm=%.3e drho_max=%.3e\n', hist.drho_norm(end), hist.drho_max(end));

    save(fullfile(outdir, sprintf('cc_multtol_%s.mat', tag)), ...
        'rho_start','rho_final','hist','cfg','el','bestw1','bi','src','firstN2','maxRunN2','-v7');
    writematrix(rho_final(:), fullfile(outdir, sprintf('rho_multtol_%s.csv', tag)));
end

fprintf('\nAll multiplicity-tolerance restarts complete. Results saved to %s\n', outdir);
end
