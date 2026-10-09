%RUN_PINNED_PINNED_YUKSEL_FREQ_HISTORY  Yuksel omega_1..3 history, pinned-pinned beam.
%
%   Companion of the two paper Fig. 9 runners in this folder for the third
%   method of Table 1, the two-stage inertial-load method of Yuksel & Yilmaz
%   (2025) (analysis/Yuksel/Matlab/top99neo_inertial_freq.m), on the same
%   simply-supported (pinned-pinned) beam of Du & Olhoff (2007), Fig. 2a:
%   8 x 1 domain, 240 x 30 mesh, volume fraction 0.5, both end nodes pinned at
%   mid-height.
%
%   Uses the frozen conference-benchmark Yuksel profile
%   (confbench_method_config('yuksel', ...)); with the default stop
%   parameters below this is exactly the configuration of the Yuksel rows of
%   the paper's Table 1 (320 iterations at 240x30).  Recording the design of
%   every iteration does not change the design path.
%
%   Both stages are OC compliance minimization: stage 1 under a unit point
%   load at mid-span, which only serves to estimate the fundamental mode,
%   stage 2 under the design-dependent inertial load F = M(x)*u_hat.
%   Iterations are numbered globally (stage 1, then stage 2) and the dashed
%   line on the frequency plot marks the handoff.  Stage 1 does not maximize a
%   frequency, so its part of the history is not a frequency-optimization
%   history.  At 240x30 stage 1 takes 168 iterations and stage 2 152 (final
%   E1 omega_1 = 159.44 rad/s); E1 omega_1 is within 0.5 % of each stage's
%   final value from iteration 36 and from iteration 185 (17 iterations into
%   stage 2).
%
%   The plotted omega_1..3 are the three lowest STRUCTURAL modes under the
%   benchmark's common evaluator E1 (e1_structural_omegas), as for the
%   proposed method in Fig. 9b, so the last point is Table 1's omega_1.
%   Unlike the other two runners, the design of iteration k is the design
%   AFTER the update of iteration k, because that is what the solver records.
%
%   Stop parameters.  Each stage stops once max|x - x_old| < its tolerance
%   (from its second iteration on) or after max_iter iterations of that
%   stage; the run ends when stage 2 stops.  stage1_tol also decides the
%   design stage 2 starts from, so changing it changes the whole of stage 2,
%   not only where stage 1 ends.
%
%   stop_criterion = 'relative_l2_change' makes both stages test
%   ||x - x_old||_2/||x_old||_2 < tolerance instead (optimization.stop_criterion),
%   with stage1_rel_tol and stage2_rel_tol; stage1_tol and stage2_tol are then
%   ignored, as the relative tolerances are with 'max_change'.
%
%   Every save_every_it-th iterate (and the last one) is also rendered, so the
%   topology can be compared with the frequency history; the snapshot of
%   iteration k is the design whose omega_1..3 are plotted at k.
%
%   Output (next to this file):
%     Yuksel_240x30_freq_iterations.fig / .png
%     topologies/Yuksel_240x30_it<k>.png

nelx = 240;
nely = 30;
save_every_it = 25;   % topology snapshot every save_every_it iterations (+ last); 0 = none
stage1_tol = 0.04;    % stage 1 stops when max|x - x_old| < stage1_tol; 0.01 = Table 1
stage2_tol = 0.04;    % stage 2 (the run) stops when max|x - x_old| < stage2_tol; 0.01 = Table 1
stop_criterion = 'max_change';   % 'max_change' (Table 1 rule) | 'relative_l2_change'
stage1_rel_tol = 1e-3;   % stage 1: ||x - x_old||_2/||x_old||_2 < stage1_rel_tol; no calibrated value
stage2_rel_tol = 1e-3;   % stage 2: ||x - x_old||_2/||x_old||_2 < stage2_rel_tol; no calibrated value
max_iter = [];        % iteration cap of EACH stage; [] = frozen profile value (1000)

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
addpath(fullfile(repo, 'tools', 'Matlab'));
addpath(fullfile(repo, 'examples', 'Performance', 'conference_bench'));
addpath(fullfile(repo, 'examples', 'Performance', 'benchmark_profile'));

[cfg, profileId] = confbench_method_config('yuksel', nelx, nely);
cfg.postprocessing.record_design_history = true;
switch stop_criterion
    case 'max_change'
        tol1 = stage1_tol; tol2 = stage2_tol;
        stopDesc = sprintf('stage1_tol = %g, stage2_tol = %g', tol1, tol2);
    case 'relative_l2_change'
        cfg.optimization.stop_criterion = stop_criterion;
        tol1 = stage1_rel_tol; tol2 = stage2_rel_tol;
        stopDesc = sprintf('||dx||_2/||x||_2: stage1_rel_tol = %g, stage2_rel_tol = %g', tol1, tol2);
    otherwise
        error('stop_criterion must be ''max_change'' or ''relative_l2_change'' (got ''%s'').', ...
            stop_criterion);
end
cfg.optimization.yuksel.stage1_tol = tol1;
cfg.optimization.yuksel.stage2_tol = tol2;
cfg.optimization.convergence_tol = tol2;   % mirrors stage2_tol, as in the frozen profile
if ~isempty(max_iter)
    cfg.optimization.max_iters = max_iter;
    cfg.optimization.yuksel.stage1_max_iters = max_iter;
end

[x, ~, ~, nIter, ~, ~, telemetry] = run_topopt_from_json(cfg);
xHist = telemetry.design_history;
nStage1 = telemetry.stopping.iter_stage1;
assert(size(xHist, 2) == nIter && isequal(xHist(:,end), x(:)), ...
    'Design history does not match the iteration count or the final design.');

omegaE1 = e1_structural_omegas(xHist, nelx, nely, 3);
save_frequency_iteration_plot(omegaE1, 'Yuksel', nelx, nely, here, nStage1);
save_topology_snapshots(xHist, omegaE1, save_every_it, ...
    'Yuksel', nelx, nely, fullfile(here, 'topologies'));

% The helper must reproduce the frozen evaluator on the final design.
ev = study_evaluate_design(x, nelx, nely, 0.5, 'ComputeBinaryDiagnostic', false);
assert(abs(omegaE1(end,1) - ev.selected_omega_raw_E1) <= 1e-8*ev.selected_omega_raw_E1, ...
    'e1_structural_omegas disagrees with study_evaluate_design.');

fprintf(['Yuksel (%s) pinned-pinned %dx%d, %s: %d iterations ' ...
    '(stage 1: %d, stage 2: %d, %s); E1 structural omega_1..3 of the final design = ' ...
    '%.2f, %.2f, %.2f rad/s\n'], ...
    profileId, nelx, nely, stopDesc, nIter, nStage1, ...
    telemetry.stopping.iter_stage2, telemetry.stopping.stop_reason, omegaE1(end,:));
