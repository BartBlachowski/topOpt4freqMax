%RUN_PINNED_PINNED_OURAPPROACH_FREQ_HISTORY  OurApproach omega_1..3 history, pinned-pinned beam.
%
%   Reconstructs the "OurApproach frequency history" figure (paper Fig. 9b)
%   for the simply-supported (pinned-pinned) beam of Du & Olhoff (2007),
%   Fig. 2a (docs/s00158-007-0101-y.pdf): 8 x 1 domain, 240 x 30 mesh,
%   volume fraction 0.5, both end nodes pinned at mid-height.
%
%   Uses the frozen conference-benchmark Proposed profile
%   (confbench_method_config('proposed', ...)); with the default stop
%   parameters below this is exactly the configuration of the Proposed rows
%   of the paper's Table 1 (236 iterations at 240x30).  Recording the design
%   of every iteration does not change the design path.
%
%   The plotted omega_1..3 are the three lowest STRUCTURAL modes of every
%   iterate under the benchmark's common evaluator E1 (e1_structural_omegas,
%   mirroring study_evaluate_design; see its header for the one deviation, on
%   the uniform starting design).  The solver's native eigenfrequencies
%   are not used: with the profile's void floor of 1e-9 they are localized
%   modes of near-void elements from about iteration 5 to convergence at
%   240x30 (final 108.8 / 109.9 / 117.8 rad/s against the structural
%   omega_1 = 157.64 rad/s).
%
%   Stop parameters.  The run stops when max|x - x_old| <= conv_tol on the
%   raw design field (the profile's native rule) or after max_iter
%   iterations.  At 240x30 max|x - x_old| falls to about 0.02 by iteration 45
%   and then hovers between 0.01 and 0.022, so with conv_tol = 0.01 the stop
%   falls on whichever iteration happens to dip below it.  conv_tol = 0.025
%   stops at iteration 42, after the topology has frozen.  Avoid 0.02: at
%   320x40 conv_tol = 0.019 stops at 98 and 0.020 at 47.
%
%   stop_criterion = 'relative_l2_change' replaces that rule by
%   ||x - x_old||_2/||x_old||_2 < rel_tol on the same raw design field
%   (optimization.stop_criterion); conv_tol is then ignored, as rel_tol is
%   with 'max_change'.
%
%   stop_criterion = 'stagnation' is the common rule of all three methods (R1,
%   examples/Performance/stop_criterion_study): stop once, over the last
%   stag_window+1 analysed designs, the compliance objective has varied by
%   < stag_obj_tol relative to its latest value AND the grayness
%   4*mean(xPhys(1-xPhys)) by < stag_gray_tol; conv_tol and rel_tol are then
%   ignored.
%
%   Every save_every_it-th iterate (and the last one) is also rendered, so the
%   topology can be compared with the frequency plateau of the history; the
%   snapshot of iteration k is the design whose omega_1..3 are plotted at k.
%
%   Output (next to this file):
%     OurApproach_240x30_freq_iterations.fig / .png
%     topologies/OurApproach_240x30_it<k>.png

nelx = 800;
nely = 100;
save_every_it = 25;   % topology snapshot every save_every_it iterations (+ last); 0 = none
conv_tol = 0.01;      % stop when max|x - x_old| <= conv_tol; 0.01 = Table 1
conv_tol = 0.04;	
% stop_criterion = 'max_change';   % previous setting
stop_criterion = 'stagnation';   % 'max_change' (Table 1 rule, conv_tol) | 'relative_l2_change' (rel_tol) | 'stagnation'
rel_tol = 1e-3;       % stop when ||x - x_old||_2/||x_old||_2 < rel_tol; no calibrated value
stag_window = 10;     % ('stagnation') window W: the last W+1 analysed designs
stag_obj_tol = 1e-3;  % ('stagnation') range(objective)/objective over the window
stag_gray_tol = 5e-3; % ('stagnation') range(4*mean(xPhys(1-xPhys))) over the window
max_iter = [];        % iteration cap; [] = frozen profile value (2000)

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
addpath(fullfile(repo, 'tools', 'Matlab'));
addpath(fullfile(repo, 'examples', 'Performance', 'conference_bench'));
addpath(fullfile(repo, 'examples', 'Performance', 'benchmark_profile'));

[cfg, profileId] = confbench_method_config('proposed', nelx, nely);
cfg.optimization.approach = 'OurApproach';
cfg.postprocessing.record_design_history = true;
switch stop_criterion
    case 'max_change'
        cfg.optimization.convergence_tol = conv_tol;
        stopDesc = sprintf('conv_tol = %g', conv_tol);
    case 'relative_l2_change'
        cfg.optimization.stop_criterion = stop_criterion;
        cfg.optimization.convergence_tol = rel_tol;
        stopDesc = sprintf('rel_tol = %g (||dx||_2/||x||_2)', rel_tol);
    case 'stagnation'
        cfg.optimization.stop_criterion = stop_criterion;
        cfg.optimization.stagnation = struct('window', stag_window, ...
            'objective_tol', stag_obj_tol, 'grayness_tol', stag_gray_tol);
        stopDesc = sprintf('stagnation over %d+1 designs (objective %g, grayness %g)', ...
            stag_window, stag_obj_tol, stag_gray_tol);
    otherwise
        error(['stop_criterion must be ''max_change'', ''relative_l2_change'' or ' ...
               '''stagnation'' (got ''%s'').'], stop_criterion);
end
if ~isempty(max_iter)
    cfg.optimization.max_iters = max_iter;
end

[x, ~, ~, nIter, ~, ~, telemetry] = run_topopt_from_json(cfg);
omegaE1 = e1_structural_omegas(telemetry.design_history, nelx, nely, 3);
save_frequency_iteration_plot(omegaE1, 'OurApproach', nelx, nely, here);
save_topology_snapshots(telemetry.design_history, omegaE1, save_every_it, ...
    'OurApproach', nelx, nely, fullfile(here, 'topologies'));

% The helper must reproduce the frozen evaluator on the final design.
ev = study_evaluate_design(x, nelx, nely, 0.5, 'ComputeBinaryDiagnostic', false);
wFinal = e1_structural_omegas(x, nelx, nely, 3);
assert(abs(wFinal(1) - ev.selected_omega_raw_E1) <= 1e-8*ev.selected_omega_raw_E1, ...
    'e1_structural_omegas disagrees with study_evaluate_design.');

fprintf(['OurApproach (%s) pinned-pinned %dx%d, %s: %d iterations (%s); E1 structural ' ...
    'omega_1..3 of the last iterate = %.2f, %.2f, %.2f rad/s; final design E1 omega_1 = %.2f rad/s\n'], ...
    profileId, nelx, nely, stopDesc, nIter, telemetry.stopping.stop_reason, omegaE1(end,:), ...
    ev.selected_omega_raw_E1);
