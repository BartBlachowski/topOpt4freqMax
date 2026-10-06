%RUN_PINNED_PINNED_OURAPPROACH_FREQ_HISTORY  OurApproach omega_1..3 history, pinned-pinned beam.
%
%   Reconstructs the "OurApproach frequency history" figure (paper Fig. 9b)
%   for the simply-supported (pinned-pinned) beam of Du & Olhoff (2007),
%   Fig. 2a (docs/s00158-007-0101-y.pdf): 8 x 1 domain, 240 x 30 mesh,
%   volume fraction 0.5, both end nodes pinned at mid-height.
%
%   Uses the frozen conference-benchmark Proposed profile unchanged
%   (confbench_method_config('proposed', ...)), i.e. exactly the
%   configuration of the Proposed rows of the paper's Table 1 (236 iterations
%   at 240x30).  Recording the design of every iteration does not change the
%   design path.
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
%   Output (next to this file):
%     OurApproach_240x30_freq_iterations.fig / .png

nelx = 240;
nely = 30;

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
addpath(fullfile(repo, 'tools', 'Matlab'));
addpath(fullfile(repo, 'examples', 'Performance', 'conference_bench'));
addpath(fullfile(repo, 'examples', 'Performance', 'benchmark_profile'));

[cfg, profileId] = confbench_method_config('proposed', nelx, nely);
cfg.optimization.approach = 'OurApproach';
cfg.postprocessing.record_design_history = true;

[x, ~, ~, nIter, ~, ~, telemetry] = run_topopt_from_json(cfg);
omegaE1 = e1_structural_omegas(telemetry.design_history, nelx, nely, 3);
save_frequency_iteration_plot(omegaE1, 'OurApproach', nelx, nely, here);

% The helper must reproduce the frozen evaluator on the final design.
ev = study_evaluate_design(x, nelx, nely, 0.5, 'ComputeBinaryDiagnostic', false);
wFinal = e1_structural_omegas(x, nelx, nely, 3);
assert(abs(wFinal(1) - ev.selected_omega_raw_E1) <= 1e-8*ev.selected_omega_raw_E1, ...
    'e1_structural_omegas disagrees with study_evaluate_design.');

fprintf(['OurApproach (%s) pinned-pinned %dx%d: %d iterations; E1 structural omega_1..3 ' ...
    'of the last iterate = %.2f, %.2f, %.2f rad/s; final design E1 omega_1 = %.2f rad/s\n'], ...
    profileId, nelx, nely, nIter, omegaE1(end,:), ev.selected_omega_raw_E1);
