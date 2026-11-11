function summary = run_alpha_sweep(which, outRoot)
%RUN_ALPHA_SWEEP  AppliSci revision (C2): recompute Tables 2-5.
%
%   summary = run_alpha_sweep('clamped')   clamped beam 400x50   (Tables 2-3)
%   summary = run_alpha_sweep('building')  building 80x240       (Tables 4-5)
%   summary = run_alpha_sweep('both')
%
%   Formulation agreed for the revision (2026-11-10): weighted SUM of
%   compliances, c = alpha*c_1 + (1-alpha)*c_2, with f_j = omega_j0^2 M(x) phi_j0
%   (optimization.load_case_weighting = "compliance").  Same optimizer and
%   stopping rule as Table 1: OC, move 0.2, stagnation_x (W = 10,
%   eps_x = eps_M = 1e-2).  Topology modes are STRUCTURAL modes only
%   (structural_mode_check), correlation is the classical MAC.
%
%   Each run writes, under outRoot/<case>/alpha_<tag>/:
%     config.json            the exact configuration passed to the runner
%     <case>_correlation.csv MAC matrix (rows: solid-domain modes, cols: topology modes)
%     result.mat             x, omega, iterations, telemetry
%   and outRoot/<case>/summary.csv with one row per alpha, written by
%   rebuild_summaries (which can also regenerate it later from the saved
%   per-alpha files, without re-optimizing).  omega1/omega2 in summary.csv are
%   the lowest two STRUCTURAL topology modes (correlation CSV header); the
%   runner's raw omega(1:2), which can be void-localized modes, are kept as
%   omega1_raw/omega2_raw.

if nargin < 1 || isempty(which), which = 'both'; end
here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
if nargin < 2 || isempty(outRoot)
    outRoot = fullfile(here, 'results');
end

cases = {};
switch lower(which)
    case 'clamped',  cases = {'clamped'};
    case 'building', cases = {'building'};
    case 'both',     cases = {'clamped', 'building'};
    otherwise, error('run_alpha_sweep:BadCase', 'Use clamped, building or both.');
end

alphas = [1.00 0.75 0.50 0.25 0.00];
summary = struct();
for ic = 1:numel(cases)
    name = cases{ic};
    switch name
        case 'clamped'
            base = jsondecode(fileread(fullfile(repoRoot, 'examples', 'ClampedBeam', 'BeamTopOptFreq.json')));
        case 'building'
            base = jsondecode(fileread(fullfile(repoRoot, 'examples', 'Building', 'BuildingTopOptFreq.json')));
    end
    base = revisionSettings(base);
    for ia = 1:numel(alphas)
        a = alphas(ia);
        tag = strrep(sprintf('%.2f', a), '.', 'p');
        runDir = fullfile(outRoot, name, ['alpha_' tag]);
        if ~exist(runDir, 'dir'), mkdir(runDir); end
        cfg = base;
        cfg.domain.load_cases(1).factor = a;
        cfg.domain.load_cases(2).factor = 1 - a;
        cfg.meta.notes = sprintf('AppliSci revision C2, %s, alpha = %.2f', name, a);
        jsonPath = fullfile(runDir, [name '.json']);
        fid = fopen(jsonPath, 'w'); fprintf(fid, '%s', jsonencode(cfg, 'PrettyPrint', true)); fclose(fid);
        copyfile(jsonPath, fullfile(runDir, 'config.json'));

        fprintf('\n===== %s, alpha = %.2f =====\n', name, a);
        prevDir = pwd; cd(runDir); cleanup = onCleanup(@() cd(prevDir));
        [x, omega, tIter, nIter, ~, nIterStage, telemetry] = run_topopt_from_json(jsonPath);
        clear cleanup;
        save(fullfile(runDir, 'result.mat'), 'x', 'omega', 'tIter', 'nIter', 'nIterStage', 'telemetry', 'cfg', 'a');
    end
    T = rebuild_summaries(name, outRoot);
    summary.(name) = T;
end
end

function cfg = revisionSettings(cfg)
o = cfg.optimization;
o.approach = 'ourApproach';
o.optimizer = 'OC';
o.objective = 'compliance';
o.load_case_weighting = 'compliance';
o.move_limit = 0.2;
o.max_iters = 2000;
o.convergence_tol = 0.01;          % not read by stagnation_x
o.stop_criterion = 'stagnation_x';
o.stagnation = struct('window', 10, 'grayness_tol', 1e-2, 'relative_change_tol', 1e-2);
o.semi_harmonic_baseline = 'solid';
o.harmonic_normalize = false;      % no effect on semi_harmonic loads
for f = {'debug_semi_harmonic', 'semi_harmonic_rho_source'}
    if isfield(o, f{1}), o = rmfield(o, f{1}); end
end
cfg.optimization = o;

p = cfg.postprocessing;
p.compute_modes = 100;
p.structural_mode_check = true;
p.visualize_live = false;
p.visualize_modes = struct('enabled', false, 'count', 3);
p.visualize_topology_modes = struct('enabled', true, 'count', 3);
p.write_correlation_table = false;
p.correlation = struct('enabled', true, 'metric', 'mac', 'write_csv', true, 'best_k', 3);
p.save_snapshot_image = false;
p.save_final_image = true;
p.save_frequency_iterations = false;
cfg.postprocessing = p;
end

