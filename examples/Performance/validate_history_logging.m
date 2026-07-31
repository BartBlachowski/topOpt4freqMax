clear; clc;

% WP4 exit criterion plus the section 4.2.1 threshold calibration.
%
% Two things are established here:
%
%   1. Instrumentation invariance -- enabling the iteration history changes no
%      trajectory.  Offline acceptance is evaluated on histories recorded with
%      logging ON but replayed with it OFF, so a logging-induced difference
%      would silently decouple the accepted design from the timed one.
%
%   2. The measured feasibility and stationarity floors each method actually
%      sustains, which is what plan section 4.2.1 requires before epsilon_V and
%      tau may be frozen.  The provisional values were preregistered with no
%      supporting data: rV had never been logged by any run in this repository.

thisDir = fileparts(mfilename('fullpath'));
baseCfg = jsondecode(fileread(fullfile(thisDir, 'performance_comparison.json')));
baseCfg.postprocessing.visualize_live = false;
baseCfg.postprocessing.save_final_image = false;
baseCfg.postprocessing.save_snapshot_image = false;
baseCfg.postprocessing.save_frequency_iterations = false;
baseCfg.optimization.filter.radius = 2;
baseCfg.optimization.filter.radius_units = 'element';

meshes     = [160, 20; 320, 40];
approaches = {'Olhoff',         'Yuksel',         'OurApproach'};
labels     = {'OlhoffApproach', 'YukselApproach', 'ProposedApproach'};
nMethods   = numel(approaches);

W = 10;   % persistence window, frozen (plan section 4.2)

failures = {};
rows = struct('method', {}, 'mesh', {}, 'iterations', {}, ...
    'runtime_logging_off_s', {}, 'runtime_logging_on_s', {}, 'overhead_pct', {}, ...
    'trajectory_identical', {}, ...
    'rV_final', {}, 'rV_sustained_floor', {}, ...
    'd_inf_final', {}, 'd_inf_sustained_floor', {}, ...
    'd_inf_design_final', {}, 'grayness_final', {}, ...
    'move_active_frac_final', {}, 'k_cont', {});

for r = 1:size(meshes,1)
    for m = 1:nMethods
        cfg = baseCfg;
        cfg.domain.mesh.nelx = meshes(r,1);
        cfg.domain.mesh.nely = meshes(r,2);
        cfg.optimization.approach = approaches{m};
        meshStr = sprintf('%dx%d', meshes(r,1), meshes(r,2));

        fprintf('%-18s %-9s logging off ...\n', labels{m}, meshStr);
        cfgOff = cfg;
        cfgOff.benchmark.record_history = false;
        tOff = tic;
        [xOff, omOff, ~, nOff, ~, stOff, telOff] = run_topopt_from_json(cfgOff);
        runtimeOff = toc(tOff);

        fprintf('%-18s %-9s logging on  ...\n', labels{m}, meshStr);
        cfgOn = cfg;
        cfgOn.benchmark.record_history = true;
        tOn = tic;
        [xOn, omOn, ~, nOn, ~, stOn, telOn] = run_topopt_from_json(cfgOn);
        runtimeOn = toc(tOn);

        identical = isequaln(xOff, xOn) && isequaln(omOff, omOn) ...
            && isequaln(nOff, nOn) && isequaln(stOff, stOn);
        if ~identical
            failures{end+1} = sprintf('%s %s: logging changed the trajectory', ...
                labels{m}, meshStr); %#ok<SAGROW>
        end

        H = telOn.history;
        if isempty(H) || H.n == 0
            failures{end+1} = sprintf('%s %s: no history recorded', ...
                labels{m}, meshStr); %#ok<SAGROW>
            continue;
        end
        if H.n ~= nOn
            failures{end+1} = sprintf(['%s %s: history has %d rows for %d ' ...
                'iterations'], labels{m}, meshStr, H.n, nOn); %#ok<SAGROW>
        end
        % A NaN inside the criterion columns would force the acceptance
        % evaluator to special-case it, so the schema must not produce one
        % after the first increment.
        if any(~isfinite(H.d_inf(2:end))) || any(~isfinite(H.rV))
            failures{end+1} = sprintf('%s %s: NaN in d_inf or rV', ...
                labels{m}, meshStr); %#ok<SAGROW>
        end

        rows(end+1) = struct( ...
            'method', labels{m}, 'mesh', meshStr, 'iterations', nOn, ...
            'runtime_logging_off_s', runtimeOff, 'runtime_logging_on_s', runtimeOn, ...
            'overhead_pct', 100*(runtimeOn - runtimeOff)/runtimeOff, ...
            'trajectory_identical', identical, ...
            'rV_final', H.rV(end), ...
            'rV_sustained_floor', localSustainedFloor(H.rV, W), ...
            'd_inf_final', H.d_inf(end), ...
            'd_inf_sustained_floor', localSustainedFloor(H.d_inf, W), ...
            'd_inf_design_final', H.d_inf_design(end), ...
            'grayness_final', H.grayness(end), ...
            'move_active_frac_final', H.move_active_frac(end), ...
            'k_cont', H.k_cont); %#ok<SAGROW>
    end
end

sep = repmat('-', 1, 118);
fprintf('\nWP4 instrumentation invariance and overhead\n%s\n', sep);
fprintf('%-18s %-9s %8s %12s %12s %10s %10s\n', ...
    'Method', 'Mesh', 'iters', 'off (s)', 'on (s)', 'overhead', 'identical');
fprintf('%s\n', sep);
for i = 1:numel(rows)
    fprintf('%-18s %-9s %8d %12.2f %12.2f %9.1f%% %10d\n', ...
        rows(i).method, rows(i).mesh, rows(i).iterations, ...
        rows(i).runtime_logging_off_s, rows(i).runtime_logging_on_s, ...
        rows(i).overhead_pct, rows(i).trajectory_identical);
end

fprintf('\nSection 4.2.1 calibration data\n%s\n', sep);
fprintf('%-18s %-9s %12s %12s %12s %12s %10s %8s\n', ...
    'Method', 'Mesh', 'rV final', 'rV floor', 'd_inf final', 'd_inf floor', 'grayness', 'k_cont');
fprintf('%s\n', sep);
for i = 1:numel(rows)
    fprintf('%-18s %-9s %12.3e %12.3e %12.3e %12.3e %10.4f %8d\n', ...
        rows(i).method, rows(i).mesh, ...
        rows(i).rV_final, rows(i).rV_sustained_floor, ...
        rows(i).d_inf_final, rows(i).d_inf_sustained_floor, ...
        rows(i).grayness_final, rows(i).k_cont);
end
fprintf('%s\n', sep);
fprintf(['"floor" = the smallest value sustained over %d consecutive iterations,\n' ...
    'i.e. the tightest threshold that method could actually satisfy under the\n' ...
    'persistence rule. epsilon_V must be set from the WORST method''s rV floor.\n'], W);

fprintf('\nPhysical vs design increment (bearing on which field the gate uses)\n%s\n', sep);
fprintf('%-18s %-9s %14s %14s %10s\n', 'Method', 'Mesh', 'd_inf(xPhys)', 'd_inf(x)', 'ratio');
fprintf('%s\n', sep);
for i = 1:numel(rows)
    ratio = rows(i).d_inf_final / max(rows(i).d_inf_design_final, eps);
    fprintf('%-18s %-9s %14.3e %14.3e %10.2f\n', ...
        rows(i).method, rows(i).mesh, ...
        rows(i).d_inf_final, rows(i).d_inf_design_final, ratio);
end
fprintf('%s\n', sep);

passed = isempty(failures);
if passed
    fprintf('\nPASS: logging changed no trajectory and the history schema is complete.\n');
else
    fprintf('\nFAIL: %d problem(s):\n', numel(failures));
    for k = 1:numel(failures)
        fprintf('  - %s\n', failures{k});
    end
end

validation = struct();
validation.description = ['WP4 instrumentation invariance and section 4.2.1 ' ...
    'threshold calibration data'];
validation.persistence_window_W = W;
validation.rows = rows;
validation.failures = failures(:);
validation.pass = passed;

validationPath = fullfile(thisDir, 'history_logging_validation.json');
fid = fopen(validationPath, 'w');
assert(fid >= 0, 'validate_history_logging:OpenFailed', ...
    'Cannot open %s for writing.', validationPath);
fprintf(fid, '%s\n', jsonencode(validation));
fclose(fid);
fprintf('Validation record saved to: %s\n', validationPath);

assert(passed, 'validate_history_logging:NotInvariant', ...
    'History logging is not trajectory-invariant or the schema is incomplete.');

function v = localSustainedFloor(col, W)
% Smallest value x such that some window of W consecutive iterations has every
% entry <= x -- the tightest threshold this run could satisfy under the
% persistence rule of plan section 4.2.
col = col(:);
col = col(isfinite(col));
if numel(col) < W
    v = NaN;
    return;
end
best = inf;
for i = 1:(numel(col) - W + 1)
    best = min(best, max(col(i:i+W-1)));
end
v = best;
end
