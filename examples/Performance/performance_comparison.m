clear; clc;
close all;

% Load base beam configuration (simply supported beam, 8x1 m)
jsonPath = fullfile(fileparts(mfilename('fullpath')), 'performance_comparison.json');
data = jsondecode(fileread(jsonPath));

% Normalize spelling to avoid silent "optimisation" vs "optimization" bugs.
if isfield(data, 'optimisation') && ~isfield(data, 'optimization')
    data.optimization = data.optimisation;
end
if ~isfield(data, 'optimization')
    error('performance_comparison:MissingOptimizationField', ...
        'Missing "optimization" section in %s', jsonPath);
end

% Disable visualization and image saving for clean performance measurement
data.postprocessing.visualize_live    = false;
data.postprocessing.save_final_image  = false;
data.postprocessing.save_snapshot_image = false;

% Fix filter radius to 2 finite elements regardless of resolution.
% The base JSON uses physical units (0.04 m), which gives < 1 element at
% coarser meshes and causes checkerboard patterns.  Switching to 'element'
% units with radius = 2 keeps the filter consistent across all resolutions.
data.optimization.filter.radius       = 2;
data.optimization.filter.radius_units = 'element';

% -------------------------------------------------------------------------
% Resolutions: those from Table 1 in the paper (160x20, 240x30, 320x40)
% plus two additional ones (240x30 already in paper; 400x50 is new)
% -------------------------------------------------------------------------

resolutions = [
    160,  20;
    240,  30;
    320,  40;
    400,  50;
    480,  60;
    560,  70;
    640,  80;
    720,  90;
    800,  100;
];

% resolutions = [
%     160,  20;
%     240,  30;
%     320,  40;
%     400,  50;
% ];

% resolutions = [
%     600,  75;
%     800,  100;
% ];

nRes = size(resolutions, 1);

% Methods to compare
approaches   = {'Olhoff',          'Yuksel',         'OurApproach'       };
methodLabels = {'OlhoffApproach',  'YukselApproach', 'ProposedApproach'  };
nMethods     = numel(approaches);

nSamples = 1;

% Storage: rows = resolutions, columns = methods
omega_all  = NaN(nRes, nMethods);
tIter_all  = NaN(nRes, nMethods);
nIter_all  = NaN(nRes, nMethods);
mem_all    = NaN(nRes, nMethods);
nIterStage1_all = NaN(nRes, nMethods);
nIterStage2_all = NaN(nRes, nMethods);
tInit_all  = NaN(nRes, nMethods);
tLoop_all  = NaN(nRes, nMethods);
tPost_all  = NaN(nRes, nMethods);
tTotal_all = NaN(nRes, nMethods);
tReconstructed_all = NaN(nRes, nMethods);
stage1Share_all = NaN(nRes, nMethods);
stage2Share_all = NaN(nRes, nMethods);
stopReason_all = repmat({'N/A'}, nRes, nMethods);
finalMaxChange_all = NaN(nRes, nMethods);
finalRmsChange_all = NaN(nRes, nMethods);
finalRelObjectiveChange_all = NaN(nRes, nMethods);
finalGrayness_all = NaN(nRes, nMethods);
convergenceTolerance_all = NaN(nRes, nMethods);
runRecords = struct([]);
runRecordIndex = 0;

% -------------------------------------------------------------------------
% Run all (resolution × method) combinations, averaged over nSamples runs
% -------------------------------------------------------------------------
for r = 1:nRes
    data.domain.mesh.nelx = resolutions(r, 1);
    data.domain.mesh.nely = resolutions(r, 2);

    for m = 1:nMethods
        data.optimization.approach = approaches{m};

        omega_s = NaN(1, nSamples);
        tIter_s = NaN(1, nSamples);
        nIter_s = NaN(1, nSamples);
        mem_s   = NaN(1, nSamples);
        nIterStage1_s = NaN(1, nSamples);
        nIterStage2_s = NaN(1, nSamples);
        tInit_s = NaN(1, nSamples);
        tLoop_s = NaN(1, nSamples);
        tPost_s = NaN(1, nSamples);
        tTotal_s = NaN(1, nSamples);
        stopReason_s = repmat({'N/A'}, 1, nSamples);
        finalMaxChange_s = NaN(1, nSamples);
        finalRmsChange_s = NaN(1, nSamples);
        finalRelObjectiveChange_s = NaN(1, nSamples);
        finalGrayness_s = NaN(1, nSamples);
        convergenceTolerance_s = NaN(1, nSamples);

        for s = 1:nSamples
            fprintf('Running %-18s  mesh %4dx%-3d  sample %d/%d ...\n', ...
                methodLabels{m}, resolutions(r,1), resolutions(r,2), s, nSamples);

            totalWallTic = tic;
            [x, omega, tIter, nIter, mem, nIterStage, telemetry] = run_topopt_from_json(data);
            totalWallTime = toc(totalWallTic);

            omega_s(s) = omega(1);
            tIter_s(s) = tIter;
            nIter_s(s) = nIter;
            mem_s(s)   = mem;
            nIterStage1_s(s) = nIterStage.stage1;
            nIterStage2_s(s) = nIterStage.stage2;
            tInit_s(s) = telemetry.timing.initialization_time;
            tLoop_s(s) = telemetry.timing.optimization_loop_time;
            tPost_s(s) = telemetry.timing.postprocessing_time;
            tTotal_s(s) = totalWallTime;
            stopReason_s{s} = telemetry.stopping.stop_reason;
            finalMaxChange_s(s) = telemetry.stopping.final_max_density_change;
            finalRmsChange_s(s) = telemetry.stopping.final_rms_density_change;
            finalRelObjectiveChange_s(s) = telemetry.stopping.final_relative_objective_change;
            finalGrayness_s(s) = telemetry.stopping.final_grayness;
            convergenceTolerance_s(s) = telemetry.stopping.convergence_tolerance;

            if strcmpi(approaches{m}, 'Yuksel')
                assert(nIter == nIterStage.stage1 + nIterStage.stage2, ...
                    'performance_comparison:YukselIterationMismatch', ...
                    'Yuksel iter_total must equal iter_stage1 + iter_stage2.');
            end

            runRecordIndex = runRecordIndex + 1;
            runRecords(runRecordIndex) = make_run_record( ...
                display_method_name(approaches{m}), methodLabels{m}, ...
                resolutions(r,1), resolutions(r,2), s, ...
                x, omega, tIter, nIter, mem, nIterStage, telemetry, totalWallTime);
        end

        omega_all(r, m) = mean(omega_s);
        tIter_all(r, m) = mean(tIter_s);
        nIter_all(r, m) = round(mean(nIter_s));
        mem_all(r, m)   = mean(mem_s);
        nIterStage1_all(r, m) = round(mean(nIterStage1_s));
        nIterStage2_all(r, m) = round(mean(nIterStage2_s));
        tInit_all(r, m) = mean(tInit_s);
        tLoop_all(r, m) = mean(tLoop_s);
        tPost_all(r, m) = mean(tPost_s);
        tTotal_all(r, m) = mean(tTotal_s);
        tReconstructed_all(r, m) = tIter_all(r,m) * nIter_all(r,m);
        stopReason_all{r,m} = strjoin(unique(stopReason_s), '|');
        finalMaxChange_all(r,m) = mean(finalMaxChange_s);
        finalRmsChange_all(r,m) = mean(finalRmsChange_s);
        finalRelObjectiveChange_all(r,m) = mean(finalRelObjectiveChange_s);
        finalGrayness_all(r,m) = mean(finalGrayness_s);
        convergenceTolerance_all(r,m) = mean(convergenceTolerance_s);
        if strcmpi(approaches{m}, 'Yuksel')
            stage1Share_all(r,m) = 100 * nIterStage1_all(r,m) / nIter_all(r,m);
            stage2Share_all(r,m) = 100 * nIterStage2_all(r,m) / nIter_all(r,m);
        end
    end
end

% Preserve the legacy reconstructed timing separately.  tTotal_all is now
% the measured wall-clock time around the complete top-level solver call.

% -------------------------------------------------------------------------
% Print performance table (mirrors Table 1 from Yuksel et al.)
% -------------------------------------------------------------------------
sepWidth = 188;
sep = repmat('-', 1, sepWidth);

fprintf('\n');
fprintf('Table 1. Run time comparison between methods for maximizing the first\n');
fprintf('natural frequency of a simply supported beam (8 m x 1 m, vf = 0.5).\n');
fprintf('Results averaged over %d runs.\n', nSamples);
fprintf('\n');
fprintf('%-20s  %-9s  %10s  %10s  %10s  %9s  %9s  %10s  %10s  %12s  %12s  %12s  %12s\n', ...
    'Method', 'Mesh', 'iter_total', 'stage1', 'stage2', 'S1 share', 'S2 share', ...
    'init (s)', 'loop (s)', 'post (s)', 'wall (s)', 's/iter', 'Max RAM MB');
fprintf('%s\n', sep);

for r = 1:nRes
    meshStr = sprintf('%dx%d', resolutions(r,1), resolutions(r,2));

    for m = 1:nMethods
        if isnan(tTotal_all(r,m))
            nIterStr  = 'N/A';
            initTimeStr = 'N/A';
            loopTimeStr = 'N/A';
            postTimeStr = 'N/A';
            wallTimeStr = 'N/A';
            iterStr   = 'N/A';
            ramStr    = 'N/A';
        else
            nIterStr = sprintf('%d',   nIter_all(r,m));
            initTimeStr = sprintf('%.3f', tInit_all(r,m));
            loopTimeStr = sprintf('%.3f', tLoop_all(r,m));
            postTimeStr = sprintf('%.3f', tPost_all(r,m));
            wallTimeStr = sprintf('%.3f', tTotal_all(r,m));
            iterStr  = sprintf('%.2f', tIter_all(r,m));
            ramStr   = sprintf('%.0f', mem_all(r,m));
        end
        if isnan(nIterStage1_all(r,m))
            stage1Str = 'N/A';
            stage1ShareStr = 'N/A';
        else
            stage1Str = sprintf('%d', nIterStage1_all(r,m));
            stage1ShareStr = sprintf('%.1f%%', stage1Share_all(r,m));
        end
        if isnan(nIterStage2_all(r,m))
            stage2Str = 'N/A';
            stage2ShareStr = 'N/A';
        else
            stage2Str = sprintf('%d', nIterStage2_all(r,m));
            stage2ShareStr = sprintf('%.1f%%', stage2Share_all(r,m));
        end
        fprintf('%-20s  %-9s  %10s  %10s  %10s  %9s  %9s  %10s  %10s  %12s  %12s  %12s  %12s\n', ...
            methodLabels{m}, meshStr, nIterStr, stage1Str, stage2Str, ...
            stage1ShareStr, stage2ShareStr, initTimeStr, loopTimeStr, ...
            postTimeStr, wallTimeStr, iterStr, ramStr);
    end

    if r < nRes
        fprintf('%s\n', sep);
    end
end

fprintf('%s\n', sep);
fprintf('\n');
fprintf(['Timing definitions: init = configuration/solver setup; loop = timed optimization loops; ' ...
    'post = final modal analysis/reporting; wall = measured around run_topopt_from_json.\n']);
fprintf(['Iteration definitions: iter_total = all optimization iterations; iter_stage1/stage2 apply ' ...
    'only to Yuksel; shares are percentages of iter_total. N/A means not meaningful.\n\n']);

% -------------------------------------------------------------------------
% Print Table 1 in the paper's grouped-column layout (Mesh rows, one column
% group per method: t_iter, n_iter, T (s)), and export it as a LaTeX table.
% -------------------------------------------------------------------------
groupLabels = {'Olhoff--Du', 'Yuksel--Yilmaz', 'Proposed'};
paperTexPath = fullfile(fileparts(mfilename('fullpath')), 'table1_paper_style.tex');
print_table1_paper_style(resolutions, groupLabels, tIter_all, nIter_all, tTotal_all, paperTexPath);

% -------------------------------------------------------------------------
% Stopping diagnostics and explicit benchmark convergence parameters
% -------------------------------------------------------------------------
fprintf('\nStopping diagnostics (N/A means the metric is not meaningful or unavailable):\n');
fprintf('%s\n', sep);
fprintf('%-20s %-9s %-24s %12s %12s %12s %12s %12s\n', ...
    'Method', 'Mesh', 'Stop reason', 'max dx', 'RMS dx', 'rel obj/freq', 'grayness', 'tol used');
fprintf('%s\n', sep);
for r = 1:nRes
    meshStr = sprintf('%dx%d', resolutions(r,1), resolutions(r,2));
    for m = 1:nMethods
        fprintf('%-20s %-9s %-24s %12s %12s %12s %12s %12s\n', ...
            methodLabels{m}, meshStr, stopReason_all{r,m}, ...
            metric_string(finalMaxChange_all(r,m)), ...
            metric_string(finalRmsChange_all(r,m)), ...
            metric_string(finalRelObjectiveChange_all(r,m)), ...
            metric_string(finalGrayness_all(r,m)), ...
            metric_string(convergenceTolerance_all(r,m)));
    end
end
fprintf('%s\n', sep);
fprintf(['Yuksel Stage 1 configuration: stage1_tol=%.17g, stage1_max_iters=%d. ' ...
    'Stage 2 tolerance=%.17g.\n\n'], ...
    data.optimization.yuksel.stage1_tol, ...
    data.optimization.yuksel.stage1_max_iters, ...
    data.optimization.yuksel.stage2_tol);

% -------------------------------------------------------------------------
% Also print achieved natural frequencies for reference
% -------------------------------------------------------------------------
fprintf('Achieved first natural frequency omega_1 [rad/s]:\n');
fprintf('%s\n', sep);
fprintf('%-20s  %-9s  %16s\n', 'Method', 'Mesh size', 'omega_1 (rad/s)');
fprintf('%s\n', sep);

for r = 1:nRes
    meshStr = sprintf('%dx%d', resolutions(r,1), resolutions(r,2));
    for m = 1:nMethods
        if isnan(omega_all(r,m))
            omStr = 'N/A';
        else
            omStr = sprintf('%.1f', omega_all(r,m));
        end
        fprintf('%-20s  %-9s  %16s\n', methodLabels{m}, meshStr, omStr);
    end
    if r < nRes
        fprintf('%s\n', sep);
    end
end
fprintf('%s\n', sep);

% -------------------------------------------------------------------------
% Save Table 1 as CSV
% -------------------------------------------------------------------------
csvPath = fullfile(fileparts(mfilename('fullpath')), 'table1_performance.csv');
displayNames = {'Olhoff', 'Yuksel', 'Proposed'};
fid = fopen(csvPath, 'w');
assert(fid >= 0, 'performance_comparison:CsvOpenFailed', 'Cannot open %s for writing.', csvPath);
fprintf(fid, ['Method,Mesh,Iterations,IterStage1,IterStage2,RunTime_s,RunTimePerIter_s,MaxRAM_MB,' ...
    'iter_total,iter_stage1,iter_stage2,stage1_share_pct,stage2_share_pct,' ...
    'initialization_time_s,optimization_loop_time_s,postprocessing_time_s,total_wall_time_s,' ...
    'stop_reason,' ...
    'final_max_density_change,final_rms_density_change,final_relative_objective_change,' ...
    'final_grayness,convergence_tolerance_used\n']);
for r = 1:nRes
    meshStr = sprintf('%dx%d', resolutions(r,1), resolutions(r,2));
    for m = 1:nMethods
        if isnan(nIterStage1_all(r,m))
            stage1Csv = '';
        else
            stage1Csv = sprintf('%d', nIterStage1_all(r,m));
        end
        if isnan(nIterStage2_all(r,m))
            stage2Csv = '';
        else
            stage2Csv = sprintf('%d', nIterStage2_all(r,m));
        end
        fprintf(fid, ['%s,%s,%d,%s,%s,%.9g,%.9g,%.9g,%d,%s,%s,%s,%s,' ...
            '%.9g,%.9g,%.9g,%.9g,%s,' ...
            '%s,%s,%s,%s,%s\n'], ...
            displayNames{m}, meshStr, nIter_all(r,m), stage1Csv, stage2Csv, ...
            tReconstructed_all(r,m), tIter_all(r,m), mem_all(r,m), ...
            nIter_all(r,m), stage1Csv, stage2Csv, ...
            csv_metric(stage1Share_all(r,m)), csv_metric(stage2Share_all(r,m)), ...
            tInit_all(r,m), tLoop_all(r,m), tPost_all(r,m), tTotal_all(r,m), ...
            stopReason_all{r,m}, ...
            csv_metric(finalMaxChange_all(r,m)), csv_metric(finalRmsChange_all(r,m)), ...
            csv_metric(finalRelObjectiveChange_all(r,m)), csv_metric(finalGrayness_all(r,m)), ...
            csv_metric(convergenceTolerance_all(r,m)));
    end
end
fclose(fid);
fprintf('Table 1 saved to: %s\n', csvPath);

% Save per-sample records and the complete benchmark configuration as JSON.
jsonResultsPath = fullfile(fileparts(mfilename('fullpath')), 'benchmark_results.json');
benchmarkResults = struct();
benchmarkResults.metadata = struct( ...
    'benchmark_entry_point', 'examples/Performance/performance_comparison.m', ...
    'timing_note', ['total_wall_time is measured around the complete run_topopt_from_json call; ' ...
        'legacy_reconstructed_time is retained as average_iteration_time * iter_total.'], ...
    'na_representation', 'JSON null and CSV N/A mean not applicable or unavailable.', ...
    'iteration_fields', ['iter_total counts all optimization iterations; iter_stage1 and ' ...
        'iter_stage2 are Yuksel stages and sum to iter_total; shares are percentages.'], ...
    'diagnostics_enabled', data.benchmark.enable_diagnostics, ...
    'yuksel_stage1_tolerance', data.optimization.yuksel.stage1_tol, ...
    'yuksel_stage1_iteration_cap', data.optimization.yuksel.stage1_max_iters);
benchmarkResults.metadata.field_definitions = struct( ...
    'initialization_time_s', 'Configuration parsing, dispatch, and solver setup before optimization.', ...
    'optimization_loop_time_s', 'Time measured inside optimization loops; Yuksel is Stage 1 plus Stage 2.', ...
    'postprocessing_time_s', 'Final modal analysis and other work after optimization loops.', ...
    'total_wall_time_s', 'Caller-side wall time around the complete run_topopt_from_json call.', ...
    'iter_total', 'All executed optimization iterations.', ...
    'iter_stage1', 'Yuksel compliance-stage iterations; N/A for single-stage methods.', ...
    'iter_stage2', 'Yuksel inertial-stage iterations; N/A for single-stage methods.', ...
    'final_max_density_change', 'Final maximum absolute design-density change.', ...
    'final_rms_density_change', 'Final RMS design-density change when available.', ...
    'final_relative_objective_change', ...
        'Final relative change in the convergence objective (frequency for Olhoff).', ...
    'final_grayness', 'Mean 4*x*(1-x) of the final physical density field.', ...
    'convergence_tolerance', 'Numerical convergence tolerance actually used; criteria are unchanged.');
benchmarkResults.configuration = data;
benchmarkResults.runs = runRecords;
fid = fopen(jsonResultsPath, 'w');
assert(fid >= 0, 'performance_comparison:JsonOpenFailed', ...
    'Cannot open %s for writing.', jsonResultsPath);
fprintf(fid, '%s\n', jsonencode(benchmarkResults));
fclose(fid);
fprintf('Per-run benchmark results saved to: %s\n', jsonResultsPath);

% -------------------------------------------------------------------------
% Fit computational-complexity model T(N_e) = C * N_e^exp per method.
% N_e = nelx*nely is the number of finite elements in the mesh.
% -------------------------------------------------------------------------
Ne = resolutions(:,1) .* resolutions(:,2);
outDir = fileparts(mfilename('fullpath'));

% ---- Table 2: free fit -- both C and exp estimated by least-squares
% linear regression on log(T) = log(C) + exp*log(N_e). ----
[complexity_C, complexity_exp, complexity_R2, complexity_n] = ...
    fit_complexity_model(Ne, tTotal_all, 'free');

complexityCsvPath = fullfile(outDir, 'table1_complexity_fit.csv');
print_complexity_fit_table(methodLabels, displayNames, complexity_C, complexity_exp, ...
    complexity_R2, complexity_n, ...
    {'Table 2. Computational complexity fit  T(N_e) = C * N_e^exp', ...
     '(least-squares fit of log(T) vs log(N_e); N_e = nelx*nely)'}, ...
    complexityCsvPath);

% ---- Table 3: fixed-exponent fit -- exp is held fixed at an arbitrarily
% chosen value (default 1.5) and only C (the prefactor) is estimated by
% least squares. ----
fixedExp = 1.5;
[complexity_C_fixed, complexity_exp_fixed, complexity_R2_fixed, complexity_n_fixed] = ...
    fit_complexity_model(Ne, tTotal_all, 'fixed', fixedExp);

complexityCsvPathFixed = fullfile(outDir, 'table1_complexity_fit_fixedexp.csv');
print_complexity_fit_table(methodLabels, displayNames, complexity_C_fixed, complexity_exp_fixed, ...
    complexity_R2_fixed, complexity_n_fixed, ...
    {sprintf('Table 3. Fixed-exponent complexity fit  T(N_e) = C * N_e^%.2f', fixedExp), ...
     '(exponent held fixed; only C estimated by linear-space least squares on T, i.e. minimizing', ...
     'absolute run-time error sum((T - C*N_e^exp)^2); R^2 is on T, not log(T))'}, ...
    complexityCsvPathFixed);

% -------------------------------------------------------------------------
% Plot measured run times (Table 1 points) together with the fitted
% power-law curves, on both log-log and linear axes -- once for the free
% fit (Table 2), once for the fixed-exponent fit (Table 3).
% -------------------------------------------------------------------------
plot_table1_complexity(Ne, methodLabels, tTotal_all, complexity_C, complexity_exp, outDir);

plot_table1_complexity(Ne, methodLabels, tTotal_all, complexity_C_fixed, complexity_exp_fixed, ...
    outDir, 'table1_complexity_fit_fixedexp', ...
    sprintf('Fixed-exponent fit (C estimated only):  T(N_e) = C \\cdot N_e^{%.2f}', fixedExp));

function record = make_run_record(methodName, methodLabel, nelx, nely, sample, ...
        x, omega, tIter, nIter, mem, nIterStage, telemetry, totalWallTime)
    if isfinite(nIterStage.stage1) && nIter > 0
        stage1Share = 100 * nIterStage.stage1 / nIter;
        stage2Share = 100 * nIterStage.stage2 / nIter;
    else
        stage1Share = NaN;
        stage2Share = NaN;
    end
    record = struct();
    record.method = methodName;
    record.method_label = methodLabel;
    record.mesh = struct('nelx', nelx, 'nely', nely, 'elements', nelx*nely);
    record.sample = sample;
    record.iterations = struct( ...
        'iter_total', nIter, ...
        'iter_stage1', nIterStage.stage1, ...
        'iter_stage2', nIterStage.stage2, ...
        'stage1_share_pct', stage1Share, ...
        'stage2_share_pct', stage2Share);
    record.timing = struct( ...
        'initialization_time_s', telemetry.timing.initialization_time, ...
        'optimization_loop_time_s', telemetry.timing.optimization_loop_time, ...
        'postprocessing_time_s', telemetry.timing.postprocessing_time, ...
        'total_wall_time_s', totalWallTime, ...
        'runner_reported_total_wall_time_s', telemetry.timing.total_wall_time, ...
        'legacy_reconstructed_time_s', tIter*nIter, ...
        'average_iteration_time_s', tIter);
    record.stopping = telemetry.stopping;
    record.configuration = struct( ...
        'diagnostics_enabled', telemetry.diagnostics_enabled, ...
        'convergence_tolerance', telemetry.stopping.convergence_tolerance, ...
        'yuksel_stage1_max_iters', telemetry.yuksel.stage1_max_iters, ...
        'yuksel_stage1_tolerance', telemetry.yuksel.stage1_tolerance, ...
        'yuksel_stage2_tolerance', telemetry.yuksel.stage2_tolerance);
    record.results = struct( ...
        'objective_final', telemetry.objective_final, ...
        'objective_history_checksum', numeric_fingerprint(telemetry.objective_history), ...
        'final_frequencies_rad_s', omega(:)', ...
        'topology_checksum', numeric_fingerprint(x));
    record.max_ram_mb = mem;
end

function name = display_method_name(approach)
    if strcmpi(approach, 'OurApproach')
        name = 'Proposed';
    else
        name = approach;
    end
end

function value = numeric_fingerprint(x)
    x = double(x(:));
    if isempty(x)
        value = 'N/A';
        return;
    end
    weights = (1:numel(x))';
    value = sprintf('n=%d;sum=%.17g;weighted=%.17g;l2=%.17g', ...
        numel(x), sum(x), sum(weights.*x), norm(x));
end

function value = metric_string(x)
    if isempty(x) || ~isfinite(x)
        value = 'N/A';
    else
        value = sprintf('%.5e', x);
    end
end

function value = csv_metric(x)
    if isempty(x) || ~isfinite(x)
        value = 'N/A';
    else
        value = sprintf('%.17g', x);
    end
end
