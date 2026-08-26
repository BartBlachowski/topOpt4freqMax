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
% Du-Olhoff 2007 clean-room reproduction: settings that CANNOT come from the
% shared benchmark block.
%
% Four of the shared settings are not transferable to this solver, and using
% them produces numbers that are wrong rather than merely different.  Every
% override is scoped to this method alone; nothing here changes Yuksel or the
% Proposed method.
%
%  1. move limit.  The shared `optimization.move_limit` is 0.2, which is an
%     MMA/OC move limit.  In this solver the move limit is the trust region of
%     a sequential LINEAR program, and 0.2 destroys the design: measured at
%     160x20, r_min = 2 el, the run collapses to a disconnected island and
%     omega_1 ends at 2.9 rad/s instead of ~160 (NOTES.md section 8c documents
%     the same failure at move = 0.03).  The value used here, 0.005, is the
%     documented `fig3a_best` reproduction value.
%
%  2. outer-iteration budget.  While the LP solves successfully this method is
%     move-saturated: the step always travels the full move limit, so max|drho|
%     stays at `move` and the native stop test does not fire.  With the shared
%     max_iters = 10000 every mesh would run 10000 outer iterations.  The
%     budget used here, 1600, is the documented `fig3a_best` value and is what
%     produced the published reproduction.
%
%  3. void lower bound.  `void_material.rho_min` in the shared block is 1e-6,
%     a void MATERIAL DENSITY floor.  This solver's rho_min is a different
%     quantity: the DESIGN VARIABLE bound of Du & Olhoff (2007) eq. (7e), whose
%     value is 1e-3.  At 1e-6 the (K,M) pencil goes singular to working
%     precision (eigs reports RCOND = 1.6e-19); at 240x30 that produced
%     spurious omega_1 = 0 modes from outer iteration 101, melted the design to
%     volume 0.20, and ended the run on an infeasible LP that was misreported
%     as convergence.  See DIAGNOSTIC_REPRO2007_BENCHMARK.md.
%
%  4. outer tolerance.  The shared `convergence_tol` is 3e-3; this method's
%     documented outer tolerance is 1e-3.  Stated here rather than inherited,
%     so that the stopping point is not set by a value chosen for other methods.
%
% All of them are listed explicitly below, so the task file rather than the
% dispatcher is the record of what this method ran with.
%
% Reading the Olhoff column: while the LP succeeds, this method stops at its
% outer-iteration budget rather than on a convergence test, so its iteration
% count is a budget and t_iter and the scaling exponent are the meaningful
% entries -- not iter_total or wall time.  A stop reason other than
% `max_outer_iterations` means the LP failed and MUST be investigated, not
% read as convergence.
%
% Filter radius is deliberately NOT overridden: r_min = 2 elements is the
% benchmark's shared cross-resolution setting and the solver runs correctly at
% it (verified at 240x30: 1600 iterations, volume 0.5, no LP failures).  It is,
% however, not the radius that reproduces Fig. 3a (1.3 elements), so the
% omega_1 reported here for Olhoff is a valid operating point of the method and
% NOT the paper-reproduction figure.
data.optimization.repro2007 = struct( ...
    'support_type', 'SS', ...    % bc.supports are closest_point at mid-height
    'move',         0.005, ...   % documented fig3a_best value
    'max_outer',    1600, ...    % documented fig3a_best budget
    'rho_min',      1e-3, ...    % paper eq. (7e); NOT void_material.rho_min
    'tol_outer',    1e-3);       % documented fig3a_best value

% -------------------------------------------------------------------------
% Resolutions: those from Table 1 in the paper (160x20, 240x30, 320x40)
% plus two additional ones (240x30 already in paper; 400x50 is new)
% -------------------------------------------------------------------------

% resolutions = [
%     160,  20;
%     240,  30;
%     320,  40;
%     400,  50;
%     480,  60;
%     560,  70;
%     640,  80;
%     720,  90;
%     800,  100;
% ];

resolutions = [
    160,  20;
    240,  30;
    320,  40;
    400,  50;
];

% resolutions = [
%     600,  75;
%     800,  100;
% ];

nRes = size(resolutions, 1);

% Methods to compare.
%
% `approaches` is the method IDENTITY used for naming everywhere in the
% results -- console tables, CSV, JSON, LaTeX.  `solverApproaches` is the
% dispatch key handed to run_topopt_from_json.  The two are kept separate so
% that the implementation behind a column can be changed without renaming the
% column.
%
% The Olhoff column is now produced by the Du-Olhoff 2007 CLEAN-ROOM
% REPRODUCTION (Eq. 22 LP route) at Matlab/reproduction2007/, replacing the
% earlier analysis/OlhoffApproach/Matlab/topFreqOptimization_MMA.m call.  The
% reported name is unchanged; only the solver behind it moved.  See
% MIGRATION_REPRODUCTION2007_REPORT.md and Matlab/README.md.
approaches       = {'Olhoff',            'Yuksel',         'OurApproach'      };
solverApproaches = {'OlhoffDu2007Repro', 'Yuksel',         'OurApproach'      };
methodLabels     = {'OlhoffApproach',    'YukselApproach', 'ProposedApproach' };
nMethods         = numel(approaches);
assert(numel(solverApproaches) == nMethods, ...
    'performance_comparison:MethodTableMismatch', ...
    'approaches, solverApproaches and methodLabels must be the same length.');

nSamples = 1;

% Storage: rows = resolutions, columns = methods
omega_all  = NaN(nRes, nMethods);
tIter_all  = NaN(nRes, nMethods);
nIter_all  = NaN(nRes, nMethods);
mem_all    = NaN(nRes, nMethods);
nIterStage1_all = NaN(nRes, nMethods);
nIterStage2_all = NaN(nRes, nMethods);
nOuter_all = NaN(nRes, nMethods);
nInner_all = NaN(nRes, nMethods);
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
runRecords = struct('method', {}, 'method_label', {}, 'mesh', {}, 'sample', {}, ...
    'iterations', {}, 'timing', {}, 'stopping', {}, 'configuration', {}, ...
    'results', {}, 'max_ram_mb', {});
runRecordIndex = 0;

% -------------------------------------------------------------------------
% Run all (resolution × method) combinations, averaged over nSamples runs
% -------------------------------------------------------------------------
for r = 1:nRes
    data.domain.mesh.nelx = resolutions(r, 1);
    data.domain.mesh.nely = resolutions(r, 2);

    for m = 1:nMethods
        data.optimization.approach = solverApproaches{m};

        omega_s = NaN(1, nSamples);
        tIter_s = NaN(1, nSamples);
        nIter_s = NaN(1, nSamples);
        mem_s   = NaN(1, nSamples);
        nIterStage1_s = NaN(1, nSamples);
        nIterStage2_s = NaN(1, nSamples);
        nOuter_s = NaN(1, nSamples);
        nInner_s = NaN(1, nSamples);
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
            if isfield(telemetry, 'iterations')
                nOuter_s(s) = telemetry.iterations.outer;
                nInner_s(s) = telemetry.iterations.inner;
            end
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
        nOuter_all(r, m) = round(mean(nOuter_s));
        nInner_all(r, m) = round(mean(nInner_s));
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
sepWidth = 210;
sep = repmat('-', 1, sepWidth);

fprintf('\n');
fprintf('Table 1. Run time comparison between methods for maximizing the first\n');
fprintf('natural frequency of a simply supported beam (8 m x 1 m, vf = 0.5).\n');
fprintf('Results averaged over %d runs.\n', nSamples);
fprintf('\n');
fprintf('%-20s  %-9s  %10s  %8s  %10s  %10s  %10s  %9s  %9s  %10s  %10s  %12s  %12s  %12s  %12s\n', ...
    'Method', 'Mesh', 'iter_total', 'outer', 'inner', 'stage1', 'stage2', ...
    'S1 share', 'S2 share', ...
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
        % Outer/inner split: only methods with a genuine two-level loop report
        % it.  For the Olhoff column (Du-Olhoff 2007 reproduction) outer is the
        % Fig. 1 outer loop and inner the total subproblem solves.
        if isnan(nOuter_all(r,m))
            outerStr = 'N/A';
        else
            outerStr = sprintf('%d', nOuter_all(r,m));
        end
        if isnan(nInner_all(r,m))
            innerStr = 'N/A';
        else
            innerStr = sprintf('%d', nInner_all(r,m));
        end
        fprintf('%-20s  %-9s  %10s  %8s  %10s  %10s  %10s  %9s  %9s  %10s  %10s  %12s  %12s  %12s  %12s\n', ...
            methodLabels{m}, meshStr, nIterStr, outerStr, innerStr, ...
            stage1Str, stage2Str, ...
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
fprintf(['Iteration definitions: iter_total = all optimization iterations; outer/inner apply only to ' ...
    'methods with a two-level loop (Olhoff: Fig. 1 outer loop / subproblem (25) solves); ' ...
    'iter_stage1/stage2 apply only to Yuksel; shares are percentages of iter_total. ' ...
    'N/A means not meaningful.\n']);
fprintf(['Olhoff column: produced by the Du-Olhoff 2007 clean-room reproduction (Eq. 22 LP route, ' ...
    'Matlab/reproduction2007). It is move-saturated by construction, so it always stops at its ' ...
    'outer-iteration budget (%d) rather than on a convergence test; read t_iter and scaling, not ' ...
    'iter_total or wall time. Its move limit is %g, scoped to this method.\n\n'], ...
    data.optimization.repro2007.max_outer, data.optimization.repro2007.move);

% -------------------------------------------------------------------------
% Print Table 1 in the paper's grouped-column layout (Mesh rows, one column
% group per method: t_iter, n_iter, T (s)), and export it as a LaTeX table.
% -------------------------------------------------------------------------
groupLabels = {'Olhoff--Du', 'Yuksel--Yilmaz', 'Proposed'};
paperTexPath = fullfile(fileparts(mfilename('fullpath')), 'table1_paper_style.tex');
print_table1_paper_style(resolutions, groupLabels, tIter_all, nIter_all, tTotal_all, ...
    paperTexPath, nIterStage1_all, nIterStage2_all);

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
    'final_grayness,convergence_tolerance_used,' ...
    'outer_iterations,inner_iterations\n']);
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
        if isnan(nOuter_all(r,m))
            outerCsv = '';
        else
            outerCsv = sprintf('%d', nOuter_all(r,m));
        end
        if isnan(nInner_all(r,m))
            innerCsv = '';
        else
            innerCsv = sprintf('%d', nInner_all(r,m));
        end
        fprintf(fid, ['%s,%s,%d,%s,%s,%.9g,%.9g,%.9g,%d,%s,%s,%s,%s,' ...
            '%.9g,%.9g,%.9g,%.9g,%s,' ...
            '%s,%s,%s,%s,%s,%s,%s\n'], ...
            displayNames{m}, meshStr, nIter_all(r,m), stage1Csv, stage2Csv, ...
            tReconstructed_all(r,m), tIter_all(r,m), mem_all(r,m), ...
            nIter_all(r,m), stage1Csv, stage2Csv, ...
            csv_metric(stage1Share_all(r,m)), csv_metric(stage2Share_all(r,m)), ...
            tInit_all(r,m), tLoop_all(r,m), tPost_all(r,m), tTotal_all(r,m), ...
            stopReason_all{r,m}, ...
            csv_metric(finalMaxChange_all(r,m)), csv_metric(finalRmsChange_all(r,m)), ...
            csv_metric(finalRelObjectiveChange_all(r,m)), csv_metric(finalGrayness_all(r,m)), ...
            csv_metric(convergenceTolerance_all(r,m)), outerCsv, innerCsv);
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
        'iter_stage2 are Yuksel stages and sum to iter_total; shares are percentages; ' ...
        'outer and inner apply only to methods with a two-level loop (Olhoff: Fig. 1 outer ' ...
        'loop and subproblem (25) solves) and are null otherwise.'], ...
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
    'outer', 'Outer-loop iterations for two-level methods; N/A otherwise.', ...
    'inner', 'Total subproblem solves across all outer iterations; N/A otherwise.', ...
    'inner_solver', 'Subproblem solver used for the inner loop, where applicable.', ...
    'final_max_density_change', 'Final maximum absolute design-density change.', ...
    'final_rms_density_change', 'Final RMS design-density change when available.', ...
    'final_relative_objective_change', ...
        'Final relative change in the convergence objective (frequency for Olhoff).', ...
    'final_grayness', 'Mean 4*x*(1-x) of the final physical density field.', ...
    'convergence_tolerance', 'Numerical convergence tolerance actually used; criteria are unchanged.');
% Record which solver actually produced each named column.  The Olhoff column
% is the Du-Olhoff 2007 clean-room reproduction, not analysis/OlhoffApproach;
% without this the JSON would not say so.
benchmarkResults.metadata.method_dispatch = struct();
for m = 1:nMethods
    benchmarkResults.metadata.method_dispatch.( ...
        matlab.lang.makeValidName(approaches{m})) = solverApproaches{m};
end
benchmarkResults.metadata.olhoff_column_note = ['The Olhoff column is produced by ' ...
    'Matlab/reproduction2007 (Du-Olhoff 2007 clean-room reproduction, Eq. 22 LP route). ' ...
    'It is move-saturated by construction and always stops at its outer-iteration budget, ' ...
    'so iter_total is a fixed budget rather than a convergence result.'];
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
    outerIters = NaN;
    innerIters = NaN;
    innerSolver = 'N/A';
    if isfield(telemetry, 'iterations')
        outerIters  = telemetry.iterations.outer;
        innerIters  = telemetry.iterations.inner;
        innerSolver = telemetry.iterations.inner_solver;
    end
    record.iterations = struct( ...
        'iter_total', nIter, ...
        'iter_stage1', nIterStage.stage1, ...
        'iter_stage2', nIterStage.stage2, ...
        'stage1_share_pct', stage1Share, ...
        'stage2_share_pct', stage2Share, ...
        'outer', outerIters, ...
        'inner', innerIters, ...
        'inner_solver', innerSolver);
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
