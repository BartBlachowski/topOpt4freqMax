% =========================================================================
%  LEGACY CROSS-CODE TIMING DRIVER -- NOT REVIEWER EVIDENCE
% =========================================================================
%  This script prints a cross-implementation performance table (Olhoff /
%  Yuksel / Proposed).  It is NOT a stage of the revision campaign and its
%  output MUST NOT be used as reviewer evidence.
%
%  EXP1 (performance table) and EXP5 (scaling fit) were retired from the
%  reviewer evidence chain -- see
%  examples/Revision_v1/SCIENTIFIC_DECISION_EXP1_EXP5.md -- because the local
%  comparators are NOT faithful reference implementations: OlhoffApproach
%  performs a trial eigensolve after every MMA update (roughly doubling its
%  eigensolves per outer iteration) and adds a Heaviside projection, a
%  continuation schedule, and a grayness penalty absent from the published
%  method, while the local Yuksel-inspired code does not reproduce the
%  published iteration counts.  A timing, memory, speedup, or scaling
%  comparison against either measures implementation choices, not methods, at
%  ANY level of instrumentation.
%
%  The same prohibition therefore applies to THIS script.  Its numbers may not
%  populate a manuscript table or figure, a response-letter statement, or any
%  speedup / runtime-ratio / memory / scaling claim.  No quantitative
%  cross-code performance benchmark remains in the manuscript.
%
%  Retained for local exploration only.  The active campaign is
%  S1 -> EXP2 -> EXP2b -> EXP3 -> A4 (examples/Revision_v1).
% =========================================================================

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
];

% resolutions = [
%     600,  75;
%     800,  100;
% ];

nRes = size(resolutions, 1);

% Methods to compare. The Olhoff branch is the local OlhoffApproach
% implementation only. The unsuccessful OlhoffApproachExact reconstruction is
% archived diagnostic code and is excluded from production comparisons.
approaches   = {'Olhoff',                                  'Yuksel',                  'OurApproach'      };
methodLabels = {'OlhoffApproach (local Olhoff-inspired)', 'YukselApproach (local)', 'ProposedApproach' };
nMethods     = numel(approaches);

nSamples = 5;

% Storage: rows = resolutions, columns = methods
omega_all  = NaN(nRes, nMethods);
tIter_all  = NaN(nRes, nMethods);
nIter_all  = NaN(nRes, nMethods);
mem_all    = NaN(nRes, nMethods);

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

        for s = 1:nSamples
            fprintf('Running %-18s  mesh %4dx%-3d  sample %d/%d ...\n', ...
                methodLabels{m}, resolutions(r,1), resolutions(r,2), s, nSamples);

            [~, omega, tIter, nIter, mem] = run_topopt_from_json(data);

            omega_s(s) = omega(1);
            tIter_s(s) = tIter;
            nIter_s(s) = nIter;
            mem_s(s)   = mem;
        end

        omega_all(r, m) = mean(omega_s);
        tIter_all(r, m) = mean(tIter_s);
        nIter_all(r, m) = round(mean(nIter_s));
        mem_all(r, m)   = mean(mem_s);
    end
end

% Total run time = average iteration time × number of iterations
tTotal_all = tIter_all .* nIter_all;

% -------------------------------------------------------------------------
% Print performance table (mirrors Table 1 from Yuksel et al.)
% -------------------------------------------------------------------------
sepWidth = 107;
sep = repmat('-', 1, sepWidth);

fprintf('\n');
fprintf('Table 1. Run time comparison between methods for maximizing the first\n');
fprintf('natural frequency of a simply supported beam (8 m x 1 m, vf = 0.5).\n');
fprintf('Results averaged over %d runs.\n', nSamples);
fprintf('\n');
fprintf('%-20s  %-9s  %12s  %16s  %20s  %18s\n', ...
    'Method', 'Mesh size', 'Iterations', 'Run time (s)', 'Run time/iter (s/iter)', 'Max RAM (MB)');
fprintf('%s\n', sep);

for r = 1:nRes
    meshStr = sprintf('%dx%d', resolutions(r,1), resolutions(r,2));

    for m = 1:nMethods
        if isnan(tTotal_all(r,m))
            nIterStr  = 'N/A';
            timeStr   = 'N/A';
            iterStr   = 'N/A';
            ramStr    = 'N/A';
        else
            nIterStr = sprintf('%d',   nIter_all(r,m));
            timeStr  = sprintf('%.1f', tTotal_all(r,m));
            iterStr  = sprintf('%.2f', tIter_all(r,m));
            ramStr   = sprintf('%.0f', mem_all(r,m));
        end
        fprintf('%-20s  %-9s  %12s  %16s  %20s  %18s\n', ...
            methodLabels{m}, meshStr, nIterStr, timeStr, iterStr, ramStr);
    end

    if r < nRes
        fprintf('%s\n', sep);
    end
end

fprintf('%s\n', sep);
fprintf('\n');

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
displayNames = {'OlhoffApproach (local Olhoff-inspired)', ...
                'YukselApproach (local)', 'ProposedApproach'};
fid = fopen(csvPath, 'w');
fprintf(fid, 'Method,Mesh,Iterations,RunTime_s,RunTimePerIter_s,MaxRAM_MB\n');
for r = 1:nRes
    meshStr = sprintf('%dx%d', resolutions(r,1), resolutions(r,2));
    for m = 1:nMethods
        if isnan(tTotal_all(r,m))
            fprintf(fid, '%s,%s,,,\n', displayNames{m}, meshStr);
        else
            fprintf(fid, '%s,%s,%d,%.1f,%.2f,%.0f\n', ...
                displayNames{m}, meshStr, nIter_all(r,m), tTotal_all(r,m), ...
                tIter_all(r,m), mem_all(r,m));
        end
    end
end
fclose(fid);
fprintf('Table 1 saved to: %s\n', csvPath);
