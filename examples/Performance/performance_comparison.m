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

        for s = 1:nSamples
            fprintf('Running %-18s  mesh %4dx%-3d  sample %d/%d ...\n', ...
                methodLabels{m}, resolutions(r,1), resolutions(r,2), s, nSamples);

            [~, omega, tIter, nIter, mem, nIterStage] = run_topopt_from_json(data);

            omega_s(s) = omega(1);
            tIter_s(s) = tIter;
            nIter_s(s) = nIter;
            mem_s(s)   = mem;
            nIterStage1_s(s) = nIterStage.stage1;
            nIterStage2_s(s) = nIterStage.stage2;
        end

        omega_all(r, m) = mean(omega_s);
        tIter_all(r, m) = mean(tIter_s);
        nIter_all(r, m) = round(mean(nIter_s));
        mem_all(r, m)   = mean(mem_s);
        nIterStage1_all(r, m) = round(mean(nIterStage1_s));
        nIterStage2_all(r, m) = round(mean(nIterStage2_s));
    end
end

% Total run time = average iteration time × number of iterations
tTotal_all = tIter_all .* nIter_all;

% -------------------------------------------------------------------------
% Print performance table (mirrors Table 1 from Yuksel et al.)
% -------------------------------------------------------------------------
sepWidth = 131;
sep = repmat('-', 1, sepWidth);

fprintf('\n');
fprintf('Table 1. Run time comparison between methods for maximizing the first\n');
fprintf('natural frequency of a simply supported beam (8 m x 1 m, vf = 0.5).\n');
fprintf('Results averaged over %d runs.\n', nSamples);
fprintf('\n');
fprintf('%-20s  %-9s  %12s  %11s  %11s  %16s  %20s  %18s\n', ...
    'Method', 'Mesh size', 'Iterations', 'Iter Stage1', 'Iter Stage2', 'Run time (s)', 'Run time/iter (s/iter)', 'Max RAM (MB)');
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
        if isnan(nIterStage1_all(r,m))
            stage1Str = '-';
        else
            stage1Str = sprintf('%d', nIterStage1_all(r,m));
        end
        if isnan(nIterStage2_all(r,m))
            stage2Str = '-';
        else
            stage2Str = sprintf('%d', nIterStage2_all(r,m));
        end
        fprintf('%-20s  %-9s  %12s  %11s  %11s  %16s  %20s  %18s\n', ...
            methodLabels{m}, meshStr, nIterStr, stage1Str, stage2Str, timeStr, iterStr, ramStr);
    end

    if r < nRes
        fprintf('%s\n', sep);
    end
end

fprintf('%s\n', sep);
fprintf('\n');

% -------------------------------------------------------------------------
% Print Table 1 in the paper's grouped-column layout (Mesh rows, one column
% group per method: t_iter, n_iter, T (s)), and export it as a LaTeX table.
% -------------------------------------------------------------------------
groupLabels = {'Olhoff--Du', 'Yuksel--Yilmaz', 'Proposed'};
paperTexPath = fullfile(fileparts(mfilename('fullpath')), 'table1_paper_style.tex');
print_table1_paper_style(resolutions, groupLabels, tIter_all, nIter_all, tTotal_all, paperTexPath);

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
fprintf(fid, 'Method,Mesh,Iterations,IterStage1,IterStage2,RunTime_s,RunTimePerIter_s,MaxRAM_MB\n');
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
        if isnan(tTotal_all(r,m))
            fprintf(fid, '%s,%s,,%s,%s,,,\n', displayNames{m}, meshStr, stage1Csv, stage2Csv);
        else
            fprintf(fid, '%s,%s,%d,%s,%s,%.1f,%.2f,%.0f\n', ...
                displayNames{m}, meshStr, nIter_all(r,m), stage1Csv, stage2Csv, tTotal_all(r,m), ...
                tIter_all(r,m), mem_all(r,m));
        end
    end
end
fclose(fid);
fprintf('Table 1 saved to: %s\n', csvPath);

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
