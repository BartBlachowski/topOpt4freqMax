clear; clc; close all;
% REGENERATE_FROM_CSV  Rebuild Table 1 (paper-style), the complexity-fit
% tables (free exponent + fixed exponent) and their log-log/linear plots
% from the already-saved table1_performance.csv, without re-running any
% simulations.

thisDir = fileparts(mfilename('fullpath'));
csvPath = fullfile(thisDir, 'table1_performance.csv');

T = readtable(csvPath);

methods = unique(T.Method, 'stable');
meshes  = unique(T.Mesh,   'stable');
nMethods = numel(methods);
nRes     = numel(meshes);

resolutions = zeros(nRes, 2);
Ne = zeros(nRes, 1);
for r = 1:nRes
    parts = split(string(meshes{r}), 'x');
    resolutions(r, :) = [str2double(parts(1)), str2double(parts(2))];
    Ne(r) = resolutions(r,1) * resolutions(r,2);
end

tIter_all  = NaN(nRes, nMethods);
nIter_all  = NaN(nRes, nMethods);
tTotal_all = NaN(nRes, nMethods);
nIterStage1_all = NaN(nRes, nMethods);
nIterStage2_all = NaN(nRes, nMethods);
for r = 1:nRes
    for m = 1:nMethods
        mask = strcmp(T.Method, methods{m}) & strcmp(T.Mesh, meshes{r});
        if any(mask) && ~isnan(T.RunTime_s(mask))
            tTotal_all(r, m) = T.RunTime_s(mask);
            tIter_all(r, m)  = T.RunTimePerIter_s(mask);
            nIter_all(r, m)  = T.Iterations(mask);
            nIterStage1_all(r, m) = T.IterStage1(mask);
            nIterStage2_all(r, m) = T.IterStage2(mask);
        end
    end
end

methodLabels = methods';   % {'Olhoff', 'Yuksel', 'Proposed'} as stored in the CSV
displayNames = methodLabels;
groupLabels  = {'Olhoff--Du', 'Yuksel--Yilmaz', 'Proposed'};

% -------------------------------------------------------------------------
% Table 1 (paper style)
% -------------------------------------------------------------------------
paperTexPath = fullfile(thisDir, 'table1_paper_style.tex');
print_table1_paper_style(resolutions, groupLabels, tIter_all, nIter_all, tTotal_all, ...
    paperTexPath, nIterStage1_all, nIterStage2_all);

% -------------------------------------------------------------------------
% Table 2: free fit -- both C and exp estimated by least squares.
% -------------------------------------------------------------------------
[complexity_C, complexity_exp, complexity_R2, complexity_n] = ...
    fit_complexity_model(Ne, tTotal_all, 'free');

complexityCsvPath = fullfile(thisDir, 'table1_complexity_fit.csv');
print_complexity_fit_table(methodLabels, displayNames, complexity_C, complexity_exp, ...
    complexity_R2, complexity_n, ...
    {'Table 2. Computational complexity fit  T(N_e) = C * N_e^exp', ...
     '(least-squares fit of log(T) vs log(N_e); N_e = nelx*nely)'}, ...
    complexityCsvPath);

% -------------------------------------------------------------------------
% Table 3: fixed-exponent fit -- exp held fixed (arbitrary, default 1.5),
% only C estimated by least squares.
% -------------------------------------------------------------------------
fixedExp = 1.5;
[complexity_C_fixed, complexity_exp_fixed, complexity_R2_fixed, complexity_n_fixed] = ...
    fit_complexity_model(Ne, tTotal_all, 'fixed', fixedExp);

complexityCsvPathFixed = fullfile(thisDir, 'table1_complexity_fit_fixedexp.csv');
print_complexity_fit_table(methodLabels, displayNames, complexity_C_fixed, complexity_exp_fixed, ...
    complexity_R2_fixed, complexity_n_fixed, ...
    {sprintf('Table 3. Fixed-exponent complexity fit  T(N_e) = C * N_e^%.2f', fixedExp), ...
     '(exponent held fixed; only C estimated by linear-space least squares on T, i.e. minimizing', ...
     'absolute run-time error sum((T - C*N_e^exp)^2); R^2 is on T, not log(T))'}, ...
    complexityCsvPathFixed);

% -------------------------------------------------------------------------
% Plots: log-log + linear, once per fit.
% -------------------------------------------------------------------------
plot_table1_complexity(Ne, methodLabels, tTotal_all, complexity_C, complexity_exp, thisDir);

plot_table1_complexity(Ne, methodLabels, tTotal_all, complexity_C_fixed, complexity_exp_fixed, ...
    thisDir, 'table1_complexity_fit_fixedexp', ...
    sprintf('Fixed-exponent fit (C estimated only):  T(N_e) = C \\cdot N_e^{%.2f}', fixedExp));
