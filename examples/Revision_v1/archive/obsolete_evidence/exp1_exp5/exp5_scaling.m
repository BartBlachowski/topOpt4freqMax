function results = exp5_scaling(perfResults, outDir)
%EXP5_SCALING  Per-iteration scaling analysis (log-log) for all three methods.
%
%   Addresses reviewer demands:
%     M4 (mn8) : Support or correct the O(n_e^1.3) per-iteration scaling claim
%                (Section 5.2) with a log-log plot or revised exponent values.
%
%   If PERFRESULTS is provided (output of exp1_perf_table), the function uses
%   those timing data.  Otherwise it uses the Table 1 paper values as a
%   reference for fitting, and optionally runs new timing experiments.
%
%   For each method, fits a power law  t_iter = C * n_e^beta  to the
%   per-iteration timing data across mesh resolutions and reports beta.
%
%   Usage:
%     results = exp5_scaling();                  % uses paper Table 1 data
%     results = exp5_scaling(perfResults);       % uses exp1 measured data
%     results = exp5_scaling([], '/out/path');

if nargin < 1,  perfResults = []; end
if nargin < 2 || isempty(outDir)
    outDir = fileparts(mfilename('fullpath'));
end

scriptDir = fileparts(mfilename('fullpath'));
localEnsurePaths(scriptDir);

fprintf('\n=== EXP5: Per-iteration scaling analysis ===\n\n');

meshSizes = [160, 20; 240, 30; 320, 40; 400, 50];
nElem     = meshSizes(:,1) .* meshSizes(:,2);
meshLabel = {'160x20','240x30','320x40','400x50'};

if ~isempty(perfResults) && isstruct(perfResults) && isfield(perfResults,'tIter_mean')
    fprintf('Using measured timing data from exp1_perf_table.\n\n');
    tIter_data = perfResults.tIter_mean;
    methodLabels = perfResults.methodLabels;
    nMethods = numel(methodLabels);
else
    fprintf('Using paper Table 1 values (reported averages over 10 runs).\n');
    fprintf('Source: main.tex Table 1 -- per-iteration times.\n\n');

    tIter_data = [
        0.09,  0.02,  0.02;  % 160x20:  Olhoff, Yuksel, Proposed
        0.20,  0.03,  0.04;  % 240x30
        0.39,  0.05,  0.08;  % 320x40
        0.64,  0.08,  0.12;  % 400x50
    ];
    methodLabels = {'OlhoffApproach','YukselApproach','ProposedApproach'};
    nMethods = numel(methodLabels);
end

beta_all = NaN(nMethods, 1);
C_all    = NaN(nMethods, 1);
R2_all   = NaN(nMethods, 1);

fprintf('%-22s  %8s  %12s  %8s\n', 'Method', 'beta', 'C (prefactor)', 'R^2');
sep = repmat('-', 1, 60);
fprintf('%s\n', sep);

for m = 1:nMethods
    tVec = tIter_data(:, m);
    valid = isfinite(tVec) & tVec > 0;
    if sum(valid) < 2
        fprintf('%-22s  insufficient data\n', methodLabels{m});
        continue;
    end
    logN = log(nElem(valid));
    logT = log(tVec(valid));

    A = [ones(sum(valid),1), logN];
    b = logT;
    coef = A \ b;
    logC = coef(1);
    beta = coef(2);
    Cpre = exp(logC);

    yPred = A * coef;
    SS_res = sum((b - yPred).^2);
    SS_tot = sum((b - mean(b)).^2);
    R2 = 1 - SS_res / max(SS_tot, eps);

    beta_all(m) = beta;
    C_all(m)    = Cpre;
    R2_all(m)   = R2;

    fprintf('%-22s  %8.3f  %12.2e  %8.4f\n', methodLabels{m}, beta, Cpre, R2);
end
fprintf('%s\n\n', sep);

localPrintScalingInterpretation(methodLabels, beta_all, R2_all);

localPlotLogLog(nElem, tIter_data, methodLabels, beta_all, C_all, meshLabel, outDir);

results = struct( ...
    'meshSizes',    meshSizes, ...
    'nElem',        nElem, ...
    'methodLabels', {methodLabels}, ...
    'tIter_data',   tIter_data, ...
    'beta',         beta_all, ...
    'C_prefactor',  C_all, ...
    'R2',           R2_all);

matFile = fullfile(outDir, 'exp5_scaling_results.mat');
save(matFile, 'results');
fprintf('Results saved to: %s\n', matFile);
end

% =========================================================================
function localPrintScalingInterpretation(methodLabels, beta, R2)
fprintf('Interpretation:\n');
fprintf('  The paper (Section 5.2) claims both methods scale as O(n_e^1.3).\n');
fprintf('  Measured exponents from log-log regression:\n');
for m = 1:numel(methodLabels)
    if ~isnan(beta(m))
        fprintf('    %-22s  beta=%.3f  (R^2=%.4f)\n', methodLabels{m}, beta(m), R2(m));
    end
end
fprintf('\n');
fprintf('  The expected exponent for a direct sparse solver on quasi-banded meshes\n');
fprintf('  is approximately O(n_e^1.5) in 2D (Cholesky).  Values near 1.0-1.2\n');
fprintf('  are consistent with efficient sparse factorization of rectangular meshes.\n');
fprintf('  If measured betas differ significantly from 1.3, the Section 5.2 claim\n');
fprintf('  should be corrected or the log-log figure (saved to outDir) should be\n');
fprintf('  included in the paper to show the actual scaling.\n\n');
end

% =========================================================================
function localPlotLogLog(nElem, tIter_data, methodLabels, beta, C, meshLabel, outDir)
colors  = {'b', 'r', 'g'};
markers = {'o', 's', '^'};
nMethods = numel(methodLabels);

fig = figure('Visible','off', 'Position', [100 100 600 450]);
hold on; grid on;

neRange = linspace(min(nElem)*0.8, max(nElem)*1.2, 100);
for m = 1:nMethods
    tVec  = tIter_data(:, m);
    valid = isfinite(tVec) & tVec > 0;
    if ~any(valid), continue; end

    h = loglog(nElem(valid), tVec(valid), ...
        [colors{m}, markers{m}], ...
        'MarkerSize', 8, 'LineWidth', 1.5, ...
        'DisplayName', methodLabels{m});
    if ~isnan(beta(m)) && ~isnan(C(m))
        tFit = C(m) * neRange .^ beta(m);
        loglog(neRange, tFit, [colors{m}, '--'], ...
            'LineWidth', 1, ...
            'DisplayName', sprintf('%s fit: \\beta=%.2f', methodLabels{m}, beta(m)));
    end
end

xlabel('Number of elements n_e', 'FontSize', 12);
ylabel('Per-iteration time (s)', 'FontSize', 12);
title('Per-iteration scaling: t_{iter} \propto n_e^{\beta}', 'FontSize', 13);
legend('Location', 'northwest', 'FontSize', 9);
xticks(nElem);
xticklabels(meshLabel);

textStr = 'Fitted exponents \beta:';
for m = 1:nMethods
    if ~isnan(beta(m))
        textStr = [textStr, sprintf('\n  %s: %.2f', methodLabels{m}, beta(m))]; %#ok<AGROW>
    end
end
annotation('textbox', [0.62, 0.18, 0.35, 0.20], 'String', textStr, ...
    'FontSize', 9, 'BackgroundColor', 'white', 'EdgeColor', 'black');

imgFile = fullfile(outDir, 'exp5_scaling_loglog.png');
saveas(fig, imgFile);
figFile = fullfile(outDir, 'exp5_scaling_loglog.fig');
saveas(fig, figFile);
close(fig);
fprintf('Log-log scaling plot saved to: %s\n', imgFile);
end

% =========================================================================
function localEnsurePaths(scriptDir)
repoRoot = fileparts(fileparts(scriptDir));
toolsDir = fullfile(repoRoot, 'tools', 'Matlab');
if exist(toolsDir,'dir')==7, addpath(toolsDir); end
end
