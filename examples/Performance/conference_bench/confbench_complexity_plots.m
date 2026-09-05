function plotFiles = confbench_complexity_plots(cfg, records, scaling)
%CONFBENCH_COMPLEXITY_PLOTS  The four complexity-fit figures of the campaign.
%
%   plotFiles = CONFBENCH_COMPLEXITY_PLOTS(cfg, records, scaling) writes, into
%   cfg.outputDir:
%
%       table1_complexity_fit.png              free exponent,      log-log
%       table1_complexity_fit_linear.png       free exponent,      linear
%       table1_complexity_fit_fixedexp.png     exponent fixed 1.5, log-log
%       table1_complexity_fit_fixedexp_linear.png                  linear
%       table1_complexity_fit.csv              free-exponent fit table
%       table1_complexity_fit_fixedexp.csv     fixed-exponent fit table
%
%   WHICH ROWS ARE FITTED.  Exactly the rows CONFBENCH_SCALING_FIT accepts:
%   the ok runs.  A censored row (CAP_HIT) is measured data and is plotted, but
%   it is a LOWER BOUND on the run time that method would have needed, so
%   fitting through it biases the exponent downwards.  Such rows are drawn as
%   hollow markers labelled "(excluded from fit)".  Fitting the same rows as
%   confbench_scaling_fit is what keeps the exponent printed on these figures
%   identical to the one in BENCHMARK_NOTES.md and benchmark_results.json --
%   two different exponents for one method across two artifacts of one campaign
%   is a defect, not a nuance.
%
%   WHEN THE CAMPAIGN REFUSES A FIT.  The four figures are ALWAYS produced --
%   a preflight or smoke run gets them too, because measured time against mesh
%   size is exactly what you want to see from such a run.  What is NOT always
%   produced is a CURVE: if CONFBENCH_SCALING_FIT declined to fit (smoke run,
%   preflight, truncated budget), the measured points are plotted bare, the
%   fit tables record N/A, and the refusal reason is printed in the title.  A
%   figure carrying a fitted exponent for data the campaign rule forbids
%   fitting would launder that refusal, which is the one thing these artifacts
%   exist to prevent.
%
%   The figures are produced OUTSIDE every solver timing boundary, from the
%   already-recorded results, so generating them cannot perturb a measurement.
%
%   See also CONFBENCH_SCALING_FIT, FIT_COMPLEXITY_MODEL, PLOT_TABLE1_COMPLEXITY.

plotFiles = struct();

% Does the campaign rule permit a fitted curve on these data at all?
fitAllowed    = nargin >= 3 && ~isempty(scaling) && isstruct(scaling) && ...
                isfield(scaling, 'fitted') && scaling.fitted;
refusalReason = '';
if ~fitAllowed
    if nargin >= 3 && ~isempty(scaling) && isstruct(scaling) && isfield(scaling, 'reason') ...
            && ~isempty(scaling.reason)
        refusalReason = scaling.reason;
    else
        refusalReason = 'no scaling fit was performed for this run';
    end
end

% Canonical series order.  It fixes the colour and marker of each method
% (Du-Olhoff blue/o, Yuksel orange/s, Proposed green/^) so figures from
% different campaigns stay comparable at a glance.
canonicalKeys = {'olhoff', 'yuksel', 'proposed'};

present = unique({records.method_key}, 'stable');
keys    = canonicalKeys(ismember(canonicalKeys, present));
if isempty(keys)
    warning('confbench_complexity_plots:NoMethods', ...
        'No recognised methods in records; complexity plots skipped.');
    return
end

% records(:) forces a COLUMN struct array, so the cell array is nRuns x 1 and
% cell2mat stacks the meshes vertically into nRuns x 2.  On a row struct array
% it would concatenate them side by side into a single 1 x 2*nRuns row and every
% method would collapse to one mesh.
meshes = unique(cell2mat(arrayfun(@(r) r.mesh(:).', records(:), 'UniformOutput', false)), ...
    'rows', 'stable');
Ne = sort(meshes(:,1) .* meshes(:,2));

nRes     = numel(Ne);
nMethods = numel(keys);

tTotal_all   = NaN(nRes, nMethods);   % every measured point, plotted
tTotal_fit   = NaN(nRes, nMethods);   % only the points allowed into the fit
fitMask      = false(nRes, nMethods);
methodLabels = cell(1, nMethods);

for m = 1:nMethods
    methodLabels{m} = confbench_display_name(keys{m});
    sel = records(strcmp({records.method_key}, keys{m}));
    for i = 1:numel(sel)
        r = find(Ne == sel(i).mesh(1)*sel(i).mesh(2), 1);
        if isempty(r); continue; end
        t = sel(i).times.total_wall_time_s;
        tTotal_all(r, m) = t;
        if sel(i).ok
            tTotal_fit(r, m) = t;
            fitMask(r, m)    = true;
        end
    end
end

% ---- Free fit: both C and exp estimated on log(T) vs log(Ne) -------------
fixedExp = 1.5;
[C_free, exp_free, R2_free, n_free] = fit_complexity_model(Ne, tTotal_fit, 'free');
[C_fix,  exp_fix,  R2_fix,  n_fix ] = fit_complexity_model(Ne, tTotal_fit, 'fixed', fixedExp);

titleFree  = 'Computational complexity fit:  T(N_e) = C \cdot N_e^{exp}';
titleFixed = sprintf('Fixed-exponent fit (C estimated only):  T(N_e) = C \\cdot N_e^{%.2f}', fixedExp);
noteFree   = 'Fitted rows = the ok rows only, identical to BENCHMARK_NOTES.md.';
noteFixed  = 'Fitted rows = the ok rows only.';

if ~fitAllowed
    % Discard the numbers rather than draw them.  NaN suppresses the curve in
    % PLOT_TABLE1_COMPLEXITY and prints N/A in PRINT_COMPLEXITY_FIT_TABLE, so
    % the measured points survive and the fit does not.
    C_free(:) = NaN;  exp_free(:) = NaN;  R2_free(:) = NaN;
    C_fix(:)  = NaN;  exp_fix(:)  = NaN;  R2_fix(:)  = NaN;
    titleFree  = ['Measured run time (NO FIT: ', refusalReason, ')'];
    titleFixed = titleFree;
    noteFree   = ['NOT FITTED: ', refusalReason];
    noteFixed  = noteFree;
end

csvFree = fullfile(cfg.outputDir, 'table1_complexity_fit.csv');
print_complexity_fit_table(methodLabels, methodLabels, C_free, exp_free, R2_free, n_free, ...
    {'Computational complexity fit  T(N_e) = C * N_e^exp', ...
     '(least-squares fit of log(T) vs log(N_e); N_e = nelx*nely)', noteFree}, csvFree);

csvFixed = fullfile(cfg.outputDir, 'table1_complexity_fit_fixedexp.csv');
print_complexity_fit_table(methodLabels, methodLabels, C_fix, exp_fix, R2_fix, n_fix, ...
    {sprintf('Fixed-exponent complexity fit  T(N_e) = C * N_e^%.2f', fixedExp), ...
     '(exponent held fixed; only C estimated by linear-space least squares on T, i.e. minimizing', ...
     'absolute run-time error sum((T - C*N_e^exp)^2); R^2 is on T, not log(T))', noteFixed}, csvFixed);

% ---- The four figures ---------------------------------------------------
plot_table1_complexity(Ne, methodLabels, tTotal_all, C_free, exp_free, ...
    cfg.outputDir, 'table1_complexity_fit', titleFree, fitMask);

plot_table1_complexity(Ne, methodLabels, tTotal_all, C_fix, exp_fix, ...
    cfg.outputDir, 'table1_complexity_fit_fixedexp', titleFixed, fitMask);

plotFiles.complexity_fit_png        = fullfile(cfg.outputDir, 'table1_complexity_fit.png');
plotFiles.complexity_fit_lin_png    = fullfile(cfg.outputDir, 'table1_complexity_fit_linear.png');
plotFiles.complexity_fixed_png      = fullfile(cfg.outputDir, 'table1_complexity_fit_fixedexp.png');
plotFiles.complexity_fixed_lin_png  = fullfile(cfg.outputDir, 'table1_complexity_fit_fixedexp_linear.png');
plotFiles.complexity_fit_csv        = csvFree;
plotFiles.complexity_fixed_csv      = csvFixed;

% The free-exponent figure and the campaign's own scaling table must agree.
% They are computed by different code paths on purpose, so disagreement is a
% real defect and is reported rather than left for a reader to notice.
if fitAllowed
    for m = 1:nMethods
        k = find(strcmp({scaling.methods.method}, methodLabels{m}), 1);
        if isempty(k) || ~isfinite(scaling.methods(k).p) || ~isfinite(exp_free(m))
            continue
        end
        if abs(scaling.methods(k).p - exp_free(m)) > 1e-6
            warning('confbench_complexity_plots:ExponentMismatch', ...
                ['%s: exponent on the figure (%.4f) differs from the campaign ' ...
                 'scaling table (%.4f).  The two fits no longer see the same rows.'], ...
                methodLabels{m}, exp_free(m), scaling.methods(k).p);
        end
    end
end
end
