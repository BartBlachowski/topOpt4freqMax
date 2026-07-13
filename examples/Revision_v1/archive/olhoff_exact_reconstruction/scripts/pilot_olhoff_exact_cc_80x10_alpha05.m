function result = pilot_olhoff_exact_cc_80x10_alpha05(outDir)
%PILOT_OLHOFF_EXACT_CC_80X10_ALPHA05  Stabilized OlhoffExact pilot.
%
% Scope: MATLAB only; OlhoffApproachExact only.  This pilot changes one
% stabilization parameter relative to the saved undamped 80x10 pilot:
% cfg.alpha = 0.5 in the outer update rho_new = rho + alpha*Delta_rho.

warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:singularMatrix');

if nargin < 1 || isempty(outDir)
    scriptDir = fileparts(mfilename('fullpath'));
    outDir = fullfile(scriptDir, 'output', 'pilot_olhoff_exact_cc_80x10_alpha05');
end
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));
addpath(fullfile(repoRoot, 'analysis', 'OlhoffApproachExact', 'Matlab'));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));

logPath = fullfile(outDir, 'pilot_run.log');
diary(logPath);
cleanupD = onCleanup(@() diary('off')); %#ok<NASGU>

fprintf('Stabilized OlhoffApproachExact CC 80x10 alpha=0.5 pilot started: %s\n', ...
    char(datetime('now', 'TimeZone', 'UTC')));
fprintf('Scope: MATLAB only, OlhoffApproachExact only, no manuscript edits, no full revision experiments.\n\n');

baselineDir = fullfile(scriptDir, 'output', 'pilot_olhoff_exact_cc_80x10');
baselineConvPath = fullfile(baselineDir, 'convergence_history.csv');
baselineS1Path = fullfile(baselineDir, 's1_mode_summary.json');
baselineResultPath = fullfile(baselineDir, 'pilot_result.json');
baselineLogPath = fullfile(baselineDir, 'pilot_run.log');

cfgUndamped = localBaseCfg();
cfgDamped = cfgUndamped;
cfgDamped.alpha = 0.5;

settingsCheck = localVerifySingleAlphaChange(cfgUndamped, cfgDamped);
localWriteJson(fullfile(outDir, 'settings_verification.json'), settingsCheck);
if ~settingsCheck.pass
    error('PilotOlhoffAlpha05:SettingsChanged', ...
        'Settings verification failed: more than alpha differs.');
end

fprintf('Settings verification passed: only alpha changed (%.3g -> %.3g).\n', ...
    cfgUndamped.alpha, cfgDamped.alpha);
fprintf('Running CC %dx%d, volfrac=%.3g, mass=%s, p=%.3g, rmin_elem=%.3g, outer_max_iter=%d.\n\n', ...
    cfgDamped.nelx, cfgDamped.nely, cfgDamped.volfrac, cfgDamped.mass_mode, ...
    cfgDamped.penal, cfgDamped.rmin_elem, cfgDamped.outer_max_iter);

tic;
[rho_final, hist] = topopt_freq_exact(cfgDamped);
elapsed_s = toc;
fprintf('\nOptimization finished in %.1f s, outer_iters=%d.\n', elapsed_s, hist.outer_iters);

[model, omega, freqHz, Phi, K, M, free] = localFinalModes(rho_final, cfgDamped, 10);
fprintf('\nFinal first 6 eigenfrequencies:\n');
fprintf('  mode  omega(rad/s)       Hz\n');
for k = 1:6
    fprintf('  %4d  %12.6f  %12.6f\n', k, omega(k), freqHz(k));
end

localSaveTopology(outDir, rho_final, cfgDamped, omega(1));
convT = localWriteConvergence(outDir, hist, cfgDamped);
localSaveModeShapes(outDir, Phi, rho_final, cfgDamped, omega, 6);

fprintf('\nRunning S1 diagnosis for first 10 modes.\n');
s1 = localS1Diagnose(rho_final, Phi, omega, freqHz, K, M, free, model, cfgDamped);
localWriteJson(fullfile(outDir, 's1_mode_summary.json'), s1);
writetable(struct2table(s1.modes), fullfile(outDir, 's1_modes.csv'));

baseline = localLoadBaseline(baselineConvPath, baselineS1Path, baselineResultPath, baselineLogPath);
comparison = localCompareAgainstBaseline(baseline, convT, s1, hist, cfgDamped.outer_tol);
classification = localClassify(hist, cfgDamped, s1, comparison);

result = struct();
result.study = 'Stabilized OlhoffApproachExact CC 80x10 alpha=0.5 pilot';
result.created_utc = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'));
result.scope = 'Single alpha-damping pilot; OlhoffApproachExact only';
result.changed_parameter = struct('name', 'alpha', 'baseline', cfgUndamped.alpha, 'pilot', cfgDamped.alpha);
result.settings_verification = settingsCheck;
result.mesh = struct('nelx', cfgDamped.nelx, 'nely', cfgDamped.nely, ...
    'L', cfgDamped.L, 'H', cfgDamped.H, 'N_e', cfgDamped.nelx * cfgDamped.nely);
result.config = cfgDamped;
result.elapsed_s = elapsed_s;
result.outer_iters = hist.outer_iters;
result.capped = hist.outer_iters >= cfgDamped.outer_max_iter;
result.final_design_change = hist.drho_norm(end);
result.outer_tol = cfgDamped.outer_tol;
result.final_volume = mean(rho_final);
result.final_N = hist.final_N;
result.omega_final = omega(:)';
result.first_6_omega_rad_s = omega(1:6)';
result.first_6_frequency_hz = freqHz(1:6)';
result.s1_overall = s1.overall;
result.topology_connectedness = s1.components;
result.comparison_vs_undamped = comparison;
result.classification = classification;

save(fullfile(outDir, 'stabilized_pilot_result.mat'), ...
    'result', 'cfgDamped', 'cfgUndamped', 'rho_final', 'hist', 'omega', 'freqHz', 's1', 'comparison');
localWriteJson(fullfile(outDir, 'stabilized_pilot_result.json'), result);
localWriteSummary(fullfile(outDir, 'stabilized_pilot_summary.md'), result, s1, comparison);
localWriteManifest(fullfile(outDir, 'manifest.json'), outDir, result);

fprintf('\nClassification: %s\n', classification.label);
fprintf('Summary: %s\n', fullfile(outDir, 'stabilized_pilot_summary.md'));
diary('off');
end

function cfg = localBaseCfg()
cfg = struct();
cfg.L = 8.0;
cfg.H = 1.0;
cfg.nelx = 80;
cfg.nely = 10;
cfg.E0 = 1e7;
cfg.nu = 0.3;
cfg.rho0 = 1.0;
cfg.t = 1.0;
cfg.volfrac = 0.5;
cfg.rho_min = 1e-3;
cfg.penal = 3.0;
cfg.mass_mode = 'du2007_c1';
cfg.rmin_elem = 2.5;
cfg.sensitivity_filter = true;
cfg.support_type = 'CC';
cfg.n_target = 1;
cfg.n_modes = 10;
cfg.mult_tol = 1e-3;
cfg.outer_max_iter = 400;
cfg.outer_tol = 1e-3;
cfg.inner_max_iter = 30;
cfg.inner_tol = 1e-4;
cfg.move_lim = Inf;
cfg.outer_move = 0.2;
cfg.acceptance_check = false;
cfg.alpha = 1.0;
cfg.verbose = true;
end

function check = localVerifySingleAlphaChange(a, b)
fields = sort(fieldnames(a));
diffs = {};
for i = 1:numel(fields)
    f = fields{i};
    if ~isfield(b, f)
        diffs{end+1} = sprintf('%s missing in damped config', f); %#ok<AGROW>
        continue
    end
    if strcmp(f, 'alpha')
        continue
    end
    if ~isequaln(a.(f), b.(f))
        diffs{end+1} = f; %#ok<AGROW>
    end
end
extra = setdiff(fieldnames(b), fieldnames(a));
for i = 1:numel(extra)
    diffs{end+1} = sprintf('%s extra in damped config', extra{i}); %#ok<AGROW>
end
check = struct();
check.pass = isempty(diffs) && a.alpha == 1.0 && b.alpha == 0.5;
check.baseline_alpha = a.alpha;
check.pilot_alpha = b.alpha;
check.differences_other_than_alpha = diffs;
check.unchanged = rmfield(a, 'alpha');
end

function [model, omega, freqHz, Phi, K, M, free] = localFinalModes(rho, cfg, nModes)
dx = cfg.L / cfg.nelx;
dy = cfg.H / cfg.nely;
nEl = cfg.nelx * cfg.nely;
nDof = 2 * (cfg.nelx + 1) * (cfg.nely + 1);

[Ke_star, Me_star] = fe_q4_exact(cfg.nu, cfg.t, dx, dy);
Ke_phys = cfg.E0 * Ke_star;
Me_phys = cfg.rho0 * Me_star;

nodeNrs = reshape(1:(cfg.nelx+1)*(cfg.nely+1), cfg.nely+1, cfg.nelx+1);
cVec = reshape(2*nodeNrs(1:cfg.nely,1:cfg.nelx)+1, nEl, 1);
cMat = [cVec, cVec+1, cVec+2*cfg.nely+2, cVec+2*cfg.nely+3, ...
        cVec+2*cfg.nely, cVec+2*cfg.nely+1, cVec-2, cVec-1];

[Il, Jl] = find(tril(ones(8)));
iK = reshape(cMat(:,Il)', [], 1);
jK = reshape(cMat(:,Jl)', [], 1);
Ke_phys_l = Ke_phys(sub2ind([8,8], Il, Jl));
Me_phys_l = Me_phys(sub2ind([8,8], Il, Jl));

[K, M] = assemble_KM_exact(rho, Ke_phys_l, Me_phys_l, iK, jK, nDof, cfg.penal, cfg.mass_mode);
fixed = build_supports_exact(cfg.support_type, nodeNrs);
free = setdiff(1:nDof, fixed)';

opts.tol = 1e-10;
opts.maxit = 600;
[V, D, flag] = eigs(K(free,free), M(free,free), nModes, 'SM', opts);
if flag ~= 0
    opts.tol = 1e-8;
    opts.maxit = 2000;
    opts.p = min(numel(free)-1, max(40, 4*nModes));
    [V, D, flag] = eigs(K(free,free), M(free,free), nModes, 'SM', opts);
    if flag ~= 0
        warning('PilotOlhoffAlpha05:FinalEigsFlag', 'Final eigs flag=%d.', flag);
    end
end
[lam, ord] = sort(real(diag(D)));
V = real(V(:, ord));
Mf = M(free, free);
for j = 1:size(V, 2)
    sc = sqrt(abs(V(:,j)' * (Mf * V(:,j))));
    if sc > 1e-14
        V(:,j) = V(:,j) / sc;
    end
end
omega = sqrt(max(lam, 0));
freqHz = omega / (2*pi);
Phi = zeros(nDof, nModes);
Phi(free, :) = V;

model = struct('nEl', nEl, 'nDof', nDof, 'nodeNrs', nodeNrs, 'cMat', cMat, ...
    'Ke_phys', Ke_phys, 'Me_phys', Me_phys, 'dx', dx, 'dy', dy);
end

function localSaveTopology(outDir, rho, cfg, omega1)
topo = reshape(rho, cfg.nely, cfg.nelx);
writematrix(topo, fullfile(outDir, 'topology_final.csv'));
fig = figure('Color', 'white', 'Visible', 'off');
ax = axes('Parent', fig);
imagesc(ax, 1 - topo);
axis(ax, 'equal', 'off');
colormap(ax, gray);
title(ax, sprintf('OlhoffExact CC 80x10 alpha=0.5 | omega_1=%.2f rad/s', omega1), ...
    'Interpreter', 'tex', 'FontSize', 9);
try
    exportgraphics(fig, fullfile(outDir, 'topology_final.png'), 'Resolution', 180, 'BackgroundColor', 'white');
catch
    print(fig, fullfile(outDir, 'topology_final.png'), '-dpng', '-r180');
end
close(fig);
end

function T = localWriteConvergence(outDir, hist, cfg)
ni = hist.outer_iters;
iter = (1:ni)';
omegaTrial = hist.omega_trial(1:ni, :);
T = table(iter, omegaTrial(:,1), omegaTrial(:,2), hist.beta(1:ni), ...
    sqrt(max(hist.beta(1:ni), 0)), hist.volume(1:ni), hist.N(1:ni), ...
    hist.N_trial(1:ni), hist.drho_norm(1:ni), hist.drho_max(1:ni), ...
    hist.step_alpha(1:ni), hist.inner_iters(1:ni), ...
    'VariableNames', {'iteration','omega_1_rad_s','omega_2_rad_s', ...
    'beta_lambda_sq','beta_rad_s','volume','multiplicity_N_pre', ...
    'multiplicity_N_trial','design_change_norm','design_change_max', ...
    'step_alpha','inner_iters'});
writetable(T, fullfile(outDir, 'convergence_history.csv'));
writetable(T(:, {'iteration','multiplicity_N_pre','multiplicity_N_trial'}), ...
    fullfile(outDir, 'multiplicity_history.csv'));

fig = figure('Color', 'white', 'Visible', 'off');
ax1 = subplot(3,1,1, 'Parent', fig);
plot(ax1, iter, T.omega_1_rad_s, 'b-', iter, T.omega_2_rad_s, 'r--', 'LineWidth', 1);
grid(ax1, 'on'); ylabel(ax1, 'omega');
legend(ax1, 'omega_1', 'omega_2', 'Location', 'best');
title(ax1, sprintf('OlhoffExact CC 80x10 alpha=%.2f', cfg.alpha));
ax2 = subplot(3,1,2, 'Parent', fig);
plot(ax2, iter, T.beta_rad_s, 'k-', 'LineWidth', 1);
grid(ax2, 'on'); ylabel(ax2, 'sqrt(beta)');
ax3 = subplot(3,1,3, 'Parent', fig);
semilogy(ax3, iter, max(T.design_change_norm, eps), 'm-', 'LineWidth', 1);
hold(ax3, 'on');
yline(ax3, cfg.outer_tol, 'k--');
grid(ax3, 'on'); ylabel(ax3, 'design change'); xlabel(ax3, 'outer iteration');
try
    exportgraphics(fig, fullfile(outDir, 'convergence_history.png'), 'Resolution', 180, 'BackgroundColor', 'white');
catch
    print(fig, fullfile(outDir, 'convergence_history.png'), '-dpng', '-r180');
end
close(fig);
end

function localSaveModeShapes(outDir, Phi, rho, cfg, omega, nSave)
for k = 1:min(nSave, size(Phi, 2))
    localSaveModeShape(fullfile(outDir, sprintf('mode_%02d_shape.png', k)), ...
        Phi(:,k), rho, cfg, omega(k), k);
end
end

function localSaveModeShape(path, phi, rho, cfg, omega, modeIdx)
xG = repmat((0:cfg.nelx)*(cfg.L/cfg.nelx), cfg.nely+1, 1);
yG = repmat((0:cfg.nely)'*(cfg.H/cfg.nely), 1, cfg.nelx+1);
ux = reshape(phi(1:2:end), cfg.nely+1, cfg.nelx+1);
uy = reshape(phi(2:2:end), cfg.nely+1, cfg.nelx+1);
mag = hypot(ux, uy);
sc = 0.06 * max(cfg.L, cfg.H) / max(max(mag(:)), eps);
xD = xG + sc * ux;
yD = yG + sc * uy;

fig = figure('Color', 'white', 'Visible', 'off');
ax = axes('Parent', fig);
hold(ax, 'on');
imagesc(ax, [0 cfg.L], [0 cfg.H], flipud(reshape(rho(:), cfg.nely, cfg.nelx)));
set(ax, 'YDir', 'normal');
colormap(ax, gray);
alpha(0.35);
stride = max(1, ceil((max(cfg.nelx, cfg.nely) + 1) / 100));
rows = unique([1:stride:(cfg.nely+1), cfg.nely+1]);
cols = unique([1:stride:(cfg.nelx+1), cfg.nelx+1]);
for r = rows
    plot(ax, xD(r,:), yD(r,:), '-', 'Color', [0.05 0.25 0.75], 'LineWidth', 0.7);
end
for c = cols
    plot(ax, xD(:,c), yD(:,c), '-', 'Color', [0.05 0.25 0.75], 'LineWidth', 0.7);
end
plot(ax, [0 cfg.L cfg.L 0 0], [0 0 cfg.H cfg.H 0], '-', 'Color', [0.1 0.1 0.1], 'LineWidth', 1);
axis(ax, 'equal');
xlim(ax, [-0.05*cfg.L, 1.05*cfg.L]);
ylim(ax, [-0.15*cfg.H, 1.15*cfg.H]);
set(ax, 'XTick', [], 'YTick', []);
box(ax, 'on');
title(ax, sprintf('Mode %d | %.4g rad/s | alpha=0.5', modeIdx, omega), ...
    'Interpreter', 'none', 'FontSize', 8);
try
    exportgraphics(fig, path, 'Resolution', 180, 'BackgroundColor', 'white');
catch
    print(fig, path, '-dpng', '-r180');
end
close(fig);
end

function s1 = localS1Diagnose(rho, Phi, omega, freqHz, K, M, free, model, cfg)
nModes = min(10, size(Phi, 2));
lowThr = 0.1;
solidThr = 0.5;
lowMask = rho < lowThr;
[component, compStats] = localComponents(rho, cfg.nelx, cfg.nely, solidThr);
largestId = localLargestSupport(compStats);
rows = repmat(localBlankMode(), nModes, 1);
for k = 1:nModes
    [kinElem, strElem] = localElemEnergies(Phi(:,k), omega(k), rho, model.cMat, ...
        model.Ke_phys, model.Me_phys, cfg.mass_mode, cfg.penal);
    rows(k) = localDiagnoseMode(k, omega(k), freqHz(k), kinElem, strElem, ...
        rho, component, compStats, largestId, lowMask, lowThr);
end
s1 = struct();
s1.mesh = struct('nelx', cfg.nelx, 'nely', cfg.nely, 'L', cfg.L, 'H', cfg.H, ...
    'hx', cfg.L/cfg.nelx, 'hy', cfg.H/cfg.nely);
s1.mass_mode = cfg.mass_mode;
s1.penal = cfg.penal;
s1.low_density_threshold = lowThr;
s1.solid_threshold = solidThr;
s1.n_modes = nModes;
s1.components = localComponentSummary(compStats, largestId);
s1.modes = rows;
s1.overall = localOverall(rows);
fprintf('S1 overall: %s | localized_low_density=%d | physical_global=%d\n', ...
    s1.overall.classification, s1.overall.localized_low_density_count, ...
    s1.overall.physical_global_count);
end

function [kinElem, strElem] = localElemEnergies(phi, omega, rho, cMat, KePhys, MePhys, massMode, penal)
nEl = size(cMat, 1);
kinElem = zeros(nEl, 1);
strElem = zeros(nEl, 1);
for e = 1:nEl
    ue = phi(cMat(e,:));
    [me, ~] = mass_interp(rho(e), massMode);
    strElem(e) = 0.5 * real(ue' * (rho(e)^penal * KePhys) * ue);
    kinElem(e) = 0.5 * omega^2 * real(ue' * (me * MePhys) * ue);
end
end

function row = localBlankMode()
row = struct('mode', NaN, 'omega_rad_s', NaN, 'frequency_hz', NaN, ...
    'kinetic_energy_total', NaN, 'strain_energy_total', NaN, ...
    'low_density_kinetic_fraction', NaN, 'low_density_strain_fraction', NaN, ...
    'kinetic_effective_element_fraction', NaN, 'strain_effective_element_fraction', NaN, ...
    'dominant_solid_component_kinetic_fraction', NaN, ...
    'dominant_solid_component_strain_fraction', NaN, ...
    'dominant_component_touches_both_supports', false, ...
    'largest_support_component_kinetic_fraction', NaN, ...
    'largest_support_component_strain_fraction', NaN, ...
    'classification', '', 'classification_reason', '');
end

function row = localDiagnoseMode(k, omega, freqHz, kinElem, strElem, rho, ...
    component, compStats, largestId, lowMask, lowThr)
row = localBlankMode();
row.mode = k;
row.omega_rad_s = omega;
row.frequency_hz = freqHz;
row.kinetic_energy_total = sum(kinElem);
row.strain_energy_total = sum(strElem);
row.low_density_kinetic_fraction = localFrac(kinElem(lowMask), kinElem);
row.low_density_strain_fraction = localFrac(strElem(lowMask), strElem);
[~, row.kinetic_effective_element_fraction] = localLoc(kinElem);
[~, row.strain_effective_element_fraction] = localLoc(strElem);
[domId, domK, domS] = localDominantComponent(component, kinElem, strElem);
row.dominant_solid_component_kinetic_fraction = domK;
row.dominant_solid_component_strain_fraction = domS;
if domId > 0 && domId <= numel(compStats)
    row.dominant_component_touches_both_supports = ...
        compStats(domId).touches_left && compStats(domId).touches_right;
end
if largestId > 0
    supportMask = component == largestId;
    row.largest_support_component_kinetic_fraction = localFrac(kinElem(supportMask), kinElem);
    row.largest_support_component_strain_fraction = localFrac(strElem(supportMask), strElem);
end
[row.classification, row.classification_reason] = localClassifyMode(row, lowThr);
end

function [cls, reason] = localClassifyMode(row, lowThr)
if row.low_density_kinetic_fraction >= 0.35 || row.low_density_strain_fraction >= 0.35
    cls = 'localized low-density mode';
    reason = sprintf('low-density energy fractions: kinetic %.3f, strain %.3f', ...
        row.low_density_kinetic_fraction, row.low_density_strain_fraction);
elseif row.kinetic_effective_element_fraction < 0.025 || row.strain_effective_element_fraction < 0.025
    cls = 'localized non-low-density mode';
    reason = sprintf('localized participation: effK %.3f, effS %.3f', ...
        row.kinetic_effective_element_fraction, row.strain_effective_element_fraction);
elseif row.largest_support_component_kinetic_fraction >= 0.60 && ...
        row.largest_support_component_strain_fraction >= 0.60 && ...
        row.dominant_component_touches_both_supports
    cls = 'physical global mode';
    reason = sprintf('energy on support-spanning component: K %.3f, S %.3f', ...
        row.largest_support_component_kinetic_fraction, ...
        row.largest_support_component_strain_fraction);
else
    cls = 'ambiguous';
    reason = sprintf('mixed indicators: ldK %.3f, ldS %.3f, suppK %.3f, suppS %.3f', ...
        row.low_density_kinetic_fraction, row.low_density_strain_fraction, ...
        row.largest_support_component_kinetic_fraction, ...
        row.largest_support_component_strain_fraction);
end
reason = sprintf('%s; low-density threshold rho<%.2g', reason, lowThr);
end

function overall = localOverall(rows)
classes = string({rows.classification});
first3 = classes(1:min(3, numel(classes)));
physCount = sum(classes == "physical global mode");
locCount = sum(classes == "localized low-density mode");
locNdCount = sum(classes == "localized non-low-density mode");
ambCount = sum(classes == "ambiguous");
if any(first3 == "localized low-density mode")
    cls = 'likely localized/spurious low-density mode influence';
    reason = 'At least one of the first three modes is localized low-density.';
elseif any(first3 == "localized non-low-density mode")
    cls = 'localized but not in low-density region';
    reason = 'At least one of the first three modes is localized outside low-density regions.';
elseif all(first3 == "physical global mode") || ...
        (sum(first3 == "physical global mode") >= 2 && ~any(first3 == "localized low-density mode"))
    cls = 'likely physically valid';
    reason = 'First modes are physical global modes.';
else
    cls = 'ambiguous';
    reason = 'Mixed classification among first three modes.';
end
overall = struct('classification', cls, 'reason', reason, ...
    'physical_global_count', physCount, ...
    'localized_low_density_count', locCount, ...
    'localized_non_low_density_count', locNdCount, ...
    'ambiguous_count', ambCount);
end

function [component, stats] = localComponents(rho, nelx, nely, threshold)
solid = reshape(rho(:) >= threshold, nely, nelx);
component = zeros(nely, nelx);
stats = repmat(struct('id', 0, 'size', 0, 'touches_left', false, ...
    'touches_right', false, 'touches_bottom', false, 'touches_top', false), 0, 1);
compId = 0;
for ix = 1:nelx
    for iy = 1:nely
        if ~solid(iy,ix) || component(iy,ix) ~= 0
            continue
        end
        compId = compId + 1;
        q = zeros(nnz(solid), 2);
        head = 1;
        tail = 1;
        q(tail,:) = [iy, ix];
        component(iy,ix) = compId;
        st = struct('id', compId, 'size', 0, 'touches_left', false, ...
            'touches_right', false, 'touches_bottom', false, 'touches_top', false);
        while head <= tail
            cy = q(head,1);
            cx = q(head,2);
            head = head + 1;
            st.size = st.size + 1;
            st.touches_left = st.touches_left || cx == 1;
            st.touches_right = st.touches_right || cx == nelx;
            st.touches_bottom = st.touches_bottom || cy == 1;
            st.touches_top = st.touches_top || cy == nely;
            nbr = [cy-1,cx; cy+1,cx; cy,cx-1; cy,cx+1];
            for ii = 1:4
                ny = nbr(ii,1);
                nx = nbr(ii,2);
                if ny >= 1 && ny <= nely && nx >= 1 && nx <= nelx && ...
                        solid(ny,nx) && component(ny,nx) == 0
                    tail = tail + 1;
                    q(tail,:) = [ny, nx];
                    component(ny,nx) = compId;
                end
            end
        end
        stats(end+1,1) = st; %#ok<AGROW>
    end
end
component = component(:);
end

function summary = localComponentSummary(stats, largestId)
summary = struct();
summary.count = numel(stats);
summary.largest_support_connected_component_id = largestId;
if isempty(stats)
    summary.support_connected_component_count = 0;
    summary.largest_component_size = 0;
    summary.largest_support_connected_component_size = 0;
else
    summary.support_connected_component_count = sum([stats.touches_left] & [stats.touches_right]);
    summary.largest_component_size = max([stats.size]);
    if largestId > 0
        summary.largest_support_connected_component_size = stats(largestId).size;
    else
        summary.largest_support_connected_component_size = 0;
    end
end
end

function id = localLargestSupport(stats)
id = 0;
best = -Inf;
for i = 1:numel(stats)
    if stats(i).touches_left && stats(i).touches_right && stats(i).size > best
        best = stats(i).size;
        id = stats(i).id;
    end
end
end

function [domId, fracK, fracS] = localDominantComponent(component, kinElem, strElem)
ids = unique(component(component > 0));
if isempty(ids)
    domId = NaN;
    fracK = NaN;
    fracS = NaN;
    return
end
score = zeros(numel(ids), 1);
fK = score;
fS = score;
for i = 1:numel(ids)
    mask = component == ids(i);
    fK(i) = localFrac(kinElem(mask), kinElem);
    fS(i) = localFrac(strElem(mask), strElem);
    score(i) = max(fK(i), fS(i));
end
[~, idx] = max(score);
domId = ids(idx);
fracK = fK(idx);
fracS = fS(idx);
end

function f = localFrac(part, allVals)
den = sum(max(real(allVals(:)), 0));
if den <= 0
    f = NaN;
else
    f = sum(max(real(part(:)), 0)) / den;
end
end

function [idx, eff] = localLoc(vals)
e = max(real(vals(:)), 0);
total = sum(e);
if total <= 0
    idx = NaN;
    eff = NaN;
else
    p = e / total;
    idx = sum(p.^2);
    eff = 1 / (numel(e) * idx);
end
end

function baseline = localLoadBaseline(convPath, s1Path, resultPath, logPath)
baseline = struct('available', false);
if exist(convPath, 'file') == 2
    baseline.convergence = readtable(convPath);
    baseline.available = true;
end
if exist(s1Path, 'file') == 2
    baseline.s1 = jsondecode(fileread(s1Path));
end
if exist(resultPath, 'file') == 2
    baseline.result = jsondecode(fileread(resultPath));
end
if exist(logPath, 'file') == 2
    baseline.log_history = localParseBaselineLog(logPath);
end
end

function T = localParseBaselineLog(path)
txt = splitlines(string(fileread(path)));
rows = [];
for i = 1:numel(txt)
    line = char(txt(i));
    tok = regexp(line, ['^\s*(\d+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+', ...
        '(\d+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+', ...
        '([0-9.Ee+-]+)\s+(\d+)'], 'tokens', 'once');
    if isempty(tok)
        continue
    end
    vals = str2double(tok);
    rows(end+1, :) = vals; %#ok<AGROW>
end
if isempty(rows)
    T = table();
else
    T = array2table(rows, 'VariableNames', {'iteration','omega_1_rad_s','omega_2_rad_s', ...
        'multiplicity_N','volume','design_change_norm','beta_rad_s','step_alpha','inner_iters'});
end
end

function comparison = localCompareAgainstBaseline(baseline, dampedT, s1, hist, outerTol)
comparison = struct();
comparison.baseline_available = isfield(baseline, 'convergence') && ~isempty(baseline.convergence);
comparison.baseline = struct();
comparison.damped = struct();
if comparison.baseline_available
    bT = baseline.convergence;
    comparison.baseline.iterations = height(bT);
    comparison.baseline.omega1_final = bT.omega_1_rad_s(end);
    comparison.baseline.beta_final = bT.beta_lambda_sq(end);
    comparison.baseline.cycle = localCycleMetrics(bT.iteration, bT.omega_1_rad_s, ...
        sqrt(max(bT.beta_lambda_sq, 0)), localMaybeBaselineDrho(baseline));
end
if isfield(baseline, 's1')
    comparison.baseline.s1_overall = baseline.s1.overall;
    comparison.baseline.mode1_ld_strain = baseline.s1.modes(1).low_density_strain_fraction;
    comparison.baseline.components = baseline.s1.components;
end

comparison.damped.iterations = height(dampedT);
comparison.damped.omega1_final = dampedT.omega_1_rad_s(end);
comparison.damped.beta_final = dampedT.beta_lambda_sq(end);
comparison.damped.final_design_change = hist.drho_norm(end);
comparison.damped.cycle = localCycleMetrics(dampedT.iteration, dampedT.omega_1_rad_s, ...
    dampedT.beta_rad_s, dampedT.design_change_norm);
comparison.damped.s1_overall = s1.overall;
comparison.damped.mode1_ld_strain = s1.modes(1).low_density_strain_fraction;
comparison.damped.components = s1.components;
comparison.outer_tol = outerTol;

if isfield(comparison.baseline, 'cycle')
    comparison.omega_cycle_gap_reduction = comparison.baseline.cycle.omega_parity_gap - ...
        comparison.damped.cycle.omega_parity_gap;
    comparison.beta_cycle_gap_reduction = comparison.baseline.cycle.beta_parity_gap - ...
        comparison.damped.cycle.beta_parity_gap;
else
    comparison.omega_cycle_gap_reduction = NaN;
    comparison.beta_cycle_gap_reduction = NaN;
end
comparison.two_cycle_removed = comparison.damped.cycle.two_cycle_present == false;
end

function drho = localMaybeBaselineDrho(baseline)
drho = [];
if isfield(baseline, 'log_history') && ~isempty(baseline.log_history) && ...
        any(strcmp(baseline.log_history.Properties.VariableNames, 'design_change_norm'))
    drho = baseline.log_history.design_change_norm;
end
end

function m = localCycleMetrics(iter, omega1, betaRad, drho)
window = min(80, numel(iter));
idx = (numel(iter)-window+1):numel(iter);
it = iter(idx);
om = omega1(idx);
bt = betaRad(idx);
evenMask = mod(it, 2) == 0;
oddMask = ~evenMask;
m = struct();
m.window = window;
m.omega_even_mean = mean(om(evenMask), 'omitnan');
m.omega_odd_mean = mean(om(oddMask), 'omitnan');
m.omega_parity_gap = abs(m.omega_even_mean - m.omega_odd_mean);
m.beta_even_mean = mean(bt(evenMask), 'omitnan');
m.beta_odd_mean = mean(bt(oddMask), 'omitnan');
m.beta_parity_gap = abs(m.beta_even_mean - m.beta_odd_mean);
if ~isempty(drho) && numel(drho) >= numel(iter)
    d = drho(idx);
    m.design_change_mean = mean(d, 'omitnan');
    m.design_change_final = d(end);
else
    m.design_change_mean = NaN;
    m.design_change_final = NaN;
end
m.two_cycle_present = m.omega_parity_gap > 5 && ...
    (isnan(m.design_change_mean) || m.design_change_mean > 0.02);
end

function classification = localClassify(hist, cfg, s1, comparison)
capped = hist.outer_iters >= cfg.outer_max_iter;
converged = ~capped && hist.drho_norm(end) <= cfg.outer_tol;
noLocalized = s1.overall.localized_low_density_count == 0;
if converged && noLocalized
    label = 'converged accepted';
    reason = 'Reached outer tolerance before cap and S1 found no localized low-density modes.';
elseif comparison.damped.cycle.two_cycle_present
    label = 'still 2-cycle';
    reason = sprintf('Last-window omega parity gap %.4g rad/s with design-change mean %.4g.', ...
        comparison.damped.cycle.omega_parity_gap, comparison.damped.cycle.design_change_mean);
elseif capped
    label = 'capped but improved';
    reason = sprintf('Reached cap, but two-cycle criterion is false; omega parity gap %.4g rad/s.', ...
        comparison.damped.cycle.omega_parity_gap);
else
    label = 'failed';
    reason = 'Run ended without meeting accepted or improved criteria.';
end
classification = struct('label', label, 'reason', reason, ...
    'converged', converged, 'capped', capped, 'no_localized_low_density_modes', noLocalized);
end

function localWriteSummary(path, result, s1, comparison)
lines = {};
lines{end+1} = '# Stabilized OlhoffApproachExact Pilot';
lines{end+1} = '';
lines{end+1} = sprintf('Generated: %s', result.created_utc);
lines{end+1} = '';
lines{end+1} = '## Scope';
lines{end+1} = '';
lines{end+1} = 'Single CC 80x10 OlhoffApproachExact pilot with alpha-damping. No ourApproach changes, no manuscript edits, no full revision experiments.';
lines{end+1} = '';
lines{end+1} = '## Settings Verification';
lines{end+1} = '';
lines{end+1} = sprintf('- only changed parameter: `alpha` %.3g -> %.3g', ...
    result.changed_parameter.baseline, result.changed_parameter.pilot);
lines{end+1} = sprintf('- verification pass: `%s`', string(result.settings_verification.pass));
lines{end+1} = sprintf('- mesh: %dx%d, volfrac=%.3g, mass=%s, p=%.3g, rmin_elem=%.3g', ...
    result.config.nelx, result.config.nely, result.config.volfrac, ...
    result.config.mass_mode, result.config.penal, result.config.rmin_elem);
lines{end+1} = sprintf('- inner MMA: max_iter=%d, tol=%.3g, move_lim=%s, outer_move=%.3g', ...
    result.config.inner_max_iter, result.config.inner_tol, mat2str(result.config.move_lim), ...
    result.config.outer_move);
lines{end+1} = '';
lines{end+1} = '## Acceptance';
lines{end+1} = '';
lines{end+1} = '| metric | value |';
lines{end+1} = '|---|---:|';
lines{end+1} = sprintf('| classification | %s |', result.classification.label);
lines{end+1} = sprintf('| outer iterations | %d/%d |', result.outer_iters, result.config.outer_max_iter);
lines{end+1} = sprintf('| final design change | %.6g |', result.final_design_change);
lines{end+1} = sprintf('| outer tolerance | %.6g |', result.outer_tol);
lines{end+1} = sprintf('| final volume | %.6g |', result.final_volume);
lines{end+1} = sprintf('| final multiplicity N | %d |', result.final_N);
lines{end+1} = '';
lines{end+1} = '## First 6 Eigenfrequencies';
lines{end+1} = '';
lines{end+1} = '| mode | omega rad/s | Hz |';
lines{end+1} = '|---:|---:|---:|';
for k = 1:6
    lines{end+1} = sprintf('| %d | %.8g | %.8g |', k, result.first_6_omega_rad_s(k), ...
        result.first_6_frequency_hz(k)); %#ok<AGROW>
end
lines{end+1} = '';
lines{end+1} = '## S1 Diagnosis';
lines{end+1} = '';
lines{end+1} = sprintf('- overall: %s', s1.overall.classification);
lines{end+1} = sprintf('- localized low-density modes: %d/10', s1.overall.localized_low_density_count);
lines{end+1} = sprintf('- physical global modes: %d/10', s1.overall.physical_global_count);
lines{end+1} = sprintf('- mode 1 low-density strain fraction: %.8g', s1.modes(1).low_density_strain_fraction);
lines{end+1} = sprintf('- support-connected solid components: %d', s1.components.support_connected_component_count);
lines{end+1} = '';
lines{end+1} = '## Comparison Against Undamped Pilot';
lines{end+1} = '';
lines{end+1} = '| metric | undamped alpha=1 | damped alpha=0.5 |';
lines{end+1} = '|---|---:|---:|';
if isfield(comparison.baseline, 'cycle')
    lines{end+1} = sprintf('| omega parity gap, last window | %.6g | %.6g |', ...
        comparison.baseline.cycle.omega_parity_gap, comparison.damped.cycle.omega_parity_gap);
    lines{end+1} = sprintf('| beta parity gap, last window | %.6g | %.6g |', ...
        comparison.baseline.cycle.beta_parity_gap, comparison.damped.cycle.beta_parity_gap);
    lines{end+1} = sprintf('| design change final | %.6g | %.6g |', ...
        comparison.baseline.cycle.design_change_final, comparison.damped.cycle.design_change_final);
end
if isfield(comparison.baseline, 's1_overall')
    lines{end+1} = sprintf('| localized low-density modes | %d | %d |', ...
        comparison.baseline.s1_overall.localized_low_density_count, ...
        comparison.damped.s1_overall.localized_low_density_count);
    lines{end+1} = sprintf('| mode 1 ld_strain_frac | %.6g | %.6g |', ...
        comparison.baseline.mode1_ld_strain, comparison.damped.mode1_ld_strain);
    lines{end+1} = sprintf('| support-connected components | %d | %d |', ...
        comparison.baseline.components.support_connected_count, ...
        comparison.damped.components.support_connected_component_count);
end
lines{end+1} = sprintf('| 2-cycle removed |  | %s |', string(comparison.two_cycle_removed));
lines{end+1} = '';
lines{end+1} = '## Conclusion';
lines{end+1} = '';
lines{end+1} = sprintf('Classification: **%s**. %s', result.classification.label, result.classification.reason);
localWriteText(path, strjoin(lines, newline));
end

function localWriteManifest(path, outDir, result)
files = {'settings_verification.json','stabilized_pilot_result.mat', ...
    'stabilized_pilot_result.json','stabilized_pilot_summary.md', ...
    'topology_final.csv','topology_final.png','convergence_history.csv', ...
    'convergence_history.png','multiplicity_history.csv','s1_mode_summary.json', ...
    's1_modes.csv','pilot_run.log'};
modeFiles = cell(1, 6);
for k = 1:6
    modeFiles{k} = sprintf('mode_%02d_shape.png', k);
end
manifest = struct();
manifest.study = result.study;
manifest.created_utc = result.created_utc;
manifest.output_dir = outDir;
manifest.files = [files, modeFiles];
manifest.classification = result.classification;
manifest.changed_parameter = result.changed_parameter;
localWriteJson(path, manifest);
end

function localWriteJson(path, data)
try
    txt = jsonencode(data, PrettyPrint=true);
catch
    txt = jsonencode(data);
end
localWriteText(path, txt);
end

function localWriteText(path, txt)
fid = fopen(path, 'w');
if fid < 0
    error('PilotOlhoffAlpha05:WriteFailed', 'Cannot write %s', path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', char(txt));
end
