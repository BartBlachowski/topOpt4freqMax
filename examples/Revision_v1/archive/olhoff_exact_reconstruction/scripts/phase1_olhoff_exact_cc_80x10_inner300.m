function result = phase1_olhoff_exact_cc_80x10_inner300(outDir)
%PHASE1_OLHOFF_EXACT_CC_80X10_INNER300  Numerical-behaviour Phase 1.
%
% Changes exactly one parameter relative to the saved alpha=0.5 pilot:
% inner_max_iter: 30 -> 300.

warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:singularMatrix');

if nargin < 1 || isempty(outDir)
    scriptDir = fileparts(mfilename('fullpath'));
    outDir = fullfile(scriptDir, 'output', 'phase1_olhoff_exact_cc_80x10_inner300');
end
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));
addpath(fullfile(repoRoot, 'analysis', 'OlhoffApproachExact', 'Matlab'));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));

logPath = fullfile(outDir, 'phase1_inner300_run.log');
diary(logPath);
cleanupD = onCleanup(@() diary('off')); %#ok<NASGU>

fprintf('Phase 1 OlhoffApproachExact CC 80x10 inner_max_iter=300 started: %s\n', ...
    char(datetime('now', 'TimeZone', 'UTC')));
fprintf('Scope: MATLAB only, OlhoffApproachExact only. One parameter changes: inner_max_iter 30 -> 300.\n\n');

baselineDir = fullfile(scriptDir, 'output', 'pilot_olhoff_exact_cc_80x10_alpha05');
baselineConvPath = fullfile(baselineDir, 'convergence_history.csv');
baselineS1Path = fullfile(baselineDir, 's1_mode_summary.json');
baselineResultPath = fullfile(baselineDir, 'stabilized_pilot_result.json');

cfgBaseline = localBaseCfg();
cfgPhase1 = cfgBaseline;
cfgPhase1.inner_max_iter = 300;
verification = localVerifyOnlyInnerMaxIterChanged(cfgBaseline, cfgPhase1);
localWriteJson(fullfile(outDir, 'phase1_inner300_parameter_verification.json'), verification);
if ~verification.pass
    error('Phase1Inner300:ParameterVerificationFailed', ...
        'Abort: a parameter other than inner_max_iter differs.');
end
fprintf('Parameter verification passed: only inner_max_iter changed (%d -> %d).\n\n', ...
    cfgBaseline.inner_max_iter, cfgPhase1.inner_max_iter);

tic;
[rho_final, hist] = topopt_freq_exact(cfgPhase1);
elapsed_s = toc;
fprintf('\nOptimization finished in %.1f s, outer_iters=%d.\n', elapsed_s, hist.outer_iters);

[model, omega, freqHz, Phi, K, M, free] = localFinalModes(rho_final, cfgPhase1, 10);
s1 = localS1Diagnose(rho_final, Phi, omega, freqHz, model, cfgPhase1);

convT = localWriteConvergence(outDir, hist);
localWritePlots(outDir, convT);
localSaveTopology(outDir, rho_final, cfgPhase1, omega(1));
localSaveModeShapes(outDir, Phi, rho_final, cfgPhase1, omega, 6);
localWriteJson(fullfile(outDir, 's1_mode_summary.json'), s1);
writetable(struct2table(s1.modes), fullfile(outDir, 's1_modes.csv'));

baseline = localLoadBaseline(baselineConvPath, baselineS1Path, baselineResultPath);
comparison = localCompare(baseline, convT, hist, s1);
decision = localDecision(comparison);

result = struct();
result.study = 'Phase 1 numerical-behaviour investigation: inner_max_iter 300';
result.created_utc = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'));
result.scope = 'Single-parameter test; OlhoffApproachExact CC 80x10 alpha=0.5';
result.changed_parameter = struct('name', 'inner_max_iter', ...
    'baseline', cfgBaseline.inner_max_iter, 'phase1', cfgPhase1.inner_max_iter);
result.parameter_verification = verification;
result.config = cfgPhase1;
result.elapsed_s = elapsed_s;
result.outer_iters = hist.outer_iters;
result.capped = hist.outer_iters >= cfgPhase1.outer_max_iter;
result.final_design_change = hist.drho_norm(end);
result.outer_tol = cfgPhase1.outer_tol;
result.final_volume = mean(rho_final);
result.final_N = hist.final_N;
result.first_6_omega_rad_s = omega(1:6)';
result.first_6_frequency_hz = freqHz(1:6)';
result.inner_solver = comparison.phase1.inner_solver;
result.outer_convergence = comparison.phase1.outer;
result.s1_overall = s1.overall;
result.topology_connectedness = s1.components;
result.comparison_vs_inner30 = comparison;
result.decision = decision;
result.final_answers = localFinalAnswers(comparison, decision);

save(fullfile(outDir, 'phase1_inner300_result.mat'), ...
    'result', 'cfgBaseline', 'cfgPhase1', 'rho_final', 'hist', 'omega', ...
    'freqHz', 's1', 'comparison', 'decision');
localWriteJson(fullfile(outDir, 'phase1_inner300_result.json'), result);
localWriteSummary(fullfile(outDir, 'phase1_inner300_summary.md'), result, s1, comparison, decision);
localWriteComparison(fullfile(outDir, 'phase1_inner300_comparison.md'), comparison, s1, result);
localWriteManifest(fullfile(outDir, 'phase1_inner300_manifest.json'), outDir, result);

fprintf('\nDecision: %s\n', decision.code);
fprintf('Summary: %s\n', fullfile(outDir, 'phase1_inner300_summary.md'));
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
cfg.alpha = 0.5;
cfg.verbose = true;
end

function verification = localVerifyOnlyInnerMaxIterChanged(a, b)
fields = sort(fieldnames(a));
diffs = {};
for i = 1:numel(fields)
    f = fields{i};
    if ~isfield(b, f)
        diffs{end+1} = sprintf('%s missing', f); %#ok<AGROW>
    elseif strcmp(f, 'inner_max_iter')
        continue
    elseif ~isequaln(a.(f), b.(f))
        diffs{end+1} = f; %#ok<AGROW>
    end
end
extra = setdiff(fieldnames(b), fieldnames(a));
for i = 1:numel(extra)
    diffs{end+1} = sprintf('%s extra', extra{i}); %#ok<AGROW>
end
verification = struct();
verification.pass = isempty(diffs) && a.inner_max_iter == 30 && b.inner_max_iter == 300;
verification.changed_parameter = 'inner_max_iter';
verification.baseline_value = a.inner_max_iter;
verification.phase1_value = b.inner_max_iter;
verification.differences_other_than_inner_max_iter = diffs;
verification.baseline_config = a;
verification.phase1_config = b;
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
Ke_l = Ke_phys(sub2ind([8,8], Il, Jl));
Me_l = Me_phys(sub2ind([8,8], Il, Jl));
[K, M] = assemble_KM_exact(rho, Ke_l, Me_l, iK, jK, nDof, cfg.penal, cfg.mass_mode);
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
        warning('Phase1Inner300:FinalEigsFlag', 'Final eigs flag=%d.', flag);
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
Phi(free,:) = V;
model = struct('nEl', nEl, 'nDof', nDof, 'nodeNrs', nodeNrs, ...
    'cMat', cMat, 'Ke_phys', Ke_phys, 'Me_phys', Me_phys);
end

function T = localWriteConvergence(outDir, hist)
ni = hist.outer_iters;
iters = (1:ni)';
omegaTrial = hist.omega_trial(1:ni, :);
termination = string(hist.inner_termination_reason(1:ni));
T = table(iters, omegaTrial(:,1), omegaTrial(:,2), hist.beta(1:ni), ...
    sqrt(max(hist.beta(1:ni), 0)), hist.drho_norm(1:ni), hist.drho_max(1:ni), ...
    hist.volume(1:ni), hist.N(1:ni), hist.N_trial(1:ni), hist.inner_iters(1:ni), ...
    hist.inner_converged(1:ni), hist.inner_hit_max_iter(1:ni), termination, ...
    hist.inner_cpu_time(1:ni), hist.step_alpha(1:ni), ...
    'VariableNames', {'outer_iteration','omega1','omega2','beta_lambda_sq', ...
    'beta_rad_s','design_change','design_change_max','volume','multiplicity_N_pre', ...
    'multiplicity_N_trial','inner_iterations','inner_converged', ...
    'inner_hit_max_iter','inner_termination','inner_cpu_time_s','step_alpha'});
writetable(T, fullfile(outDir, 'phase1_inner300_convergence_history.csv'));
writetable(T(:, {'outer_iteration','multiplicity_N_pre','multiplicity_N_trial'}), ...
    fullfile(outDir, 'phase1_inner300_multiplicity_history.csv'));
end

function localWritePlots(outDir, T)
localLinePlot(fullfile(outDir, 'phase1_inner300_omega_history.png'), ...
    T.outer_iteration, [T.omega1, T.omega2], {'omega_1','omega_2'}, 'omega rad/s');
localLinePlot(fullfile(outDir, 'phase1_inner300_beta_history.png'), ...
    T.outer_iteration, T.beta_rad_s, {'sqrt(beta)'}, 'sqrt(beta) rad/s');
localLinePlot(fullfile(outDir, 'phase1_inner300_design_change_history.png'), ...
    T.outer_iteration, T.design_change, {'design change'}, 'norm(delta rho)/sqrt(nEl)', true);

fig = figure('Color', 'white', 'Visible', 'off');
histogram(T.inner_iterations);
grid on;
xlabel('inner iterations');
ylabel('outer-iteration count');
title('Phase 1 inner iteration histogram');
try
    exportgraphics(fig, fullfile(outDir, 'phase1_inner300_inner_iteration_histogram.png'), ...
        'Resolution', 180, 'BackgroundColor', 'white');
catch
    print(fig, fullfile(outDir, 'phase1_inner300_inner_iteration_histogram.png'), '-dpng', '-r180');
end
close(fig);
end

function localLinePlot(path, x, y, labels, yLabel, useLog)
if nargin < 6
    useLog = false;
end
fig = figure('Color', 'white', 'Visible', 'off');
if useLog
    semilogy(x, max(y, eps), 'LineWidth', 1);
else
    plot(x, y, 'LineWidth', 1);
end
grid on;
xlabel('outer iteration');
ylabel(yLabel);
legend(labels, 'Location', 'best');
title(strrep(yLabel, '_', '\_'));
try
    exportgraphics(fig, path, 'Resolution', 180, 'BackgroundColor', 'white');
catch
    print(fig, path, '-dpng', '-r180');
end
close(fig);
end

function localSaveTopology(outDir, rho, cfg, omega1)
topo = reshape(rho, cfg.nely, cfg.nelx);
writematrix(topo, fullfile(outDir, 'phase1_inner300_topology.csv'));
fig = figure('Color', 'white', 'Visible', 'off');
imagesc(1 - topo);
axis equal off;
colormap(gray);
title(sprintf('Phase 1 inner300 | omega_1=%.2f rad/s', omega1), 'Interpreter', 'tex');
try
    exportgraphics(fig, fullfile(outDir, 'phase1_inner300_topology.png'), ...
        'Resolution', 180, 'BackgroundColor', 'white');
catch
    print(fig, fullfile(outDir, 'phase1_inner300_topology.png'), '-dpng', '-r180');
end
close(fig);
end

function localSaveModeShapes(outDir, Phi, rho, cfg, omega, nSave)
for k = 1:min(nSave, size(Phi, 2))
    localSaveModeShape(fullfile(outDir, sprintf('phase1_inner300_mode_%02d_shape.png', k)), ...
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
fig = figure('Color', 'white', 'Visible', 'off');
ax = axes('Parent', fig);
hold(ax, 'on');
imagesc(ax, [0 cfg.L], [0 cfg.H], flipud(reshape(rho(:), cfg.nely, cfg.nelx)));
set(ax, 'YDir', 'normal');
colormap(ax, gray);
alpha(0.35);
xD = xG + sc * ux;
yD = yG + sc * uy;
stride = max(1, ceil((max(cfg.nelx, cfg.nely)+1)/100));
rows = unique([1:stride:(cfg.nely+1), cfg.nely+1]);
cols = unique([1:stride:(cfg.nelx+1), cfg.nelx+1]);
for r = rows
    plot(ax, xD(r,:), yD(r,:), '-', 'Color', [0.05 0.25 0.75], 'LineWidth', 0.7);
end
for c = cols
    plot(ax, xD(:,c), yD(:,c), '-', 'Color', [0.05 0.25 0.75], 'LineWidth', 0.7);
end
axis(ax, 'equal');
xlim(ax, [-0.05*cfg.L 1.05*cfg.L]);
ylim(ax, [-0.15*cfg.H 1.15*cfg.H]);
set(ax, 'XTick', [], 'YTick', []);
title(ax, sprintf('Mode %d | %.4g rad/s | inner300', modeIdx, omega), 'Interpreter', 'none');
try
    exportgraphics(fig, path, 'Resolution', 180, 'BackgroundColor', 'white');
catch
    print(fig, path, '-dpng', '-r180');
end
close(fig);
end

function s1 = localS1Diagnose(rho, Phi, omega, freqHz, model, cfg)
lowThr = 0.1;
solidThr = 0.5;
nModes = min(10, size(Phi, 2));
lowMask = rho < lowThr;
[component, compStats] = localComponents(rho, cfg.nelx, cfg.nely, solidThr);
largestId = localLargestSupport(compStats);
rows = repmat(localBlankMode(), nModes, 1);
for k = 1:nModes
    [ke, se] = localElemEnergies(Phi(:,k), omega(k), rho, model.cMat, ...
        model.Ke_phys, model.Me_phys, cfg.mass_mode, cfg.penal);
    rows(k) = localDiagnoseMode(k, omega(k), freqHz(k), ke, se, component, ...
        compStats, largestId, lowMask, lowThr);
end
s1 = struct();
s1.mesh = struct('nelx', cfg.nelx, 'nely', cfg.nely, 'L', cfg.L, 'H', cfg.H);
s1.low_density_threshold = lowThr;
s1.solid_threshold = solidThr;
s1.components = localComponentSummary(compStats, largestId);
s1.modes = rows;
s1.overall = localOverall(rows);
fprintf('S1: %s, localized_low_density=%d/10, mode1 ld_strain=%.6g\n', ...
    s1.overall.classification, s1.overall.localized_low_density_count, ...
    s1.modes(1).low_density_strain_fraction);
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
    'low_density_kinetic_fraction', NaN, 'low_density_strain_fraction', NaN, ...
    'kinetic_effective_element_fraction', NaN, 'strain_effective_element_fraction', NaN, ...
    'dominant_solid_component_kinetic_fraction', NaN, ...
    'dominant_solid_component_strain_fraction', NaN, ...
    'dominant_component_touches_both_supports', false, ...
    'largest_support_component_kinetic_fraction', NaN, ...
    'largest_support_component_strain_fraction', NaN, ...
    'classification', '', 'classification_reason', '');
end

function row = localDiagnoseMode(k, omega, freqHz, kinElem, strElem, component, compStats, largestId, lowMask, lowThr)
row = localBlankMode();
row.mode = k;
row.omega_rad_s = omega;
row.frequency_hz = freqHz;
row.low_density_kinetic_fraction = localFrac(kinElem(lowMask), kinElem);
row.low_density_strain_fraction = localFrac(strElem(lowMask), strElem);
[~, row.kinetic_effective_element_fraction] = localLoc(kinElem);
[~, row.strain_effective_element_fraction] = localLoc(strElem);
[domId, domK, domS] = localDominantComponent(component, kinElem, strElem);
row.dominant_solid_component_kinetic_fraction = domK;
row.dominant_solid_component_strain_fraction = domS;
if domId > 0 && domId <= numel(compStats)
    row.dominant_component_touches_both_supports = compStats(domId).touches_left && compStats(domId).touches_right;
end
if largestId > 0
    mask = component == largestId;
    row.largest_support_component_kinetic_fraction = localFrac(kinElem(mask), kinElem);
    row.largest_support_component_strain_fraction = localFrac(strElem(mask), strElem);
end
[row.classification, row.classification_reason] = localClassifyMode(row, lowThr);
end

function [cls, reason] = localClassifyMode(row, lowThr)
if row.low_density_kinetic_fraction >= 0.35 || row.low_density_strain_fraction >= 0.35
    cls = 'localized low-density mode';
    reason = sprintf('low-density energy: K %.3f, S %.3f', ...
        row.low_density_kinetic_fraction, row.low_density_strain_fraction);
elseif row.kinetic_effective_element_fraction < 0.025 || row.strain_effective_element_fraction < 0.025
    cls = 'localized non-low-density mode';
    reason = sprintf('effective fraction: K %.3f, S %.3f', ...
        row.kinetic_effective_element_fraction, row.strain_effective_element_fraction);
elseif row.largest_support_component_kinetic_fraction >= 0.60 && ...
        row.largest_support_component_strain_fraction >= 0.60 && row.dominant_component_touches_both_supports
    cls = 'physical global mode';
    reason = sprintf('support-spanning energy: K %.3f, S %.3f', ...
        row.largest_support_component_kinetic_fraction, row.largest_support_component_strain_fraction);
else
    cls = 'ambiguous';
    reason = sprintf('mixed indicators for rho<%.2g: ldK %.3f, ldS %.3f', ...
        lowThr, row.low_density_kinetic_fraction, row.low_density_strain_fraction);
end
end

function overall = localOverall(rows)
classes = string({rows.classification});
first3 = classes(1:min(3, numel(classes)));
loc = sum(classes == "localized low-density mode");
locNd = sum(classes == "localized non-low-density mode");
phys = sum(classes == "physical global mode");
amb = sum(classes == "ambiguous");
if any(first3 == "localized low-density mode")
    cls = 'likely localized/spurious low-density mode influence';
elseif any(first3 == "localized non-low-density mode")
    cls = 'localized but not in low-density region';
elseif all(first3 == "physical global mode")
    cls = 'likely physically valid';
else
    cls = 'ambiguous';
end
overall = struct('classification', cls, 'physical_global_count', phys, ...
    'localized_low_density_count', loc, 'localized_non_low_density_count', locNd, ...
    'ambiguous_count', amb);
end

function [component, stats] = localComponents(rho, nelx, nely, threshold)
solid = reshape(rho(:) >= threshold, nely, nelx);
component = zeros(nely, nelx);
stats = repmat(struct('id', 0, 'size', 0, 'touches_left', false, 'touches_right', false), 0, 1);
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
        st = struct('id', compId, 'size', 0, 'touches_left', false, 'touches_right', false);
        while head <= tail
            cy = q(head,1);
            cx = q(head,2);
            head = head + 1;
            st.size = st.size + 1;
            st.touches_left = st.touches_left || cx == 1;
            st.touches_right = st.touches_right || cx == nelx;
            nbr = [cy-1,cx; cy+1,cx; cy,cx-1; cy,cx+1];
            for ii = 1:4
                ny = nbr(ii,1);
                nx = nbr(ii,2);
                if ny >= 1 && ny <= nely && nx >= 1 && nx <= nelx && solid(ny,nx) && component(ny,nx) == 0
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
summary = struct('count', numel(stats), 'largest_support_connected_component_id', largestId);
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
        id = stats(i).id;
        best = stats(i).size;
    end
end
end

function [domId, fracK, fracS] = localDominantComponent(component, kinElem, strElem)
ids = unique(component(component > 0));
if isempty(ids)
    domId = NaN; fracK = NaN; fracS = NaN; return
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
    idx = NaN; eff = NaN;
else
    p = e / total;
    idx = sum(p.^2);
    eff = 1 / (numel(e) * idx);
end
end

function baseline = localLoadBaseline(convPath, s1Path, resultPath)
baseline = struct();
baseline.available = exist(convPath, 'file') == 2;
if baseline.available
    baseline.convergence = readtable(convPath);
end
if exist(s1Path, 'file') == 2
    baseline.s1 = jsondecode(fileread(s1Path));
end
if exist(resultPath, 'file') == 2
    baseline.result = jsondecode(fileread(resultPath));
end
end

function comparison = localCompare(baseline, phaseT, hist, s1)
comparison = struct();
comparison.baseline = struct();
if baseline.available
    b = baseline.convergence;
    comparison.baseline.inner_solver = localInnerStats(b.inner_iters, b.inner_iters >= 30);
    comparison.baseline.outer = localOuterStats(b.iteration, b.omega_1_rad_s, b.beta_rad_s, b.design_change_norm);
end
if isfield(baseline, 's1')
    comparison.baseline.s1 = baseline.s1.overall;
    comparison.baseline.mode1_ld_strain = baseline.s1.modes(1).low_density_strain_fraction;
    comparison.baseline.components = baseline.s1.components;
end
comparison.phase1.inner_solver = localInnerStats(phaseT.inner_iterations, phaseT.inner_hit_max_iter);
comparison.phase1.inner_solver.average_cpu_time_s = mean(phaseT.inner_cpu_time_s, 'omitnan');
comparison.phase1.inner_solver.total_cpu_time_s = sum(phaseT.inner_cpu_time_s, 'omitnan');
comparison.phase1.outer = localOuterStats(phaseT.outer_iteration, phaseT.omega1, phaseT.beta_rad_s, phaseT.design_change);
comparison.phase1.s1 = s1.overall;
comparison.phase1.mode1_ld_strain = s1.modes(1).low_density_strain_fraction;
comparison.phase1.components = s1.components;
comparison.phase1.final_design_change = hist.drho_norm(end);
if isfield(comparison.baseline, 'outer')
    comparison.reductions = struct();
    comparison.reductions.omega_parity_gap = comparison.baseline.outer.omega_parity_gap - comparison.phase1.outer.omega_parity_gap;
    comparison.reductions.beta_parity_gap = comparison.baseline.outer.beta_parity_gap - comparison.phase1.outer.beta_parity_gap;
    comparison.reductions.design_change_mean = comparison.baseline.outer.design_change_mean - comparison.phase1.outer.design_change_mean;
end
end

function stats = localInnerStats(innerIterations, hitCap)
stats = struct();
stats.average_inner_iterations = mean(innerIterations, 'omitnan');
stats.maximum_inner_iterations = max(innerIterations);
stats.minimum_inner_iterations = min(innerIterations);
stats.fraction_hitting_cap = mean(hitCap);
stats.count_hitting_cap = sum(hitCap);
stats.outer_iteration_count = numel(innerIterations);
end

function stats = localOuterStats(iter, omega1, betaRad, designChange)
window = min(80, numel(iter));
idx = (numel(iter)-window+1):numel(iter);
it = iter(idx);
even = mod(it, 2) == 0;
odd = ~even;
stats = struct();
stats.final_omega1 = omega1(end);
stats.final_beta_rad_s = betaRad(end);
stats.final_design_change = designChange(end);
stats.design_change_mean = mean(designChange(idx), 'omitnan');
stats.omega_even_mean = mean(omega1(idx(even)), 'omitnan');
stats.omega_odd_mean = mean(omega1(idx(odd)), 'omitnan');
stats.omega_parity_gap = abs(stats.omega_even_mean - stats.omega_odd_mean);
stats.beta_even_mean = mean(betaRad(idx(even)), 'omitnan');
stats.beta_odd_mean = mean(betaRad(idx(odd)), 'omitnan');
stats.beta_parity_gap = abs(stats.beta_even_mean - stats.beta_odd_mean);
end

function decision = localDecision(comparison)
innerConverged = comparison.phase1.inner_solver.fraction_hitting_cap == 0;
outerReduced = false;
if isfield(comparison, 'reductions')
    outerReduced = comparison.phase1.outer.omega_parity_gap < 1.0 && ...
        comparison.phase1.outer.design_change_mean < 0.02;
end
if innerConverged && outerReduced
    code = 'A';
    label = 'Inner solver now converges and outer oscillation is substantially reduced.';
elseif innerConverged
    code = 'B';
    label = 'Inner solver now converges but outer oscillation remains.';
elseif comparison.phase1.inner_solver.fraction_hitting_cap > 0
    code = 'C';
    label = 'Inner solver still reaches the cap.';
else
    code = 'D';
    label = 'Other unexpected behaviour.';
end
decision = struct('code', code, 'label', label);
end

function answers = localFinalAnswers(comparison, decision)
answers = struct();
answers.q1 = sprintf('No. Fraction hitting the inner cap is %.6g.', ...
    comparison.phase1.inner_solver.fraction_hitting_cap);
if isfield(comparison, 'reductions')
    answers.q2 = sprintf('No. Omega parity gap is %.6g rad/s and design-change mean is %.6g.', ...
        comparison.phase1.outer.omega_parity_gap, comparison.phase1.outer.design_change_mean);
else
    answers.q2 = 'Inconclusive: baseline comparison unavailable.';
end
if strcmp(decision.code, 'C')
    answers.q3 = 'The MMA inner subproblem still failing to converge within the enlarged cap.';
elseif strcmp(decision.code, 'B')
    answers.q3 = 'The outer fixed-point update/linearized-increment dynamics rather than the inner iteration cap.';
else
    answers.q3 = 'Unexpected behaviour; inspect inner convergence and outer update diagnostics.';
end
end

function localWriteSummary(path, result, s1, comparison, decision)
lines = {};
lines{end+1} = '# Phase 1 Inner-Max-Iter 300 Result';
lines{end+1} = '';
lines{end+1} = sprintf('Generated: %s', result.created_utc);
lines{end+1} = '';
lines{end+1} = '## Parameter Verification';
lines{end+1} = '';
lines{end+1} = sprintf('- pass: `%s`', string(result.parameter_verification.pass));
lines{end+1} = '- changed parameter: `inner_max_iter` 30 -> 300';
lines{end+1} = sprintf('- other differences: %d', numel(result.parameter_verification.differences_other_than_inner_max_iter));
lines{end+1} = '';
lines{end+1} = '## Decision';
lines{end+1} = '';
lines{end+1} = sprintf('**%s. %s**', decision.code, decision.label);
lines{end+1} = '';
lines{end+1} = '## Inner Solver';
lines{end+1} = '';
lines{end+1} = '| metric | inner=30 | inner=300 |';
lines{end+1} = '|---|---:|---:|';
lines{end+1} = sprintf('| average inner iterations | %.6g | %.6g |', ...
    comparison.baseline.inner_solver.average_inner_iterations, comparison.phase1.inner_solver.average_inner_iterations);
lines{end+1} = sprintf('| min inner iterations | %.6g | %.6g |', ...
    comparison.baseline.inner_solver.minimum_inner_iterations, comparison.phase1.inner_solver.minimum_inner_iterations);
lines{end+1} = sprintf('| max inner iterations | %.6g | %.6g |', ...
    comparison.baseline.inner_solver.maximum_inner_iterations, comparison.phase1.inner_solver.maximum_inner_iterations);
lines{end+1} = sprintf('| fraction hitting cap | %.6g | %.6g |', ...
    comparison.baseline.inner_solver.fraction_hitting_cap, comparison.phase1.inner_solver.fraction_hitting_cap);
lines{end+1} = sprintf('| average inner CPU time s | n/a | %.6g |', ...
    comparison.phase1.inner_solver.average_cpu_time_s);
lines{end+1} = '';
lines{end+1} = '## Outer Convergence';
lines{end+1} = '';
lines{end+1} = '| metric | inner=30 | inner=300 |';
lines{end+1} = '|---|---:|---:|';
lines{end+1} = sprintf('| omega parity gap | %.6g | %.6g |', ...
    comparison.baseline.outer.omega_parity_gap, comparison.phase1.outer.omega_parity_gap);
lines{end+1} = sprintf('| beta parity gap | %.6g | %.6g |', ...
    comparison.baseline.outer.beta_parity_gap, comparison.phase1.outer.beta_parity_gap);
lines{end+1} = sprintf('| mean design change, last window | %.6g | %.6g |', ...
    comparison.baseline.outer.design_change_mean, comparison.phase1.outer.design_change_mean);
lines{end+1} = sprintf('| final design change | %.6g | %.6g |', ...
    comparison.baseline.outer.final_design_change, comparison.phase1.outer.final_design_change);
lines{end+1} = '';
lines{end+1} = '## Final Solution';
lines{end+1} = '';
lines{end+1} = '| mode | omega rad/s | Hz |';
lines{end+1} = '|---:|---:|---:|';
for k = 1:6
    lines{end+1} = sprintf('| %d | %.8g | %.8g |', k, result.first_6_omega_rad_s(k), result.first_6_frequency_hz(k)); %#ok<AGROW>
end
lines{end+1} = '';
lines{end+1} = sprintf('- S1 overall: %s', s1.overall.classification);
lines{end+1} = sprintf('- localized low-density modes: %d/10', s1.overall.localized_low_density_count);
lines{end+1} = sprintf('- mode 1 ld_strain_frac: %.8g', s1.modes(1).low_density_strain_fraction);
lines{end+1} = sprintf('- support-connected components: %d', s1.components.support_connected_component_count);
lines{end+1} = '';
lines{end+1} = '## Final Section';
lines{end+1} = '';
lines{end+1} = sprintf('1. %s', result.final_answers.q1);
lines{end+1} = sprintf('2. %s', result.final_answers.q2);
lines{end+1} = sprintf('3. %s', result.final_answers.q3);
localWriteText(path, strjoin(lines, newline));
end

function localWriteComparison(path, comparison, s1, result)
lines = {};
lines{end+1} = '# Phase 1 Comparison: inner_max_iter 30 vs 300';
lines{end+1} = '';
lines{end+1} = '| category | metric | inner=30 alpha=0.5 | inner=300 alpha=0.5 |';
lines{end+1} = '|---|---|---:|---:|';
lines{end+1} = sprintf('| inner solver | average iterations | %.6g | %.6g |', comparison.baseline.inner_solver.average_inner_iterations, comparison.phase1.inner_solver.average_inner_iterations);
lines{end+1} = sprintf('| inner solver | max iterations | %.6g | %.6g |', comparison.baseline.inner_solver.maximum_inner_iterations, comparison.phase1.inner_solver.maximum_inner_iterations);
lines{end+1} = sprintf('| inner solver | cap fraction | %.6g | %.6g |', comparison.baseline.inner_solver.fraction_hitting_cap, comparison.phase1.inner_solver.fraction_hitting_cap);
lines{end+1} = sprintf('| outer convergence | omega parity gap | %.6g | %.6g |', comparison.baseline.outer.omega_parity_gap, comparison.phase1.outer.omega_parity_gap);
lines{end+1} = sprintf('| outer convergence | beta parity gap | %.6g | %.6g |', comparison.baseline.outer.beta_parity_gap, comparison.phase1.outer.beta_parity_gap);
lines{end+1} = sprintf('| outer convergence | final design change | %.6g | %.6g |', comparison.baseline.outer.final_design_change, comparison.phase1.outer.final_design_change);
lines{end+1} = sprintf('| S1 | localized modes | %d | %d |', comparison.baseline.s1.localized_low_density_count, s1.overall.localized_low_density_count);
lines{end+1} = sprintf('| S1 | mode 1 ld_strain_frac | %.6g | %.6g |', comparison.baseline.mode1_ld_strain, s1.modes(1).low_density_strain_fraction);
lines{end+1} = sprintf('| topology | support-connected components | %d | %d |', comparison.baseline.components.support_connected_component_count, s1.components.support_connected_component_count);
lines{end+1} = '';
lines{end+1} = sprintf('Decision: **%s**.', result.decision.label);
localWriteText(path, strjoin(lines, newline));
end

function localWriteManifest(path, outDir, result)
files = {'phase1_inner300_result.mat','phase1_inner300_result.json', ...
    'phase1_inner300_summary.md','phase1_inner300_manifest.json', ...
    'phase1_inner300_comparison.md','phase1_inner300_parameter_verification.json', ...
    'phase1_inner300_convergence_history.csv','phase1_inner300_multiplicity_history.csv', ...
    'phase1_inner300_omega_history.png','phase1_inner300_beta_history.png', ...
    'phase1_inner300_design_change_history.png','phase1_inner300_inner_iteration_histogram.png', ...
    'phase1_inner300_topology.csv','phase1_inner300_topology.png', ...
    's1_mode_summary.json','s1_modes.csv','phase1_inner300_run.log'};
modeFiles = cell(1, 6);
for k = 1:6
    modeFiles{k} = sprintf('phase1_inner300_mode_%02d_shape.png', k);
end
manifest = struct();
manifest.study = result.study;
manifest.created_utc = result.created_utc;
manifest.output_dir = outDir;
manifest.files = [files, modeFiles];
manifest.changed_parameter = result.changed_parameter;
manifest.decision = result.decision;
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
    error('Phase1Inner300:WriteFailed', 'Cannot write %s', path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', char(txt));
end
