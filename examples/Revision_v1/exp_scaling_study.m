function study = exp_scaling_study(outDir)
%EXP_SCALING_STUDY  Extended Exp3/S1 scaling study: onset of localized modes.
%
% Goal: characterise the onset of mesh-dependent localised low-density modes
% in the authoritative formulation. This is NOT a mesh-convergence study.
%
% Mesh series (8:1 aspect ratio, alpha=1.00 only, MATLAB only):
%   s=0.5  80x10  (N_e=   800)
%   s=1.0 160x20  (N_e= 3,200)
%   s=1.5 240x30  (N_e= 7,200)
%   s=2.0 320x40  (N_e=12,800)
%   s=2.5 400x50  (N_e=20,000) — reuses existing Exp3 result if accepted
%   s=3.0 480x60  (N_e=28,800)
%
% Per-mesh: full optimisation run + S1 postprocessing (10 modes, energy,
% localisation metrics, 6 mode-shape figures).
%
% Scope constraints: no alpha sweep, no CR2, no A4, no P1, no Python, no
% manuscript edits, no mesh-convergence claim.

if nargin < 1 || isempty(outDir)
    scriptDir = fileparts(mfilename('fullpath'));
    outDir = fullfile(scriptDir, 'output', 'exp_scaling_study');
end
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(fileparts(scriptDir));
addpath(fullfile(repoRoot, 'tools',     'Matlab'));
addpath(fullfile(repoRoot, 'analysis', 'ourApproach', 'Matlab'));

logPath = fullfile(outDir, 'exp_scaling_study_run.log');
diary(logPath);
cleanupDiary = onCleanup(@() diary('off'));

fprintf('exp_scaling_study started: %s\n', char(datetime('now','TimeZone','UTC')));
fprintf('MATLAB only. alpha=1.00 only. No alpha sweep, CR2, A4, P1, Python, manuscript edits.\n');
fprintf('Purpose: onset characterisation, NOT mesh convergence.\n\n');

% Physical filter radius fixed to 2 elements at the 200x25 reference mesh.
PHYS_FILTER_RADIUS = 2.0 * (8.0 / 200.0);  % = 0.08 m

criteria = struct( ...
    'mac_threshold',               0.8, ...
    'feasibility_tolerance',       1e-8, ...
    'design_change_tolerance',     0.001, ...
    'localized_onset_mac_below',   0.8, ...
    'localized_onset_localized_majority', true);

meshCases = [ ...
    struct('label','80x10',  's',0.5, 'config', fullfile(scriptDir,'clamped_beam_80x10.json')); ...
    struct('label','160x20', 's',1.0, 'config', fullfile(scriptDir,'clamped_beam_160x20.json')); ...
    struct('label','240x30', 's',1.5, 'config', fullfile(scriptDir,'clamped_beam_240x30.json')); ...
    struct('label','320x40', 's',2.0, 'config', fullfile(scriptDir,'clamped_beam_320x40.json')); ...
    struct('label','400x50', 's',2.5, 'config', fullfile(scriptDir,'clamped_beam_400x50.json')); ...
    struct('label','480x60', 's',3.0, 'config', fullfile(scriptDir,'clamped_beam_480x60.json'))];

% Existing Exp3 400x50 accepted result — reuse if present.
exp3MatPath = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
    'exp3_authoritative_mesh_convergence', 'mesh_400x50', ...
    'exp3_authoritative_400x50_result.mat');

allResults = repmat(scalingCaseInit(), numel(meshCases), 1);

for i = 1:numel(meshCases)
    mc = meshCases(i);
    caseOutDir = fullfile(outDir, ['mesh_', mc.label]);
    if exist(caseOutDir, 'dir') ~= 7; mkdir(caseOutDir); end

    fprintf('\n=== Mesh %s (s=%.1f, N_e=%d) ===\n', mc.label, mc.s, ...
        localNelx(mc.config) * localNely(mc.config));

    % --- Try to reuse existing Exp3 400x50 result ---
    reuseOk = false;
    if strcmp(mc.label, '400x50') && exist(exp3MatPath, 'file') == 2
        try
            S = load(exp3MatPath, 'result', 'cfg', 'xFinal', 'omega', 'info');
            if isfield(S,'result') && isfield(S.result,'accepted') && S.result.accepted
                fprintf('  Reusing accepted Exp3 400x50 result from: %s\n', exp3MatPath);
                allResults(i) = scalingCaseFromExp3(S, mc, PHYS_FILTER_RADIUS, repoRoot);
                reuseOk = true;
            end
        catch ME
            fprintf('  Exp3 reuse failed: %s — running fresh.\n', ME.message);
        end
    end

    if ~reuseOk
        allResults(i) = localRunOptCase(mc, caseOutDir, PHYS_FILTER_RADIUS, criteria, repoRoot);
    end

    % --- S1 postprocessing ---
    s1OutDir = fullfile(caseOutDir, 's1_postprocessing');
    if exist(s1OutDir, 'dir') ~= 7; mkdir(s1OutDir); end
    allResults(i) = localRunS1(allResults(i), mc, s1OutDir, PHYS_FILTER_RADIUS, repoRoot);

    fprintf('  Opt classification: %s | S1 overall: %s\n', ...
        allResults(i).classification, allResults(i).s1_overall_classification);
end

study = struct();
study.study       = 'Extended Exp3/S1 scaling study: onset of localised modes';
study.purpose     = 'Characterise onset of mesh-dependent localised low-density modes. NOT mesh convergence.';
study.scope_excluded = {'alpha sweep','CR2','A4','P1','Python','manuscript edits','mesh convergence claim'};
study.alpha       = 1.0;
study.phys_filter_radius = PHYS_FILTER_RADIUS;
study.criteria    = criteria;
study.cases       = allResults;
study.onset_mesh  = localOnsetMesh(allResults, criteria);
study.created_utc = char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'));

localWriteJson(fullfile(outDir, 'exp_scaling_study_result.json'), localJsonSafe(study));
localWriteSummary(fullfile(outDir, 'exp_scaling_study_summary.md'), study);
localWriteManifest(fullfile(outDir, 'exp_scaling_study_manifest.json'), study, outDir, repoRoot);

fprintf('\n=== Scaling study complete ===\n');
fprintf('Onset mesh: %s\n', study.onset_mesh);
fprintf('Summary: %s\n', fullfile(outDir, 'exp_scaling_study_summary.md'));
diary('off');
end

% =========================================================================
%  Optimisation runner
% =========================================================================

function result = localRunOptCase(mc, caseOutDir, physFilterRadius, criteria, repoRoot)
result = scalingCaseInit();
result.mesh_label = mc.label;
result.s          = mc.s;
result.source_config = mc.config;

% Reuse existing result if already computed and MAT exists
existingMat = fullfile(caseOutDir, ['scaling_', mc.label, '_result.mat']);
if exist(existingMat, 'file') == 2
    try
        fprintf('  Reusing existing result: %s\n', existingMat);
        S = load(existingMat, 'result', 'cfg', 'xFinal', 'omega', 'info');
        result = S.result;
        result.result_mat_abs = existingMat;
        result.result_mat = localRel(existingMat, repoRoot);
        return;
    catch ME2
        fprintf('  Reuse failed (%s) — rerunning.\n', ME2.message);
    end
end

cfg = localBuildConfig(mc.config, physFilterRadius, criteria);
cfgPath = fullfile(caseOutDir, ['scaling_', mc.label, '_config.json']);
localWriteJson(cfgPath, cfg);

nelx = double(cfg.domain.mesh.nelx);
nely = double(cfg.domain.mesh.nely);
result.nelx = nelx;
result.nely = nely;
result.n_elements = nelx * nely;
result.filter_physical_radius = physFilterRadius;

xFinal = []; omega = NaN(3,1); info = struct(); tIter = NaN; %#ok<NASGU>
runTic = tic;
try
    [xFinal, omega, tIter, nIter, memUsage, info] = run_topopt_from_json(cfg);
    result.solver_success  = true;
    result.iterations      = double(nIter);
    result.timing_total_s  = toc(runTic);
    result.timing_iter_s   = tIter;
    result.peak_memory_MB  = memUsage;
    result.omega_rad_s     = omega(:);
    result.omega_1_rad_s   = omega(1);
    result.grayness        = mean(4 * xFinal(:) .* (1 - xFinal(:)));
    result.volume          = mean(xFinal(:));
    result.feasibility     = max(0, result.volume - double(cfg.optimization.volume_fraction));
    result.design_change   = localFinalHist(info, {'cr2_history','design_change'});
    result.tracked_mac     = localFinalHist(info, {'cr2_final_tracking','tracked_mode_mac'});
    result.tracked_omega   = localFinalHist(info, {'cr2_final_tracking','tracked_mode_omega'});
    result.tracked_index   = localFinalHist(info, {'cr2_final_tracking','tracked_mode_index'});
    result.a5_pass         = localA5Pass(info);
    result.classification  = localClassify(result, cfg, criteria);
    result.accepted        = strcmp(result.classification, 'accepted');

    localSaveArtifacts(caseOutDir, mc.label, xFinal, omega, info, cfg, result);
catch ME
    result.timing_total_s = toc(runTic);
    result.exception      = getReport(ME, 'extended', 'hyperlinks', 'off');
    result.classification = 'implementation failure';
    result.accepted       = false;
    fprintf(2, '  Exception [%s]: %s\n', mc.label, ME.message);
end

matPath = fullfile(caseOutDir, ['scaling_', mc.label, '_result.mat']);
save(matPath, 'result', 'cfg', 'xFinal', 'omega', 'info', '-v7.3');
result.result_mat = localRel(matPath, repoRoot);
localWriteJson(fullfile(caseOutDir, ['scaling_', mc.label, '_result.json']), localJsonSafe(result));
end

% =========================================================================
%  S1 postprocessing
% =========================================================================

function result = localRunS1(result, mc, s1OutDir, physFilterRadius, repoRoot)
% Load saved MAT (find it from result.result_mat or expect caseOutDir)
result.s1_overall_classification = 'not run';
result.s1_physical_global_count  = NaN;
result.s1_localized_count        = NaN;
result.s1_ambiguous_count        = NaN;
result.s1_disconnected_count     = NaN;
result.s1_mode1_classification   = '';
result.s1_mode1_low_density_strain_fraction = NaN;

% Locate the MAT file
matPath = '';
if isfield(result, 'result_mat') && ~isempty(result.result_mat)
    candidate = fullfile(repoRoot, result.result_mat);
    if exist(candidate, 'file') == 2
        matPath = candidate;
    end
end
% Also check if it's an absolute path stored directly
if isempty(matPath) && isfield(result, 'result_mat_abs')
    if exist(result.result_mat_abs, 'file') == 2
        matPath = result.result_mat_abs;
    end
end
if isempty(matPath)
    fprintf(2, '  [S1 %s] Cannot find result MAT file — skipping S1.\n', mc.label);
    return;
end

try
    S = load(matPath, 'cfg', 'xFinal', 'info', 'result');
    cfg    = S.cfg;
    xFinal = S.xFinal(:);
    savedInfo   = S.info;
    savedResult = S.result;

    reportStem = ['scaling_', mc.label, '_s1'];
    s1 = localS1Diagnose(cfg, xFinal, savedInfo, savedResult, s1OutDir, reportStem, repoRoot);

    result.s1_overall_classification          = s1.overall.classification;
    result.s1_physical_global_count           = s1.overall.physical_global_count;
    result.s1_localized_count                 = s1.overall.localized_low_density_count;
    result.s1_ambiguous_count                 = s1.overall.ambiguous_count;
    result.s1_disconnected_count              = s1.overall.disconnected_component_count;
    if numel(s1.modes) >= 1
        result.s1_mode1_classification              = s1.modes(1).classification;
        result.s1_mode1_low_density_strain_fraction = s1.modes(1).low_density_strain_fraction;
    end
    result.s1_artifacts = s1.artifacts;
    localWriteJson(fullfile(s1OutDir, [reportStem, '_mode_summary.json']), s1);
catch ME
    fprintf(2, '  [S1 %s] Exception: %s\n', mc.label, ME.message);
    result.s1_exception = getReport(ME, 'extended', 'hyperlinks', 'off');
end
end

function s1 = localS1Diagnose(cfg, xFinal, savedInfo, savedResult, outDir, reportStem, repoRoot)
L  = double(cfg.domain.size.length);
H  = double(cfg.domain.size.height);
nelx = double(cfg.domain.mesh.nelx);
nely = double(cfg.domain.mesh.nely);
thickness = double(cfg.domain.thickness);
E0  = double(cfg.material.E);
nu  = double(cfg.material.nu);
rho0 = double(cfg.material.rho);
Emin = E0 * double(cfg.void_material.E_min_ratio);
rho_min = double(cfg.void_material.rho_min);
penal   = double(cfg.optimization.penalization);
massInterp = s1MassInterp(cfg);
nModes = 10;
lowThr = 0.05;
solidThr = 0.5;

fprintf('  [S1] %dx%d, nelx=%d nely=%d, pmass=%.1f\n', nelx, nely, nelx, nely, massInterp.pmass);

[K, M, KE, ME, edofMat, elemCenters] = s1AssembleKM( ...
    xFinal, nelx, nely, L, H, thickness, E0, Emin, nu, rho0, rho_min, penal, massInterp);
[fixedDofs, ~] = supportsToFixedDofs(cfg.bc.supports, nelx, nely, L, H);
free = setdiff((1:size(K,1))', fixedDofs(:));

Kf = K(free,free);
Mf = M(free,free);
eigOpts = struct('disp',0,'maxit',3000,'tol',1e-12);
[V, D] = eigs(Kf, Mf, nModes, 'smallestabs', eigOpts);
lam = real(diag(D));
[lam, ord] = sort(lam,'ascend');
V   = V(:,ord);
valid = isfinite(lam) & lam > 0;
lam = lam(valid);
V   = V(:,valid);
nModes = min(nModes, numel(lam));
V   = V(:,1:nModes);
lam = lam(1:nModes);
V   = mass_normalize_modes(V, Mf);
V   = orient_modes_deterministic(V);

Phi = zeros(size(K,1), nModes);
Phi(free,:) = V;
omega = sqrt(lam);
freqHz = omega / (2*pi);

targetMode = savedInfo.gate_a0.reference_modes(:,1);
macToRef = squared_mass_weighted_mac(targetMode, Phi, M);
macToRef = macToRef(1,:)';

[component, compStats] = s1Components(xFinal, nelx, nely, solidThr);
largestId = s1LargestSupport(compStats);
lowMask   = xFinal < lowThr;

modeRows = repmat(s1BlankRow(), nModes, 1);
pngPaths = cell(min(6,nModes),1);
for k = 1:nModes
    phi = Phi(:,k);
    [kinElem, strElem] = s1ElemEnergies(phi, omega(k), xFinal, edofMat, KE, ME, ...
        E0, Emin, rho0, rho_min, penal, massInterp);
    modeRows(k) = s1DiagnoseMode(k, omega(k), freqHz(k), phi, M, K, free, edofMat, ...
        macToRef(k), kinElem, strElem, xFinal, elemCenters, component, compStats, ...
        largestId, lowMask, lowThr);
    % Write per-mode energy CSV
    s1WriteEnergyCSV(fullfile(outDir, sprintf('%s_mode_%02d_energy.csv', reportStem, k)), ...
        xFinal, elemCenters, component, compStats, kinElem, strElem, modeRows(k));
    % Save mode-shape figures for first 6
    if k <= 6
        pngPath = fullfile(outDir, sprintf('%s_mode_%02d_shape.png', reportStem, k));
        s1SaveModeShape(pngPath, phi, xFinal, nelx, nely, L, H, omega(k), k, modeRows(k));
        pngPaths{k} = localRel(pngPath, repoRoot);
    end
end

modeTable = struct2table(modeRows);
csvPath = fullfile(outDir, [reportStem, '_modes.csv']);
writetable(modeTable, csvPath);

overall = s1Overall(modeRows);

s1.study    = ['S1 scaling study mode diagnostic: ', reportStem];
s1.scope    = 'Postprocessing-only mode diagnosis; no optimisation rerun';
s1.created_utc = char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'));
s1.mesh     = struct('nelx',nelx,'nely',nely,'L',L,'H',H,'hx',L/nelx,'hy',H/nely);
s1.material = struct('E0',E0,'Emin',Emin,'nu',nu,'rho0',rho0,'rho_min',rho_min, ...
    'penal',penal,'mass_interpolation_mode',massInterp.mode,'pmass',massInterp.pmass);
s1.thresholds = struct('low_density',lowThr,'solid_component',solidThr);
s1.components = struct('count',numel(compStats),'largest_support_id',largestId);
if ~isempty(compStats)
    s1.components.support_connected_count = sum([compStats.touches_left] & [compStats.touches_right]);
else
    s1.components.support_connected_count = 0;
end
if isfield(savedResult, 'final')
    refMac   = savedResult.final.tracked_mode_mac;
    refOmega = savedResult.final.tracked_mode_omega;
    refOmeg1 = savedResult.final.omega_rad_s(1);
else
    refMac   = savedResult.tracked_mac;
    refOmega = savedResult.tracked_omega;
    refOmeg1 = savedResult.omega_1_rad_s;
end
s1.reference = struct( ...
    'saved_final_tracked_mac',   refMac, ...
    'saved_final_tracked_omega', refOmega, ...
    'saved_final_omega_1',       refOmeg1);
s1.modes   = modeRows;
s1.overall = overall;
s1.artifacts = struct('modes_csv', localRel(csvPath, repoRoot), 'mode_png', {pngPaths});
end

% =========================================================================
%  Config builder
% =========================================================================

function cfg = localBuildConfig(configPath, physFilterRadius, criteria)
cfg = jsondecode(fileread(configPath));
nelx = double(cfg.domain.mesh.nelx);
nely = double(cfg.domain.mesh.nely);
cfg.meta.name = sprintf('Scaling study alpha=1.00 %dx%d', nelx, nely);
cfg.meta.notes = ['Extended Exp3/S1 scaling study. Authoritative load F=omega0^2*M*Phi0, ', ...
    'solid reference, load_sensitivity=omitted, Gate A0 diagnostics enabled.'];

cfg.domain.load_cases(1).name   = 'alpha1.00_solid_reference_mode_1';
cfg.domain.load_cases(1).factor = 1.0;
cfg.domain.load_cases(1).loads(1).type   = 'semi_harmonic';
cfg.domain.load_cases(1).loads(1).mode   = 1;
cfg.domain.load_cases(1).loads(1).factor = 1.0;
cfg.domain.load_cases(2).name   = 'alpha0.00_solid_reference_mode_2';
cfg.domain.load_cases(2).factor = 0.0;
cfg.domain.load_cases(2).loads(1).type   = 'semi_harmonic';
cfg.domain.load_cases(2).loads(1).mode   = 2;
cfg.domain.load_cases(2).loads(1).factor = 1.0;

cfg.optimization.semi_harmonic_baseline = 'solid';
if isfield(cfg.optimization, 'semi_harmonic_rho_source')
    cfg.optimization = rmfield(cfg.optimization, 'semi_harmonic_rho_source');
end
cfg.optimization.harmonic_normalize  = false;
cfg.optimization.load_sensitivity    = 'omitted';
cfg.optimization.gate_a0_diagnostics = true;
cfg.optimization.convergence_tol     = criteria.design_change_tolerance;

cfg.optimization.filter.radius       = physFilterRadius;
cfg.optimization.filter.radius_units = 'physical';

cfg.postprocessing.compute_modes             = 1;
cfg.postprocessing.compute_modes_initial     = 0;
cfg.postprocessing.visualize_live            = false;
cfg.postprocessing.visualize_modes.enabled   = false;
cfg.postprocessing.visualize_modes.count     = 0;
cfg.postprocessing.visualize_topology_modes.enabled = false;
cfg.postprocessing.visualize_topology_modes.count   = 0;
cfg.postprocessing.save_snapshot_image       = false;
cfg.postprocessing.save_final_image          = false;
cfg.postprocessing.save_frequency_iterations = false;
cfg.postprocessing.write_correlation_table   = false;
cfg.postprocessing.correlation.enabled       = false;
cfg.postprocessing.correlation.initial_count = 0;
cfg.postprocessing.correlation.topology_count = 0;
cfg.postprocessing.correlation.metric        = 'mac';
cfg.postprocessing.correlation.write_csv     = false;
end

% =========================================================================
%  Artifact helpers
% =========================================================================

function localSaveArtifacts(caseOutDir, label, xFinal, omega, info, cfg, result)
prefix = fullfile(caseOutDir, ['scaling_', label]);
nelx = double(cfg.domain.mesh.nelx);
nely = double(cfg.domain.mesh.nely);

% Topology
topo = reshape(xFinal(:), nely, nelx);
writematrix(topo, [prefix, '_topology.csv']);

% History tables
if isfield(info,'cr2_history')
    h = info.cr2_history;
    n = size(h.objective,1);
    it = (1:n)';
    writetable(table(it, h.objective(:), h.design_change(:), h.volume(:), ...
        h.feasibility(:), h.grayness(:), h.tracked_mode_index(:), ...
        h.tracked_mode_mac(:), h.tracked_mode_omega(:), ...
        'VariableNames',{'iteration','objective','design_change','volume', ...
        'feasibility','grayness','tracked_mode_index','tracked_mode_mac', ...
        'tracked_mode_omega_rad_s'}), [prefix, '_convergence.csv']);
end

% Topology figure
fig = figure('Color','white','Visible','off');
ax  = axes('Parent',fig);
imagesc(ax, 1 - topo);
axis(ax,'equal','off');
colormap(ax,gray);
title(ax, sprintf('Scaling %s alpha=1.00', label), 'Interpreter','none');
try
    exportgraphics(fig, [prefix, '_topology.png'], 'Resolution',180, 'BackgroundColor','white');
catch
    print(fig, [prefix, '_topology.png'], '-dpng', '-r180');
end
close(fig);
end

% =========================================================================
%  Summary and manifest writers
% =========================================================================

function localWriteSummary(path, study)
lines = {};
lines{end+1} = '# Extended Exp3/S1 Scaling Study: Onset of Localised Modes';
lines{end+1} = '';
lines{end+1} = 'Purpose: characterise onset of mesh-dependent localised low-density modes.';
lines{end+1} = 'This is NOT a mesh-convergence study. MATLAB only. alpha=1.00 only.';
lines{end+1} = sprintf('Physical filter radius: %.4g m (= 2 elements at 200x25 reference mesh).', ...
    study.phys_filter_radius);
lines{end+1} = sprintf('Created: %s', study.created_utc);
lines{end+1} = '';
lines{end+1} = sprintf('**Onset mesh (first MAC<0.8 or majority localised): %s**', study.onset_mesh);
lines{end+1} = '';
lines{end+1} = '## Results Table';
lines{end+1} = '';
lines{end+1} = ['| s | mesh | N_e | classification | iters | omega_1 (rad/s) | MAC | grayness | ', ...
    's1_overall | phys | loc | amb |'];
lines{end+1} = ['|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|'];
for i = 1:numel(study.cases)
    r = study.cases(i);
    lines{end+1} = sprintf( ...
        '| %.1f | %s | %d | %s | %.0f | %.4g | %.4g | %.3g | %s | %.0f | %.0f | %.0f |', ...
        r.s, r.mesh_label, r.n_elements, r.classification, r.iterations, ...
        r.omega_1_rad_s, r.tracked_mac, r.grayness, ...
        r.s1_overall_classification, r.s1_physical_global_count, ...
        r.s1_localized_count, r.s1_ambiguous_count); %#ok<AGROW>
end
lines{end+1} = '';
lines{end+1} = '## S1 Mode 1 Detail';
lines{end+1} = '';
lines{end+1} = '| s | mesh | mode1 class | mode1 low-density strain frac |';
lines{end+1} = '|---|---|---|---:|';
for i = 1:numel(study.cases)
    r = study.cases(i);
    lines{end+1} = sprintf('| %.1f | %s | %s | %.4g |', ...
        r.s, r.mesh_label, r.s1_mode1_classification, ...
        r.s1_mode1_low_density_strain_fraction); %#ok<AGROW>
end
lines{end+1} = '';
lines{end+1} = '## Notes';
lines{end+1} = '';
lines{end+1} = '- 400x50 (s=2.5) result reused from Exp3 if previously accepted.';
lines{end+1} = '- Onset defined as: first mesh where MAC<0.8 OR majority of first 10 modes are localised.';
lines{end+1} = '- No mesh-convergence claim is made. No Exp2 alpha sweep, CR2, A4, P1, Python, or manuscript edits.';
localWriteText(path, strjoin(lines, newline));
end

function localWriteManifest(path, study, outDir, repoRoot)
m = struct();
m.study       = study.study;
m.purpose     = study.purpose;
m.created_utc = study.created_utc;
m.scope_excluded = study.scope_excluded;
m.onset_mesh  = study.onset_mesh;
m.cases = struct('label', {}, 'classification', {}, 'accepted', {}, ...
    'n_elements', {}, 'iterations', {}, 'omega_1_rad_s', {}, ...
    'tracked_mac', {}, 's1_overall', {});
for i = 1:numel(study.cases)
    r = study.cases(i);
    m.cases(i).label          = r.mesh_label;
    m.cases(i).classification = r.classification;
    m.cases(i).accepted       = r.accepted;
    m.cases(i).n_elements     = r.n_elements;
    m.cases(i).iterations     = r.iterations;
    m.cases(i).omega_1_rad_s  = r.omega_1_rad_s;
    m.cases(i).tracked_mac    = r.tracked_mac;
    m.cases(i).s1_overall     = r.s1_overall_classification;
end
localWriteJson(path, m);
end

% =========================================================================
%  S1 sub-functions (inline, no dependency on separate script)
% =========================================================================

function row = s1BlankRow()
row = struct( ...
    'mode', NaN, 'omega_rad_s', NaN, 'frequency_hz', NaN, ...
    'mac_to_solid_reference_mode_1', NaN, ...
    'modal_mass', NaN, 'modal_mass_residual', NaN, ...
    'stiffness_rayleigh', NaN, 'eigen_residual_relative', NaN, ...
    'kinetic_energy_total', NaN, 'strain_energy_total', NaN, ...
    'low_density_kinetic_fraction', NaN, 'low_density_strain_fraction', NaN, ...
    'low_density_displacement_fraction', NaN, ...
    'kinetic_localization_index', NaN, 'strain_localization_index', NaN, ...
    'kinetic_effective_element_fraction', NaN, 'strain_effective_element_fraction', NaN, ...
    'kinetic_top_1pct_fraction', NaN, 'strain_top_1pct_fraction', NaN, ...
    'kinetic_top_5pct_fraction', NaN, 'strain_top_5pct_fraction', NaN, ...
    'dominant_solid_component_id', NaN, ...
    'dominant_solid_component_kinetic_fraction', NaN, ...
    'dominant_solid_component_strain_fraction', NaN, ...
    'dominant_component_touches_left_support', false, ...
    'dominant_component_touches_right_support', false, ...
    'dominant_component_touches_both_supports', false, ...
    'largest_support_component_kinetic_fraction', NaN, ...
    'largest_support_component_strain_fraction', NaN, ...
    'classification', '', 'classification_reason', '');
end

function mi = s1MassInterp(cfg)
mi = struct('mode','power','pmass',1.0);
if isfield(cfg,'optimization') && isfield(cfg.optimization,'mass_interpolation') && ...
        isstruct(cfg.optimization.mass_interpolation)
    m = cfg.optimization.mass_interpolation;
    if isfield(m,'mode')  && ~isempty(m.mode);  mi.mode  = char(string(m.mode));  end
    if isfield(m,'pmass') && ~isempty(m.pmass); mi.pmass = double(m.pmass); end
end
modeKey = lower(strtrim(mi.mode));
if any(strcmp(modeKey,{'','power','simp_power','pmass'}))
    mi.mode = 'power';
elseif strcmp(modeKey,'linear')
    mi.mode = 'linear'; mi.pmass = 1.0;
elseif any(strcmp(modeKey,{'du2007_c1','du_olhoff_c1','eq4b'}))
    mi.mode = 'du2007_c1';
end
end

function [K, M, KE, ME, edofMat, elemCenters] = s1AssembleKM( ...
    x, nelx, nely, L, H, thickness, E0, Emin, nu, rho0, rho_min, penal, mi)
hx = L/nelx; hy = H/nely;
ndof = 2*(nelx+1)*(nely+1);
KE = thickness * s1Q4K(hx, hy, nu);
ME = thickness * s1Q4M(hx, hy);
edofMat = s1EdofMat(nelx, nely);
iK = reshape(kron(edofMat, ones(1,8))', [], 1);
jK = reshape(kron(edofMat, ones(8,1))', [], 1);
Eelem = Emin + x(:)'.^penal * (E0 - Emin);
[rhoElem, ~] = our_mass_interpolation(x(:)', rho0, rho_min, mi.mode, mi.pmass);
sK = reshape(KE(:) * Eelem, [], 1);
sM = reshape(ME(:) * rhoElem, [], 1);
K = sparse(iK, jK, sK, ndof, ndof);
M = sparse(iK, jK, sM, ndof, ndof);
K = (K+K')/2; M = (M+M')/2;
elemCenters = zeros(nelx*nely, 2);
for elx = 0:nelx-1
    for ely = 0:nely-1
        e = ely + elx*nely + 1;
        elemCenters(e,:) = [(elx+0.5)*hx, (ely+0.5)*hy];
    end
end
end

function edofMat = s1EdofMat(nelx, nely)
edofMat = zeros(nelx*nely, 8);
for elx = 0:nelx-1
    for ely = 0:nely-1
        el = ely + elx*nely + 1;
        n1 = (nely+1)*elx + ely;
        n2 = (nely+1)*(elx+1) + ely;
        edofMat(el,:) = [2*n1+1, 2*n1+2, 2*n2+1, 2*n2+2, ...
                         2*(n2+1)+1, 2*(n2+1)+2, 2*(n1+1)+1, 2*(n1+1)+2];
    end
end
end

function KE = s1Q4K(hx, hy, nu)
E = 1.0;
D = (E/(1-nu^2))*[1,nu,0; nu,1,0; 0,0,0.5*(1-nu)];
invJ = [2/hx,0; 0,2/hy];
detJ = 0.25*hx*hy;
gp = 1/sqrt(3); gpts = [-gp, gp];
KE = zeros(8,8);
for xi = gpts
    for eta = gpts
        dN_dxi  = 0.25*[-(1-eta), (1-eta), (1+eta), -(1+eta)];
        dN_deta = 0.25*[-(1-xi), -(1+xi),  (1+xi),   (1-xi)];
        dN_xy = invJ*[dN_dxi; dN_deta];
        dNx = dN_xy(1,:); dNy = dN_xy(2,:);
        B = zeros(3,8);
        B(1,1:2:end) = dNx; B(2,2:2:end) = dNy;
        B(3,1:2:end) = dNy; B(3,2:2:end) = dNx;
        KE = KE + (B'*D*B)*detJ;
    end
end
end

function ME = s1Q4M(hx, hy)
a = hx*hy;
Ms = (a/36)*[4,2,1,2; 2,4,2,1; 1,2,4,2; 2,1,2,4];
ME = kron(Ms, eye(2));
end

function [kinElem, strElem] = s1ElemEnergies(phi, omega, x, edofMat, KE, ME, ...
    E0, Emin, rho0, rho_min, penal, mi)
nEl = numel(x);
kinElem = zeros(nEl,1); strElem = zeros(nEl,1);
for e = 1:nEl
    ue = phi(edofMat(e,:));
    keS = Emin + x(e)^penal*(E0-Emin);
    [rhoS, ~] = our_mass_interpolation(x(e), rho0, rho_min, mi.mode, mi.pmass);
    strElem(e) = 0.5*real(ue'*(keS*KE)*ue);
    kinElem(e) = 0.5*omega^2*real(ue'*(rhoS*ME)*ue);
end
end

function row = s1DiagnoseMode(k, omega, freqHz, phi, M, K, free, edofMat, macVal, ...
    kinElem, strElem, x, elemCenters, component, compStats, largestId, lowMask, lowThr)
row = s1BlankRow();
row.mode = k; row.omega_rad_s = omega; row.frequency_hz = freqHz;
row.mac_to_solid_reference_mode_1 = macVal;
row.modal_mass = real(phi'*(M*phi));
row.modal_mass_residual = abs(row.modal_mass - 1);
row.stiffness_rayleigh = real(phi'*(K*phi));
res = K(free,:)*phi - omega^2*(M(free,:)*phi);
row.eigen_residual_relative = norm(res)/max(norm(K(free,:)*phi),eps);
row.kinetic_energy_total = sum(kinElem);
row.strain_energy_total  = sum(strElem);
row.low_density_kinetic_fraction = s1Frac(kinElem(lowMask), kinElem);
row.low_density_strain_fraction  = s1Frac(strElem(lowMask),  strElem);
dispElem = arrayfun(@(e) sum(phi(edofMat(e,:)).^2), (1:numel(x))');
row.low_density_displacement_fraction = s1Frac(dispElem(lowMask), dispElem);
[row.kinetic_localization_index, row.kinetic_effective_element_fraction, ...
    row.kinetic_top_1pct_fraction, row.kinetic_top_5pct_fraction] = s1Loc(kinElem);
[row.strain_localization_index, row.strain_effective_element_fraction, ...
    row.strain_top_1pct_fraction, row.strain_top_5pct_fraction] = s1Loc(strElem);
[domId, domK, domS] = s1DomComp(component, kinElem, strElem);
row.dominant_solid_component_id = domId;
row.dominant_solid_component_kinetic_fraction = domK;
row.dominant_solid_component_strain_fraction  = domS;
if domId > 0 && domId <= numel(compStats)
    row.dominant_component_touches_left_support  = compStats(domId).touches_left;
    row.dominant_component_touches_right_support = compStats(domId).touches_right;
    row.dominant_component_touches_both_supports = compStats(domId).touches_left && compStats(domId).touches_right;
end
if largestId > 0
    suppMask = component == largestId;
    row.largest_support_component_kinetic_fraction = s1Frac(kinElem(suppMask), kinElem);
    row.largest_support_component_strain_fraction  = s1Frac(strElem(suppMask),  strElem);
end
[row.classification, row.classification_reason] = s1ClassifyMode(row);
end

function [cls, reason] = s1ClassifyMode(row)
if row.low_density_kinetic_fraction >= 0.35 || row.low_density_strain_fraction >= 0.35
    cls    = 'localized low-density mode';
    reason = sprintf('low-density energy fractions: kinetic %.3f, strain %.3f', ...
        row.low_density_kinetic_fraction, row.low_density_strain_fraction);
elseif row.kinetic_effective_element_fraction < 0.025 || row.strain_effective_element_fraction < 0.025
    if ~row.dominant_component_touches_both_supports && ...
            (row.dominant_solid_component_kinetic_fraction >= 0.25 || ...
             row.dominant_solid_component_strain_fraction  >= 0.25)
        cls    = 'disconnected-component mode';
        reason = sprintf('localized; dominant component %d not support-spanning (K %.3f, S %.3f)', ...
            row.dominant_solid_component_id, row.dominant_solid_component_kinetic_fraction, ...
            row.dominant_solid_component_strain_fraction);
    else
        cls    = 'ambiguous';
        reason = sprintf('localized by participation but not clearly typed: effK %.3f, effS %.3f', ...
            row.kinetic_effective_element_fraction, row.strain_effective_element_fraction);
    end
elseif row.largest_support_component_kinetic_fraction >= 0.70 && ...
        row.largest_support_component_strain_fraction  >= 0.70 && ...
        row.dominant_component_touches_both_supports
    cls    = 'physical global mode';
    reason = sprintf('energy on support-spanning component: K %.3f, S %.3f', ...
        row.largest_support_component_kinetic_fraction, row.largest_support_component_strain_fraction);
else
    cls    = 'ambiguous';
    reason = sprintf('mixed indicators: lowK %.3f, lowS %.3f, supportK %.3f, supportS %.3f', ...
        row.low_density_kinetic_fraction, row.low_density_strain_fraction, ...
        row.largest_support_component_kinetic_fraction, row.largest_support_component_strain_fraction);
end
end

function overall = s1Overall(modeRows)
classes   = string({modeRows.classification});
physCount = sum(classes == "physical global mode");
locCount  = sum(classes == "localized low-density mode");
discCount = sum(classes == "disconnected-component mode");
ambCount  = sum(classes == "ambiguous");
first3    = classes(1:min(3,numel(classes)));
if any(first3 == "localized low-density mode")
    cls    = 'likely localized/spurious low-density mode influence';
    reason = 'At least one of the first three modes is localized low-density.';
elseif any(first3 == "disconnected-component mode")
    cls    = 'likely disconnected-component pathology';
    reason = 'At least one of the first three modes is disconnected-component.';
elseif all(first3 == "physical global mode")
    cls    = 'likely physically valid';
    reason = 'First three modes are physical global modes.';
else
    cls    = 'ambiguous';
    reason = 'First modes do not clearly separate pathologies.';
end
overall = struct('classification', cls, 'reason', reason, ...
    'physical_global_count', physCount, ...
    'localized_low_density_count', locCount, ...
    'disconnected_component_count', discCount, ...
    'ambiguous_count', ambCount);
end

function [component, stats] = s1Components(x, nelx, nely, threshold)
solid     = reshape(x(:) >= threshold, nely, nelx);
component = zeros(nely, nelx);
stats = repmat(struct('id',0,'size',0,'touches_left',false,'touches_right',false, ...
    'touches_bottom',false,'touches_top',false), 0, 1);
compId = 0;
for ix = 1:nelx
    for iy = 1:nely
        if ~solid(iy,ix) || component(iy,ix) ~= 0; continue; end
        compId = compId + 1;
        q = zeros(nnz(solid),2); head=1; tail=1;
        q(tail,:) = [iy,ix]; component(iy,ix) = compId;
        st = struct('id',compId,'size',0,'touches_left',false,'touches_right',false, ...
            'touches_bottom',false,'touches_top',false);
        while head <= tail
            cy=q(head,1); cx=q(head,2); head=head+1;
            st.size = st.size+1;
            st.touches_left   = st.touches_left   || cx==1;
            st.touches_right  = st.touches_right  || cx==nelx;
            st.touches_bottom = st.touches_bottom || cy==1;
            st.touches_top    = st.touches_top    || cy==nely;
            for a = [cy-1,cx; cy+1,cx; cy,cx-1; cy,cx+1]'
                ny=a(1); nx=a(2);
                if ny>=1&&ny<=nely&&nx>=1&&nx<=nelx&&solid(ny,nx)&&component(ny,nx)==0
                    tail=tail+1; q(tail,:)=[ny,nx]; component(ny,nx)=compId;
                end
            end
        end
        stats(end+1,1) = st; %#ok<AGROW>
    end
end
component = component(:);
end

function id = s1LargestSupport(stats)
id = 0; best = -Inf;
for i = 1:numel(stats)
    if stats(i).touches_left && stats(i).touches_right && stats(i).size > best
        best = stats(i).size; id = stats(i).id;
    end
end
end

function [domId, fracK, fracS] = s1DomComp(component, kinElem, strElem)
ids = unique(component(component > 0));
if isempty(ids); domId=NaN; fracK=NaN; fracS=NaN; return; end
scores = zeros(numel(ids),1); fKs=scores; fSs=scores;
for i = 1:numel(ids)
    mask = component==ids(i);
    fKs(i) = s1Frac(kinElem(mask), kinElem);
    fSs(i) = s1Frac(strElem(mask),  strElem);
    scores(i) = max(fKs(i),fSs(i));
end
[~,idx] = max(scores);
domId=ids(idx); fracK=fKs(idx); fracS=fSs(idx);
end

function f = s1Frac(part, all)
den = sum(max(real(all(:)),0));
if den <= 0; f = NaN; else; f = sum(max(real(part(:)),0))/den; end
end

function [idx, eff, top1, top5] = s1Loc(e)
e = max(real(e(:)),0); n = numel(e); total = sum(e);
if total <= 0; idx=NaN; eff=NaN; top1=NaN; top5=NaN; return; end
p = e/total; idx = sum(p.^2); eff = 1/(n*idx);
s = sort(p,'descend');
top1 = sum(s(1:max(1,ceil(0.01*n))));
top5 = sum(s(1:max(1,ceil(0.05*n))));
end

function s1WriteEnergyCSV(path, x, elemCenters, component, compStats, kinElem, strElem, row)
nEl = numel(x);
touchL = false(nEl,1); touchR = false(nEl,1); touchB = false(nEl,1);
for e = 1:nEl
    cid = component(e);
    if cid > 0 && cid <= numel(compStats)
        touchL(e) = compStats(cid).touches_left;
        touchR(e) = compStats(cid).touches_right;
        touchB(e) = compStats(cid).touches_left && compStats(cid).touches_right;
    end
end
kT = sum(kinElem); sT = sum(strElem);
T = table((1:nEl)', elemCenters(:,1), elemCenters(:,2), x(:), component(:), ...
    touchL, touchR, touchB, kinElem(:), strElem(:), ...
    kinElem(:)/max(kT,eps), strElem(:)/max(sT,eps), x(:)<0.05, ...
    repmat(row.mode,nEl,1), ...
    'VariableNames',{'element','x_center','y_center','density','solid_component_id', ...
    'touches_left','touches_right','touches_both', ...
    'kinetic_energy','strain_energy','kinetic_energy_fraction','strain_energy_fraction', ...
    'is_low_density','mode'});
writetable(T, path);
end

function s1SaveModeShape(path, phi, x, nelx, nely, L, H, omega, modeIdx, row)
xG = repmat((0:nelx)*(L/nelx), nely+1, 1);
yG = repmat((0:nely)'*(H/nely), 1, nelx+1);
ux = reshape(phi(1:2:end), nely+1, nelx+1);
uy = reshape(phi(2:2:end), nely+1, nelx+1);
mag = hypot(ux, uy);
sc = 0.075*max(L,H)/max(max(mag(:)),eps);
xD = xG + sc*ux; yD = yG + sc*uy;
fig = figure('Color','white','Visible','off');
ax  = axes('Parent',fig); hold(ax,'on');
topo = reshape(x(:),nely,nelx);
imagesc(ax,[0,L],[0,H],flipud(topo));
set(ax,'YDir','normal'); colormap(ax,gray); alpha(0.35);
stride = max(1,ceil((max(nelx,nely)+1)/120));
rows = unique([1:stride:(nely+1), nely+1]);
cols = unique([1:stride:(nelx+1), nelx+1]);
for r = rows; plot(ax,xD(r,:),yD(r,:),'-','Color',[0.05,0.25,0.75],'LineWidth',0.65); end
for c = cols; plot(ax,xD(:,c),yD(:,c),'-','Color',[0.05,0.25,0.75],'LineWidth',0.65); end
plot(ax,[0,L,L,0,0],[0,0,H,H,0],'-','Color',[0.15,0.15,0.15],'LineWidth',1);
axis(ax,'equal'); xlim(ax,[-0.05*L,1.05*L]); ylim(ax,[-0.15*H,1.15*H]);
set(ax,'XTick',[],'YTick',[]); box(ax,'on');
title(ax,sprintf('Mode %d | %.4g rad/s | %s',modeIdx,omega,row.classification), ...
    'Interpreter','none','FontSize',8);
try
    exportgraphics(fig,path,'Resolution',180,'BackgroundColor','white');
catch
    print(fig,path,'-dpng','-r180');
end
close(fig);
end

% =========================================================================
%  Gate / classification helpers
% =========================================================================

function pass = localA5Pass(info)
pass = false;
if ~isfield(info,'cr2_final_tracking'); return; end
idx = info.cr2_final_tracking.tracked_mode_index;
if isfinite(idx) && idx == 1; pass = true; end
end

function cls = localClassify(result, cfg, criteria)
if ~result.solver_success;           cls = 'implementation failure'; return; end
if ~isfinite(result.tracked_mac) || result.tracked_mac < criteria.mac_threshold || ...
        ~isfinite(result.tracked_index) || result.tracked_index < 1 || ~result.a5_pass
    cls = 'mode invalid'; return;
end
if result.iterations >= double(cfg.optimization.max_iters)
    cls = 'capped'; return;
end
if result.design_change <= criteria.design_change_tolerance && ...
        result.feasibility <= criteria.feasibility_tolerance
    cls = 'accepted';
else
    cls = 'capped';
end
end

function onset = localOnsetMesh(cases, criteria)
onset = 'none observed';
for i = 1:numel(cases)
    r = cases(i);
    macFail  = isfinite(r.tracked_mac) && r.tracked_mac < criteria.mac_threshold;
    locMaj   = isfinite(r.s1_localized_count) && ...
        r.s1_localized_count > (r.s1_physical_global_count + r.s1_ambiguous_count + ...
        r.s1_disconnected_count);
    if macFail || locMaj
        onset = r.mesh_label;
        return;
    end
end
end

% =========================================================================
%  Reuse helper for existing Exp3 400x50 result
% =========================================================================

function result = scalingCaseFromExp3(S, mc, physFilterRadius, repoRoot)
result = scalingCaseInit();
r = S.result;
result.mesh_label           = mc.label;
result.s                    = mc.s;
result.source_config        = mc.config;
result.nelx                 = double(S.cfg.domain.mesh.nelx);
result.nely                 = double(S.cfg.domain.mesh.nely);
result.n_elements           = result.nelx * result.nely;
result.filter_physical_radius = physFilterRadius;
result.solver_success       = true;
result.accepted             = r.accepted;
result.classification       = r.classification;
result.iterations           = r.iterations;
result.timing_total_s       = r.timing_total_s;
result.timing_iter_s        = r.timing_per_iter_s;
result.peak_memory_MB       = r.peak_memory_MB;
result.omega_rad_s          = r.final.omega_rad_s(:);
result.omega_1_rad_s        = r.final.omega_rad_s(1);
result.grayness             = r.final.grayness;
result.volume               = r.final.volume;
result.feasibility          = r.final.feasibility;
result.design_change        = r.final.design_change;
result.tracked_mac          = r.final.tracked_mode_mac;
result.tracked_omega        = r.final.tracked_mode_omega;
result.tracked_index        = r.final.tracked_mode_index;
result.a5_pass              = r.a5_lowest_mode_check.pass;
result.reused_from_exp3     = true;
% Absolute path for S1 to find the MAT
exp3Mat = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
    'exp3_authoritative_mesh_convergence', 'mesh_400x50', ...
    'exp3_authoritative_400x50_result.mat');
result.result_mat_abs = exp3Mat;
result.result_mat     = localRel(exp3Mat, repoRoot);
end

% =========================================================================
%  Misc helpers
% =========================================================================

function result = scalingCaseInit()
result = struct( ...
    'mesh_label','', 's',NaN, 'source_config','', ...
    'nelx',NaN, 'nely',NaN, 'n_elements',NaN, ...
    'filter_physical_radius',NaN, ...
    'solver_success',false, 'accepted',false, ...
    'classification','not run', 'iterations',NaN, ...
    'timing_total_s',NaN, 'timing_iter_s',NaN, 'peak_memory_MB',NaN, ...
    'omega_rad_s',NaN(3,1), 'omega_1_rad_s',NaN, ...
    'grayness',NaN, 'volume',NaN, 'feasibility',NaN, ...
    'design_change',NaN, 'tracked_mac',NaN, ...
    'tracked_omega',NaN, 'tracked_index',NaN, ...
    'a5_pass',false, 'exception','', ...
    'reused_from_exp3',false, 'result_mat','', 'result_mat_abs','', ...
    's1_overall_classification','not run', ...
    's1_physical_global_count',NaN, 's1_localized_count',NaN, ...
    's1_ambiguous_count',NaN, 's1_disconnected_count',NaN, ...
    's1_mode1_classification','', ...
    's1_mode1_low_density_strain_fraction',NaN, ...
    's1_artifacts',struct(), 's1_exception','');
end

function n = localNelx(cfgPath)
cfg = jsondecode(fileread(cfgPath)); n = double(cfg.domain.mesh.nelx);
end
function n = localNely(cfgPath)
cfg = jsondecode(fileread(cfgPath)); n = double(cfg.domain.mesh.nely);
end

function v = localFinalHist(s, path)
v = NaN; cur = s;
for k = 1:numel(path)
    if ~isstruct(cur) || ~isfield(cur,path{k}); return; end
    cur = cur.(path{k});
end
if isnumeric(cur) && ~isempty(cur); v = double(cur(end)); end
end

function safe = localJsonSafe(data)
safe = data;
if isfield(safe,'exception') && strlength(string(safe.exception)) > 4000
    safe.exception = extractBefore(string(safe.exception), 4001);
end
if isfield(safe,'cases')
    for i = 1:numel(safe.cases)
        if isfield(safe.cases(i),'exception') && ...
                strlength(string(safe.cases(i).exception)) > 4000
            safe.cases(i).exception = extractBefore(string(safe.cases(i).exception), 4001);
        end
    end
end
end

function localWriteJson(path, data)
try; txt = jsonencode(data, PrettyPrint=true); catch; txt = jsonencode(data); end
localWriteText(path, txt);
end

function localWriteText(path, txt)
fid = fopen(path, 'w');
if fid < 0; error('ScalingStudy:WriteFailed','Cannot write: %s', path); end
cu = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', char(txt));
end

function rel = localRel(path, repoRoot)
path = char(string(path)); repoRoot = char(string(repoRoot));
if startsWith(path, [repoRoot, filesep])
    rel = strrep(path(numel(repoRoot)+2:end), filesep, '/');
else
    rel = strrep(path, filesep, '/');
end
end
