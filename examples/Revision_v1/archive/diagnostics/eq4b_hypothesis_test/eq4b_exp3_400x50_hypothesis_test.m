function study = eq4b_exp3_400x50_hypothesis_test(outDir)
%EQ4B_EXP3_400X50_HYPOTHESIS_TEST Controlled Eq. 4b hypothesis test.
%
% Runs exactly one optimization benchmark: Exp3 alpha=1.00, mesh 400x50,
% with only mass interpolation changed to Du & Olhoff Eq. 4b (du2007_c1).

if nargin < 1 || isempty(outDir)
    scriptDir = fileparts(mfilename('fullpath'));
    outDir = fullfile(scriptDir, 'output', 'eq4b_exp3_400x50');
end
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'analysis', 'ourApproach', 'Matlab'));
addpath(scriptDir);

paths = localPaths(repoRoot, outDir);
diary(paths.log);
cleanupDiary = onCleanup(@() diary('off')); %#ok<NASGU>

fprintf('Eq. 4b controlled hypothesis test started: %s\n', char(datetime('now', 'TimeZone', 'UTC')));
fprintf('Scope: one Exp3 400x50 alpha=1.00 benchmark only; no sweeps, CR2, A4, P1, or manuscript edits.\n');

criteria = struct( ...
    'mac_threshold', 0.8, ...
    'feasibility_tolerance', 1e-8, ...
    'design_change_tolerance', 0.001);

validation = localValidateMassInterpolation();
localWriteJson(paths.validation_json, validation);
if ~validation.disabled_behavior_pass
    error('Eq4b:ValidationFailed', 'Disabled/default mass interpolation did not reproduce previous behavior.');
end

baselineCfg = jsondecode(fileread(paths.baseline_config));
cfg = baselineCfg;
cfg.meta.name = 'Eq4b hypothesis Exp3 400x50 alpha=1.00';
cfg.meta.notes = ['Controlled hypothesis test. Same failed Exp3 400x50 setup, ', ...
    'with exactly one implementation-level option changed: mass_interpolation.mode=du2007_c1.'];
cfg.optimization.mass_interpolation = struct( ...
    'mode', 'du2007_c1', ...
    'pmass', 1, ...
    'source', 'Du and Olhoff 2007 Eq. 4b, C1 low-density mass interpolation', ...
    'threshold', 0.1, ...
    'c1', 6e5, ...
    'c2', -5e6);
localWriteJson(paths.config_json, cfg);
localWritePortReview(paths.port_review_md, validation, repoRoot);

s1_json = fullfile(paths.s1_dir, 'eq4b_mode_summary.json');
resumeFromSaved = exist(paths.result_mat, 'file') == 2 && exist(s1_json, 'file') == 2;
if resumeFromSaved
    fprintf('Resuming from saved Eq. 4b result and S1 artifacts; optimization is not rerun.\n');
    saved = load(paths.result_mat, 'result', 'cfg', 'xFinal', 'omega', 'info');
    result = saved.result;
    cfg = saved.cfg;
    xFinal = saved.xFinal; %#ok<NASGU>
    omega = saved.omega; %#ok<NASGU>
    info = saved.info; %#ok<NASGU>
    s1 = jsondecode(fileread(s1_json));
else
    runCfg = localBuildRunCfg(cfg);
    rminPhys = double(cfg.optimization.filter.radius);
    ft = 0;
    fprintf('Mass interpolation: du2007_c1 Eq. 4b.\n');
    fprintf('Calling topopt_freq once for Eq. 4b Exp3 400x50 case...\n');
    runTic = tic;
    [xFinal, fHz, tIter, nIter, info] = topopt_freq( ...
        double(cfg.domain.mesh.nelx), double(cfg.domain.mesh.nely), ...
        double(cfg.optimization.volume_fraction), double(cfg.optimization.penalization), ...
        rminPhys, ft, double(cfg.domain.size.length), double(cfg.domain.size.height), runCfg);
    timingTotal = toc(runTic);
    omega = 2*pi*fHz(:);

    result = localBuildResult(cfg, criteria, xFinal, omega, tIter, nIter, timingTotal, info, paths, repoRoot);
    localWriteHistories(paths, info);
    localWriteTopology(paths, xFinal, cfg);
    result.acceptance.artifacts_complete = true;
    result.accepted = localAccepted(result);
    result.classification = localResultClassification(result);
    result.artifacts = localArtifacts(paths, repoRoot);
    localWriteJson(paths.result_json, localJsonSafe(result));
    save(paths.result_mat, 'result', 'cfg', 'xFinal', 'omega', 'info', '-v7.3');

    fprintf('Running existing S1 postprocessing on Eq. 4b final topology...\n');
    s1 = s1_exp3_400x50_mode_diagnostic(paths.s1_dir, paths.result_mat, 'eq4b');
end

refs = localLoadReferences(paths);
comparison = localCompareAll(refs, result, s1, paths);
conclusion = localScientificConclusion(refs, result, s1);

study = struct();
study.study = 'Eq4b controlled hypothesis test for Exp3 400x50 alpha=1.00';
study.scope = 'One benchmark; all settings identical to failed Exp3 400x50 except mass_interpolation.mode=du2007_c1';
study.created_utc = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'));
study.validation = validation;
study.criteria = criteria;
study.result = result;
study.s1 = s1;
study.references = refs.summary;
study.comparison = comparison;
study.conclusion = conclusion;
study.artifacts = localArtifacts(paths, repoRoot);

localWriteJson(paths.study_json, localJsonSafe(study));
localWriteS1Summary(paths.s1_summary_md, study);
localWriteManifest(paths.manifest_json, study, paths, repoRoot);

fprintf('Eq. 4b result classification: %s\n', result.classification);
fprintf('Eq. 4b S1 classification: %s\n', s1.overall.classification);
fprintf('Summary: %s\n', paths.s1_summary_md);
diary('off');
end

function paths = localPaths(repoRoot, outDir)
prefix = fullfile(outDir, 'eq4b');
paths = struct();
paths.baseline_result_json = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
    'exp3_authoritative_mesh_convergence', 'mesh_400x50', 'exp3_authoritative_400x50_result.json');
paths.baseline_config = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
    'exp3_authoritative_mesh_convergence', 'mesh_400x50', 'exp3_authoritative_400x50_config.json');
paths.baseline_convergence_csv = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
    'exp3_authoritative_mesh_convergence', 'mesh_400x50', 'exp3_authoritative_400x50_convergence_history.csv');
paths.baseline_topology_csv = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
    'exp3_authoritative_mesh_convergence', 'mesh_400x50', 'exp3_authoritative_400x50_topology.csv');
paths.baseline_s1_json = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
    's1_exp3_400x50_mode_diagnostic', 's1_exp3_400x50_mode_summary.json');
paths.pmass_result_json = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
    's1_mitigation_400x50', 's1_mitigation_400x50_result.json');
paths.pmass_study_json = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
    's1_mitigation_400x50', 's1_mitigation_400x50_study.json');
paths.pmass_topology_csv = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
    's1_mitigation_400x50', 's1_mitigation_400x50_topology.csv');
paths.pmass_s1_json = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
    's1_mitigation_400x50', 's1_postprocessing', 's1_mitigation_400x50_mode_summary.json');

paths.log = [prefix, '_run.log'];
paths.port_review_md = fullfile(outDir, 'eq4b_port_review.md');
paths.validation_json = fullfile(outDir, 'eq4b_validation_result.json');
paths.config_json = fullfile(outDir, 'eq4b_exp3_400x50_config.json');
paths.result_json = fullfile(outDir, 'eq4b_exp3_400x50_result.json');
paths.result_mat = fullfile(outDir, 'eq4b_exp3_400x50_result.mat');
paths.study_json = fullfile(outDir, 'eq4b_study.json');
paths.s1_summary_md = fullfile(outDir, 'eq4b_s1_summary.md');
paths.manifest_json = fullfile(outDir, 'eq4b_manifest.json');
paths.convergence_csv = fullfile(outDir, 'eq4b_exp3_400x50_convergence_history.csv');
paths.frequency_csv = fullfile(outDir, 'eq4b_exp3_400x50_frequency_history.csv');
paths.mode_tracking_csv = fullfile(outDir, 'eq4b_exp3_400x50_mode_tracking.csv');
paths.topology_csv = fullfile(outDir, 'eq4b_exp3_400x50_topology.csv');
paths.topology_json = fullfile(outDir, 'eq4b_exp3_400x50_topology.json');
paths.topology_png = fullfile(outDir, 'eq4b_exp3_400x50_topology.png');
paths.a5_json = fullfile(outDir, 'eq4b_exp3_400x50_a5_lowest_mode_check.json');
paths.s1_dir = fullfile(outDir, 's1_postprocessing');
if exist(paths.s1_dir, 'dir') ~= 7
    mkdir(paths.s1_dir);
end
end

function validation = localValidateMassInterpolation()
x = linspace(0, 1, 1001)';
rho0 = 1.0;
rho_min = 1e-6;
[rhoPower, dPower] = our_mass_interpolation(x, rho0, rho_min, 'power', 1);
rhoOld = rho_min + x * (rho0 - rho_min);
dOld = ones(size(x)) * (rho0 - rho_min);
[rhoLinear, dLinear] = our_mass_interpolation(x, rho0, rho_min, 'linear', 1);

[~, dEq4b, mEq4b, dmEq4b] = our_mass_interpolation(x, rho0, rho_min, 'du2007_c1', 1);
mAuth = x;
dmAuth = ones(size(x));
lo = x <= 0.1;
mAuth(lo) = 6e5*x(lo).^6 - 5e6*x(lo).^7;
dmAuth(lo) = 6*6e5*x(lo).^5 - 7*5e6*x(lo).^6;

validation = struct();
validation.disabled_behavior_pass = max(abs(rhoPower-rhoOld)) == 0 && ...
    max(abs(dPower-dOld)) == 0 && max(abs(rhoLinear-rhoOld)) == 0 && ...
    max(abs(dLinear-dOld)) == 0;
validation.disabled_behavior = struct( ...
    'max_abs_rho_power_pmass1_minus_old', max(abs(rhoPower-rhoOld)), ...
    'max_abs_drho_power_pmass1_minus_old', max(abs(dPower-dOld)), ...
    'max_abs_rho_linear_minus_old', max(abs(rhoLinear-rhoOld)), ...
    'max_abs_drho_linear_minus_old', max(abs(dLinear-dOld)));
validation.eq4b_equivalence_pass = max(abs(mEq4b-mAuth)) == 0 && max(abs(dmEq4b-dmAuth)) == 0;
validation.eq4b_equivalence = struct( ...
    'max_abs_m_minus_authoritative_formula', max(abs(mEq4b-mAuth)), ...
    'max_abs_dm_minus_authoritative_formula', max(abs(dmEq4b-dmAuth)), ...
    'rho_at_threshold_minus_linear', localAtThresholdValue(), ...
    'drho_at_threshold_minus_linear', localAtThresholdDerivative(), ...
    'max_abs_drho_scaled_minus_formula', max(abs(dEq4b - dmAuth*(rho0-rho_min))));
end

function v = localAtThresholdValue()
x = 0.1;
v = 6e5*x^6 - 5e6*x^7 - x;
end

function v = localAtThresholdDerivative()
x = 0.1;
v = 6*6e5*x^5 - 7*5e6*x^6 - 1;
end

function runCfg = localBuildRunCfg(cfg)
[extraFixedDofs, ~] = supportsToFixedDofs(cfg.bc.supports, ...
    double(cfg.domain.mesh.nelx), double(cfg.domain.mesh.nely), ...
    double(cfg.domain.size.length), double(cfg.domain.size.height));
runCfg = struct();
runCfg.E0 = double(cfg.material.E);
runCfg.Emin = double(cfg.material.E) * double(cfg.void_material.E_min_ratio);
runCfg.nu = double(cfg.material.nu);
runCfg.rho0 = double(cfg.material.rho);
runCfg.rho_min = double(cfg.void_material.rho_min);
runCfg.move = double(cfg.optimization.move_limit);
runCfg.conv_tol = double(cfg.optimization.convergence_tol);
runCfg.max_iters = double(cfg.optimization.max_iters);
runCfg.supportType = 'CC';
runCfg.extraFixedDofs = extraFixedDofs;
runCfg.pasS = [];
runCfg.pasV = [];
runCfg.load_sensitivity = char(string(cfg.optimization.load_sensitivity));
runCfg.gate_a0_diagnostics = true;
runCfg.harmonic_normalize = false;
runCfg.semi_harmonic_baseline = 'solid';
runCfg.load_cases = cfg.domain.load_cases;
runCfg.optimizer = upper(char(string(cfg.optimization.optimizer)));
runCfg.approach_name = 'ourApproach';
runCfg.visualize_live = false;
runCfg.visualization_quality = 'regular';
runCfg.save_frq_iterations = false;
runCfg.mass_interpolation = cfg.optimization.mass_interpolation;
end

function result = localBuildResult(cfg, criteria, xFinal, omega, tIter, nIter, timingTotal, info, paths, repoRoot)
volfrac = double(cfg.optimization.volume_fraction);
result = struct();
result.study = 'Eq4b Exp3 400x50 alpha=1.00';
result.scope = 'same Exp3 400x50 setup except mass_interpolation.mode=du2007_c1';
result.implementation_language = 'MATLAB';
result.mesh_label = '400x50';
result.mesh = cfg.domain.mesh;
result.alpha = 1.0;
result.source_baseline_config = localRel(paths.baseline_config, repoRoot);
result.case_config = localRel(paths.config_json, repoRoot);
result.authoritative_load = 'F(x) = omega0^2 * M(x) * Phi0';
result.mass_interpolation = cfg.optimization.mass_interpolation;
result.filter_physical_radius = double(cfg.optimization.filter.radius);
result.criteria = criteria;
result.solver_success = true;
result.iterations = double(nIter);
result.iteration_cap = double(cfg.optimization.max_iters);
result.final = struct();
result.final.omega_rad_s = omega(:);
result.final.frequency_hz = omega(:) / (2*pi);
result.final.grayness = mean(4 * xFinal(:) .* (1 - xFinal(:)));
result.final.volume = mean(xFinal(:));
result.final.feasibility = max(0, result.final.volume - volfrac);
result.final.design_change = localLast(info.cr2_history.design_change);
result.final.tracked_mode_index = double(info.cr2_final_tracking.tracked_mode_index);
result.final.tracked_mode_mac = double(info.cr2_final_tracking.tracked_mode_mac);
result.final.tracked_mode_omega = double(info.cr2_final_tracking.tracked_mode_omega);
result.a5_lowest_mode_check = localA5(info);
result.acceptance = struct( ...
    'not_capped', result.iterations < result.iteration_cap, ...
    'design_change_ok', result.final.design_change <= criteria.design_change_tolerance, ...
    'feasibility_ok', result.final.feasibility <= criteria.feasibility_tolerance, ...
    'tracked_mac_ok', result.final.tracked_mode_mac >= criteria.mac_threshold, ...
    'a5_lowest_mode_ok', result.a5_lowest_mode_check.pass, ...
    'artifacts_complete', false);
result.accepted = localAccepted(result);
result.classification = localResultClassification(result);
result.timing_total_s = timingTotal;
result.timing_per_iter_s = tIter;
result.artifacts = localArtifacts(paths, repoRoot);
end

function tf = localAccepted(result)
a = result.acceptance;
tf = a.not_capped && a.design_change_ok && a.feasibility_ok && ...
    a.tracked_mac_ok && a.a5_lowest_mode_ok && a.artifacts_complete;
end

function label = localResultClassification(result)
if result.accepted
    label = 'accepted';
elseif ~result.acceptance.not_capped
    label = 'capped';
elseif ~result.acceptance.tracked_mac_ok || ~result.acceptance.a5_lowest_mode_ok
    label = 'mode invalid';
else
    label = 'not accepted';
end
end

function check = localA5(info)
freqs = info.cr2_final_tracking.frequencies(:);
trackedIndex = double(info.cr2_final_tracking.tracked_mode_index);
trackedOmega = NaN;
if trackedIndex >= 1 && trackedIndex <= numel(freqs)
    trackedOmega = freqs(trackedIndex);
end
check = struct( ...
    'pass', isfinite(trackedIndex) && trackedIndex == 1 && isfinite(trackedOmega), ...
    'tracked_mode_index', trackedIndex, ...
    'tracked_mode_omega_rad_s', trackedOmega, ...
    'lowest_mode_omega_rad_s', freqs(1), ...
    'modes_below_tracked_count', max(0, trackedIndex - 1));
end

function localWriteHistories(paths, info)
h = info.cr2_history;
n = size(h.objective, 1);
iter = (1:n)';
writetable(table(iter, h.objective(:), h.design_change(:), h.volume(:), ...
    h.feasibility(:), h.grayness(:), h.tracked_mode_index(:), ...
    h.tracked_mode_mac(:), h.tracked_mode_omega(:), ...
    'VariableNames', {'iteration','objective','design_change','volume', ...
    'feasibility','grayness','tracked_mode_index','tracked_mode_mac', ...
    'tracked_mode_omega_rad_s'}), paths.convergence_csv);
freq = h.frequency;
freqNames = arrayfun(@(k) sprintf('omega_%d_rad_s', k), 1:size(freq,2), 'UniformOutput', false);
writetable(array2table([iter, freq], 'VariableNames', [{'iteration'}, freqNames]), paths.frequency_csv);
writetable(table(iter, h.tracked_mode_index(:), h.tracked_mode_mac(:), h.tracked_mode_omega(:), ...
    'VariableNames', {'iteration','tracked_mode_index','tracked_mode_mac','tracked_mode_omega_rad_s'}), ...
    paths.mode_tracking_csv);
localWriteJson(paths.a5_json, localA5(info));
end

function localWriteTopology(paths, xFinal, cfg)
nelx = double(cfg.domain.mesh.nelx);
nely = double(cfg.domain.mesh.nely);
topology = reshape(xFinal(:), nely, nelx);
writematrix(topology, paths.topology_csv);
localWriteJson(paths.topology_json, struct('nelx', nelx, 'nely', nely, ...
    'density_row_major_nely_by_nelx', topology));
fig = figure('Color', 'white', 'Visible', 'off');
ax = axes('Parent', fig);
imagesc(ax, 1 - topology);
axis(ax, 'equal', 'off');
colormap(ax, gray);
title(ax, 'Eq4b Exp3 400x50 alpha=1.00 topology', 'Interpreter', 'none');
exportgraphics(fig, paths.topology_png, 'Resolution', 180, 'BackgroundColor', 'white');
close(fig);
end

function refs = localLoadReferences(paths)
refs = struct();
refs.original.result = jsondecode(fileread(paths.baseline_result_json));
refs.original.s1 = jsondecode(fileread(paths.baseline_s1_json));
refs.original.topology = readmatrix(paths.baseline_topology_csv);
refs.original.convergence = readtable(paths.baseline_convergence_csv);
refs.pmass6.result = jsondecode(fileread(paths.pmass_result_json));
refs.pmass6.study = jsondecode(fileread(paths.pmass_study_json));
refs.pmass6.s1 = jsondecode(fileread(paths.pmass_s1_json));
refs.pmass6.topology = readmatrix(paths.pmass_topology_csv);
refs.summary = struct( ...
    'original_result_json', paths.baseline_result_json, ...
    'original_s1_json', paths.baseline_s1_json, ...
    'pmass6_result_json', paths.pmass_result_json, ...
    'pmass6_s1_json', paths.pmass_s1_json);
end

function comparison = localCompareAll(refs, result, s1, paths)
eqTopo = readmatrix(paths.topology_csv);
eqConv = readtable(paths.convergence_csv);
comparison = struct();
comparison.cases = struct();
comparison.cases.original = localCaseMetrics(refs.original.result, refs.original.s1);
comparison.cases.pmass6 = localCaseMetrics(refs.pmass6.result, refs.pmass6.s1);
comparison.cases.eq4b = localCaseMetrics(result, s1);
comparison.delta_eq4b_minus_original = localMetricDelta(comparison.cases.eq4b, comparison.cases.original);
comparison.delta_eq4b_minus_pmass6 = localMetricDelta(comparison.cases.eq4b, comparison.cases.pmass6);
comparison.topology = struct( ...
    'eq4b_vs_original', localTopologyDelta(eqTopo, refs.original.topology), ...
    'eq4b_vs_pmass6', localTopologyDelta(eqTopo, refs.pmass6.topology));
comparison.convergence = struct( ...
    'eq4b_last10_design_change_mean', mean(eqConv.design_change(max(1,end-9):end)), ...
    'eq4b_last100_design_change_max', max(eqConv.design_change(max(1,end-99):end)));
end

function m = localCaseMetrics(result, s1)
m = struct();
m.omega_1_rad_s = result.final.omega_rad_s(1);
m.tracked_mac = result.final.tracked_mode_mac;
m.tracked_omega_rad_s = result.final.tracked_mode_omega;
m.classification = result.classification;
m.accepted = result.accepted;
m.grayness = result.final.grayness;
m.iterations = result.iterations;
m.design_change = result.final.design_change;
m.localized_low_density_modes = s1.overall.localized_low_density_count;
m.physical_global_modes = s1.overall.physical_global_count;
m.ambiguous_modes = s1.overall.ambiguous_count;
m.s1_classification = s1.overall.classification;
m.mode_classifications = {s1.modes.classification};
end

function d = localMetricDelta(a, b)
d = struct( ...
    'omega_1_rad_s', a.omega_1_rad_s - b.omega_1_rad_s, ...
    'tracked_mac', a.tracked_mac - b.tracked_mac, ...
    'tracked_omega_rad_s', a.tracked_omega_rad_s - b.tracked_omega_rad_s, ...
    'grayness', a.grayness - b.grayness, ...
    'iterations', a.iterations - b.iterations, ...
    'localized_low_density_modes', a.localized_low_density_modes - b.localized_low_density_modes);
end

function d = localTopologyDelta(a, b)
delta = a - b;
d = struct( ...
    'mean_abs_difference', mean(abs(delta(:))), ...
    'rms_difference', sqrt(mean(delta(:).^2)), ...
    'max_abs_difference', max(abs(delta(:))), ...
    'correlation', localCorr(a(:), b(:)));
end

function conclusion = localScientificConclusion(refs, result, s1)
orig = localCaseMetrics(refs.original.result, refs.original.s1);
pm6 = localCaseMetrics(refs.pmass6.result, refs.pmass6.s1);
eq4b = localCaseMetrics(result, s1);
improvedGlobal = result.accepted && eq4b.tracked_mac >= 0.8 && ...
    eq4b.omega_1_rad_s > orig.omega_1_rad_s && eq4b.tracked_mac > orig.tracked_mac;
reducedLocalized = eq4b.localized_low_density_modes < orig.localized_low_density_modes && ...
    eq4b.localized_low_density_modes <= pm6.localized_low_density_modes;
rescuesFineCase = result.accepted && result.acceptance.a5_lowest_mode_ok;
if improvedGlobal && reducedLocalized && rescuesFineCase
    suff = 'yes';
    reason = 'Eq. 4b improves tracked response, reduces localized modes, and passes the fine-mesh gates in the single benchmark.';
else
    suff = 'no';
    reason = 'The single benchmark does not simultaneously improve the tracked response, reduce localized low-density modes, and establish mesh convergence.';
end
conclusion = struct();
conclusion.q1_does_eq4b_improve_accepted_tracked_global_response = localYesNo(improvedGlobal);
conclusion.q2_does_eq4b_reduce_localized_low_density_modes = localYesNo(reducedLocalized);
conclusion.q3_does_eq4b_rescue_exp3_mesh_validation = localYesNo(rescuesFineCase);
conclusion.q4_is_eq4b_sufficient_missing_mechanism = suff;
conclusion.reason = reason;
end

function s = localYesNo(tf)
if tf
    s = 'yes';
else
    s = 'no';
end
end

function localWritePortReview(path, validation, repoRoot)
fid = fopen(path, 'w');
if fid < 0, error('Eq4b:CannotWrite', 'Cannot write %s', path); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '# Eq. 4b Port Review\n\n');
fprintf(fid, '## Modified Files\n\n');
fprintf(fid, '- `analysis/ourApproach/Matlab/our_mass_interpolation.m`\n');
fprintf(fid, '- `analysis/ourApproach/Matlab/topopt_freq.m`\n');
fprintf(fid, '- `tools/Matlab/run_topopt_from_json.m`\n');
fprintf(fid, '- `examples/Revision_v1/s1_exp3_400x50_mode_diagnostic.m`\n');
fprintf(fid, '- `examples/Revision_v1/eq4b_exp3_400x50_hypothesis_test.m`\n\n');
fprintf(fid, '## Authoritative Source\n\n');
fprintf(fid, '`analysis/OlhoffApproachExact/Matlab/mass_interp.m` implements `du2007_c1` as `m=rho` above `rho=0.1` and `m=6e5*rho^6-5e6*rho^7` below or at `rho=0.1`.\n\n');
fprintf(fid, '## Ported Formula\n\n');
fprintf(fid, '`ourApproach` now applies the same normalized coefficient `m(x)` and derivative `dm/dx`, then preserves the existing density scaling: `rho_e = rho_min + m(x)*(rho0-rho_min)`.\n\n');
fprintf(fid, '## Equivalence\n\n');
fprintf(fid, '- Disabled/default behavior pass: `%s`.\n', localYesNo(validation.disabled_behavior_pass));
fprintf(fid, '- Eq. 4b formula equivalence pass: `%s`.\n', localYesNo(validation.eq4b_equivalence_pass));
fprintf(fid, '- Max disabled rho difference: `%.17g`.\n', validation.disabled_behavior.max_abs_rho_power_pmass1_minus_old);
fprintf(fid, '- Max Eq. 4b coefficient difference: `%.17g`.\n\n', validation.eq4b_equivalence.max_abs_m_minus_authoritative_formula);
fprintf(fid, '## Unavoidable Difference\n\n');
fprintf(fid, 'The authoritative helper returns normalized `m(rho_e)`. `ourApproach` stores physical density scaling separately, so the port wraps the same normalized `m(x)` with the existing `rho_min/rho0` scaling. This preserves backward compatibility and the existing mass floor.\n\n');
fprintf(fid, 'Repo root: `%s`.\n', repoRoot);
end

function localWriteS1Summary(path, study)
fid = fopen(path, 'w');
if fid < 0, error('Eq4b:CannotWrite', 'Cannot write %s', path); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
c = study.comparison.cases;
fprintf(fid, '# Eq. 4b S1 Summary\n\n');
fprintf(fid, 'Scope: one controlled Exp3 400x50 alpha=1.00 benchmark; no parameter tuning and no manuscript edits.\n\n');
fprintf(fid, '## Acceptance Gates\n\n');
a = study.result.acceptance;
fprintf(fid, '| gate | pass | value |\n|---|---|---:|\n');
fprintf(fid, '| not capped | %s | %d/%d |\n', localPass(a.not_capped), study.result.iterations, study.result.iteration_cap);
fprintf(fid, '| design_change <= tol | %s | %.12g <= %.12g |\n', localPass(a.design_change_ok), study.result.final.design_change, study.criteria.design_change_tolerance);
fprintf(fid, '| feasibility <= tol | %s | %.12g <= %.12g |\n', localPass(a.feasibility_ok), study.result.final.feasibility, study.criteria.feasibility_tolerance);
fprintf(fid, '| tracked MAC >= 0.8 | %s | %.12g |\n', localPass(a.tracked_mac_ok), study.result.final.tracked_mode_mac);
fprintf(fid, '| A5 lowest-mode check | %s | mode %d |\n', localPass(a.a5_lowest_mode_ok), study.result.a5_lowest_mode_check.tracked_mode_index);
fprintf(fid, '| artifacts complete | %s | manifest written |\n\n', localPass(a.artifacts_complete));

fprintf(fid, '## Three-Case Comparison\n\n');
fprintf(fid, '| metric | original 400x50 | pmass=6 | Eq. 4b |\n|---|---:|---:|---:|\n');
fprintf(fid, '| omega_1 rad/s | %.12g | %.12g | %.12g |\n', c.original.omega_1_rad_s, c.pmass6.omega_1_rad_s, c.eq4b.omega_1_rad_s);
fprintf(fid, '| tracked MAC | %.12g | %.12g | %.12g |\n', c.original.tracked_mac, c.pmass6.tracked_mac, c.eq4b.tracked_mac);
fprintf(fid, '| grayness | %.12g | %.12g | %.12g |\n', c.original.grayness, c.pmass6.grayness, c.eq4b.grayness);
fprintf(fid, '| iterations | %d | %d | %d |\n', c.original.iterations, c.pmass6.iterations, c.eq4b.iterations);
fprintf(fid, '| localized low-density modes | %d | %d | %d |\n', c.original.localized_low_density_modes, c.pmass6.localized_low_density_modes, c.eq4b.localized_low_density_modes);
fprintf(fid, '| physical global modes | %d | %d | %d |\n\n', c.original.physical_global_modes, c.pmass6.physical_global_modes, c.eq4b.physical_global_modes);

fprintf(fid, '## Mode Classifications\n\n');
for k = 1:numel(study.s1.modes)
    fprintf(fid, '- mode %d: %s\n', k, study.s1.modes(k).classification);
end
fprintf(fid, '\n## Final Scientific Conclusion\n\n');
fprintf(fid, '1. Does Eq. 4b improve the accepted tracked global response? **%s**.\n', study.conclusion.q1_does_eq4b_improve_accepted_tracked_global_response);
fprintf(fid, '2. Does Eq. 4b reduce localized low-density modes? **%s**.\n', study.conclusion.q2_does_eq4b_reduce_localized_low_density_modes);
fprintf(fid, '3. Does Eq. 4b rescue Exp3 mesh validation? **%s**.\n', study.conclusion.q3_does_eq4b_rescue_exp3_mesh_validation);
fprintf(fid, '4. Is Eq. 4b sufficient evidence for the missing mechanism? **%s**. %s\n\n', study.conclusion.q4_is_eq4b_sufficient_missing_mechanism, study.conclusion.reason);
fprintf(fid, 'S1 detailed diagnosis: `s1_postprocessing/eq4b_mode_diagnosis.md`.\n');
end

function localWriteManifest(path, study, paths, repoRoot)
manifest = struct();
manifest.study = study.study;
manifest.created_utc = study.created_utc;
manifest.result_json = localRel(paths.result_json, repoRoot);
manifest.result_mat = localRel(paths.result_mat, repoRoot);
manifest.port_review_md = localRel(paths.port_review_md, repoRoot);
manifest.validation_json = localRel(paths.validation_json, repoRoot);
manifest.s1_summary_md = localRel(paths.s1_summary_md, repoRoot);
manifest.s1_diagnosis_md = localRel(fullfile(paths.s1_dir, 'eq4b_mode_diagnosis.md'), repoRoot);
manifest.study_json = localRel(paths.study_json, repoRoot);
manifest.topology_csv = localRel(paths.topology_csv, repoRoot);
manifest.topology_png = localRel(paths.topology_png, repoRoot);
manifest.energy_csv_count = numel(dir(fullfile(paths.s1_dir, 'eq4b_mode_*_energy.csv')));
manifest.mode_shape_png_count = numel(dir(fullfile(paths.s1_dir, 'eq4b_mode_*_shape.png')));
manifest.conclusion = study.conclusion;
localWriteJson(path, manifest);
end

function artifacts = localArtifacts(paths, repoRoot)
artifacts = struct( ...
    'port_review_md', localRel(paths.port_review_md, repoRoot), ...
    'validation_json', localRel(paths.validation_json, repoRoot), ...
    'config_json', localRel(paths.config_json, repoRoot), ...
    'result_json', localRel(paths.result_json, repoRoot), ...
    'result_mat', localRel(paths.result_mat, repoRoot), ...
    'study_json', localRel(paths.study_json, repoRoot), ...
    's1_summary_md', localRel(paths.s1_summary_md, repoRoot), ...
    'manifest_json', localRel(paths.manifest_json, repoRoot), ...
    'convergence_csv', localRel(paths.convergence_csv, repoRoot), ...
    'frequency_csv', localRel(paths.frequency_csv, repoRoot), ...
    'mode_tracking_csv', localRel(paths.mode_tracking_csv, repoRoot), ...
    'topology_csv', localRel(paths.topology_csv, repoRoot), ...
    'topology_png', localRel(paths.topology_png, repoRoot), ...
    's1_dir', localRel(paths.s1_dir, repoRoot));
end

function localWriteJson(path, data)
txt = jsonencode(data, 'PrettyPrint', true);
fid = fopen(path, 'w');
if fid < 0
    error('Eq4b:CannotWriteJson', 'Cannot write %s', path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', txt);
end

function safe = localJsonSafe(value)
safe = value;
if isstruct(safe)
    for j = 1:numel(safe)
        fields = fieldnames(safe(j));
        for i = 1:numel(fields)
            safe(j).(fields{i}) = localJsonSafe(safe(j).(fields{i}));
        end
    end
elseif iscell(safe)
    for i = 1:numel(safe)
        safe{i} = localJsonSafe(safe{i});
    end
elseif isnumeric(safe)
    safe = double(safe);
end
end

function rel = localRel(path, repoRoot)
rel = char(string(path));
root = char(string(repoRoot));
if startsWith(rel, [root filesep])
    rel = rel(numel(root)+2:end);
end
end

function v = localLast(x)
x = x(:);
v = x(end);
end

function c = localCorr(a, b)
C = corrcoef(a(:), b(:));
if numel(C) >= 4
    c = C(1,2);
else
    c = NaN;
end
end

function s = localPass(tf)
if tf
    s = 'pass';
else
    s = 'fail';
end
end
