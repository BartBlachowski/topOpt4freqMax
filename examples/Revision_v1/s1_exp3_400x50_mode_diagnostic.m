function summary = s1_exp3_400x50_mode_diagnostic(outDir, inputMat, reportStem)
%S1_EXP3_400X50_MODE_DIAGNOSTIC Postprocess saved Exp3 400x50 alpha=1 case.
%
% This is a postprocessing-only diagnostic. It loads the saved Exp3 result,
% reassembles K/M for the final density field, computes the first 10 modes,
% and writes mode localization/energy diagnostics. It does not rerun
% optimization and does not modify solver code or manuscript files.

if nargin < 1 || isempty(outDir)
    scriptDir = fileparts(mfilename('fullpath'));
    outDir = fullfile(scriptDir, 'output', 's1_exp3_400x50_mode_diagnostic');
end
if nargin < 2
    inputMat = '';
end
if nargin < 3 || isempty(reportStem)
    reportStem = 's1_exp3_400x50';
end
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'analysis', 'ourApproach', 'Matlab'));

paths = localPaths(repoRoot, outDir, inputMat, reportStem);
diary(paths.log);
cleanupDiary = onCleanup(@() diary('off'));

fprintf('S1 Exp3 400x50 mode diagnostic started: %s\n', ...
    char(datetime('now', 'TimeZone', 'UTC')));
fprintf('Postprocessing only: saved Exp3 400x50 artifacts are loaded; optimization is not rerun.\n');

S = load(paths.input_mat);
cfg = S.cfg;
xFinal = S.xFinal(:);
savedInfo = S.info;
savedResult = S.result;

L = double(cfg.domain.size.length);
H = double(cfg.domain.size.height);
nelx = double(cfg.domain.mesh.nelx);
nely = double(cfg.domain.mesh.nely);
thickness = double(cfg.domain.thickness);
E0 = double(cfg.material.E);
nu = double(cfg.material.nu);
rho0 = double(cfg.material.rho);
Emin = E0 * double(cfg.void_material.E_min_ratio);
rho_min = double(cfg.void_material.rho_min);
penal = double(cfg.optimization.penalization);
massInterp = localMassInterpolationFromConfig(cfg);
nModes = 10;
lowDensityThreshold = 0.05;
solidThreshold = 0.5;

if numel(xFinal) ~= nelx * nely
    error('S1:InvalidTopologySize', 'Saved xFinal size does not match mesh.');
end

[K, M, KE, ME, edofMat, elemCenters] = localAssembleKM( ...
    xFinal, nelx, nely, L, H, thickness, E0, Emin, nu, rho0, rho_min, penal, massInterp);
[fixedDofs, supportDebug] = supportsToFixedDofs(cfg.bc.supports, nelx, nely, L, H);
free = setdiff((1:size(K,1))', fixedDofs(:));

fprintf('Mesh %dx%d, ndof=%d, free=%d, fixed=%d\n', ...
    nelx, nely, size(K,1), numel(free), numel(fixedDofs));

Kf = K(free, free);
Mf = M(free, free);
eigOpts = struct('disp', 0, 'maxit', 3000, 'tol', 1e-12);
[V, D] = eigs(Kf, Mf, nModes, 'smallestabs', eigOpts);
lam = real(diag(D));
[lam, order] = sort(lam, 'ascend');
V = V(:, order);
valid = isfinite(lam) & lam > 0;
lam = lam(valid);
V = V(:, valid);
if numel(lam) < nModes
    warning('S1:ModeCountShortfall', 'Requested %d modes but only %d valid modes were returned.', ...
        nModes, numel(lam));
end
nModes = min(nModes, numel(lam));
V = V(:, 1:nModes);
lam = lam(1:nModes);
V = mass_normalize_modes(V, Mf);
V = orient_modes_deterministic(V);

Phi = zeros(size(K, 1), nModes);
Phi(free, :) = V;
omega = sqrt(lam);
freqHz = omega / (2*pi);

targetMode = savedInfo.gate_a0.reference_modes(:, 1);
macToTarget = squared_mass_weighted_mac(targetMode, Phi, M);
macToTarget = macToTarget(1, :)';

[component, compStats] = localSolidComponents(xFinal, nelx, nely, solidThreshold);
largestCompId = localLargestSupportConnectedComponent(compStats);
lowMask = xFinal < lowDensityThreshold;

modeRows = repmat(localBlankModeRow(), nModes, 1);
for k = 1:nModes
    phi = Phi(:, k);
    [kineticElem, strainElem] = localElementEnergies( ...
        phi, omega(k), xFinal, edofMat, KE, ME, E0, Emin, rho0, rho_min, penal, massInterp);
    modeRows(k) = localDiagnoseMode(k, omega(k), freqHz(k), phi, M, K, free, edofMat, ...
        macToTarget(k), kineticElem, strainElem, xFinal, elemCenters, component, ...
        compStats, largestCompId, lowMask, lowDensityThreshold);
    localWriteEnergyTable(paths.energy_csv{k}, xFinal, elemCenters, component, ...
        compStats, kineticElem, strainElem, modeRows(k));
    if k <= 6
        localSaveModeShape(paths.mode_png{k}, phi, xFinal, nelx, nely, L, H, omega(k), k, modeRows(k));
    end
end

modeTable = struct2table(modeRows);
writetable(modeTable, paths.mode_summary_csv);

summary = struct();
summary.study = sprintf('S1 diagnostic pilot for %s', reportStem);
summary.scope = 'Postprocessing-only saved-artifact mode diagnosis; no optimization rerun';
summary.created_utc = char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'));
summary.input = struct( ...
    'result_mat', localRel(paths.input_mat, repoRoot), ...
    'result_json', localRel(paths.input_json, repoRoot), ...
    'config_json', localRel(paths.input_config, repoRoot), ...
    'topology_csv', localRel(paths.input_topology_csv, repoRoot));
summary.mesh = struct('nelx', nelx, 'nely', nely, 'L', L, 'H', H, ...
    'thickness', thickness, 'hx', L/nelx, 'hy', H/nely);
summary.material = struct('E0', E0, 'Emin', Emin, 'nu', nu, ...
    'rho0', rho0, 'rho_min', rho_min, 'penal', penal, ...
    'mass_interpolation_mode', massInterp.mode, 'pmass', massInterp.pmass);
summary.thresholds = struct('low_density', lowDensityThreshold, ...
    'solid_component', solidThreshold);
summary.reference = struct( ...
    'target_mode_index', 1, ...
    'saved_reference_omega_rad_s', savedInfo.gate_a0.reference_omega(1), ...
    'saved_final_tracked_mac', savedResult.final.tracked_mode_mac, ...
    'saved_final_tracked_omega_rad_s', savedResult.final.tracked_mode_omega);
summary.supports = struct( ...
    'fixed_dof_count', numel(fixedDofs), ...
    'fixed_node_count', numel(unique(ceil(fixedDofs(:)/2))), ...
    'entries', {supportDebug.entries});
summary.components = localComponentSummary(compStats, largestCompId);
summary.modes = modeRows;
summary.overall = localOverallConclusion(modeRows);
summary.artifacts = localArtifactStruct(paths, repoRoot, nModes);

localWriteJson(paths.summary_json, summary);
localWriteMarkdown(paths.report_md, summary, modeRows);
localWriteManifest(paths.manifest_json, summary, paths, repoRoot, nModes);

fprintf('S1 diagnosis: %s\n', summary.overall.classification);
fprintf('Report: %s\n', paths.report_md);
diary('off');
end

function paths = localPaths(repoRoot, outDir, inputMat, reportStem)
prefix = fullfile(outDir, reportStem);
paths = struct();
if isempty(inputMat)
    paths.input_mat = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
        'exp3_authoritative_mesh_convergence', 'mesh_400x50', ...
        'exp3_authoritative_400x50_result.mat');
    paths.input_json = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
        'exp3_authoritative_mesh_convergence', 'mesh_400x50', ...
        'exp3_authoritative_400x50_result.json');
    paths.input_config = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
        'exp3_authoritative_mesh_convergence', 'mesh_400x50', ...
        'exp3_authoritative_400x50_config.json');
    paths.input_topology_csv = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
        'exp3_authoritative_mesh_convergence', 'mesh_400x50', ...
        'exp3_authoritative_400x50_topology.csv');
else
    paths.input_mat = inputMat;
    [inputDir, inputBase] = fileparts(inputMat);
    inputStem = regexprep(inputBase, '_result$', '');
    paths.input_json = fullfile(inputDir, [inputStem, '_result.json']);
    paths.input_config = fullfile(inputDir, [inputStem, '_config.json']);
    paths.input_topology_csv = fullfile(inputDir, [inputStem, '_topology.csv']);
end
paths.log = [prefix, '_run.log'];
paths.report_md = fullfile(outDir, [reportStem, '_mode_diagnosis.md']);
paths.summary_json = [prefix, '_mode_summary.json'];
paths.mode_summary_csv = [prefix, '_modes_summary.csv'];
paths.manifest_json = [prefix, '_manifest.json'];
paths.energy_csv = cell(10, 1);
paths.mode_png = cell(6, 1);
for k = 1:10
    paths.energy_csv{k} = fullfile(outDir, sprintf('%s_mode_%02d_energy.csv', reportStem, k));
end
for k = 1:6
    paths.mode_png{k} = fullfile(outDir, sprintf('%s_mode_%02d_shape.png', reportStem, k));
end
end

function row = localBlankModeRow()
row = struct( ...
    'mode', NaN, ...
    'omega_rad_s', NaN, ...
    'frequency_hz', NaN, ...
    'mac_to_solid_reference_mode_1', NaN, ...
    'modal_mass', NaN, ...
    'modal_mass_residual', NaN, ...
    'stiffness_rayleigh', NaN, ...
    'eigen_residual_relative', NaN, ...
    'kinetic_energy_total', NaN, ...
    'strain_energy_total', NaN, ...
    'low_density_kinetic_fraction', NaN, ...
    'low_density_strain_fraction', NaN, ...
    'low_density_displacement_fraction', NaN, ...
    'kinetic_localization_index', NaN, ...
    'strain_localization_index', NaN, ...
    'kinetic_effective_element_fraction', NaN, ...
    'strain_effective_element_fraction', NaN, ...
    'kinetic_top_1pct_fraction', NaN, ...
    'strain_top_1pct_fraction', NaN, ...
    'kinetic_top_5pct_fraction', NaN, ...
    'strain_top_5pct_fraction', NaN, ...
    'dominant_solid_component_id', NaN, ...
    'dominant_solid_component_kinetic_fraction', NaN, ...
    'dominant_solid_component_strain_fraction', NaN, ...
    'dominant_component_touches_left_support', false, ...
    'dominant_component_touches_right_support', false, ...
    'dominant_component_touches_both_supports', false, ...
    'largest_support_component_kinetic_fraction', NaN, ...
    'largest_support_component_strain_fraction', NaN, ...
    'classification', '', ...
    'classification_reason', '');
end

function massInterp = localMassInterpolationFromConfig(cfg)
massInterp = struct('mode', 'power', 'pmass', 1.0);
if isfield(cfg, 'optimization') && isfield(cfg.optimization, 'mass_interpolation') && ...
        isstruct(cfg.optimization.mass_interpolation)
    mi = cfg.optimization.mass_interpolation;
    if isfield(mi, 'mode') && ~isempty(mi.mode)
        massInterp.mode = char(string(mi.mode));
    end
    if isfield(mi, 'pmass') && ~isempty(mi.pmass)
        massInterp.pmass = double(mi.pmass);
    elseif isfield(mi, 'exponent') && ~isempty(mi.exponent)
        massInterp.pmass = double(mi.exponent);
    end
elseif isfield(cfg, 'optimization') && isfield(cfg.optimization, 'pmass') && ...
        ~isempty(cfg.optimization.pmass)
    massInterp.pmass = double(cfg.optimization.pmass);
end
modeKey = lower(strtrim(massInterp.mode));
if any(strcmp(modeKey, {'', 'power', 'simp_power', 'pmass'}))
    massInterp.mode = 'power';
elseif strcmp(modeKey, 'linear')
    massInterp.mode = 'linear';
    massInterp.pmass = 1.0;
elseif any(strcmp(modeKey, {'du2007_c1', 'du_olhoff_c1', 'eq4b'}))
    massInterp.mode = 'du2007_c1';
else
    error('S1:InvalidMassInterpolation', ...
        'Unsupported mass interpolation mode "%s".', massInterp.mode);
end
if ~isfinite(massInterp.pmass) || massInterp.pmass <= 0
    error('S1:InvalidMassExponent', 'Mass exponent must be positive finite.');
end
end

function [K, M, KE, ME, edofMat, elemCenters] = localAssembleKM( ...
    x, nelx, nely, L, H, thickness, E0, Emin, nu, rho0, rho_min, penal, massInterp)
hx = L / nelx;
hy = H / nely;
ndof = 2 * (nelx + 1) * (nely + 1);
KE = thickness * localQ4Stiffness(hx, hy, nu);
ME = thickness * localQ4Mass(hx, hy);
edofMat = localEdofMat(nelx, nely);
iK = reshape(kron(edofMat, ones(1,8))', [], 1);
jK = reshape(kron(edofMat, ones(8,1))', [], 1);
Eelem = Emin + x(:)'.^penal * (E0 - Emin);
[rhoElem, ~] = our_mass_interpolation(x(:)', rho0, rho_min, massInterp.mode, massInterp.pmass);
sK = reshape(KE(:) * Eelem, [], 1);
sM = reshape(ME(:) * rhoElem, [], 1);
K = sparse(iK, jK, sK, ndof, ndof);
M = sparse(iK, jK, sM, ndof, ndof);
K = (K + K') / 2;
M = (M + M') / 2;

elemCenters = zeros(nelx*nely, 2);
for elx = 0:nelx-1
    for ely = 0:nely-1
        e = ely + elx*nely + 1;
        elemCenters(e,:) = [(elx + 0.5) * hx, (ely + 0.5) * hy];
    end
end
end

function edofMat = localEdofMat(nelx, nely)
edofMat = zeros(nelx*nely, 8);
for elx = 0:nelx-1
    for ely = 0:nely-1
        el  = ely + elx*nely + 1;
        n1  = (nely+1)*elx + ely;
        n2  = (nely+1)*(elx+1) + ely;
        n3  = n2 + 1;
        n4  = n1 + 1;
        edofMat(el,:) = [2*n1+1, 2*n1+2, ...
                         2*n2+1, 2*n2+2, ...
                         2*n3+1, 2*n3+2, ...
                         2*n4+1, 2*n4+2];
    end
end
end

function KE = localQ4Stiffness(hx, hy, nu)
E = 1.0;
D = (E / (1 - nu^2)) * [1, nu, 0; nu, 1, 0; 0, 0, 0.5*(1-nu)];
invJ = [2/hx, 0; 0, 2/hy];
detJ = 0.25 * hx * hy;
gp = 1 / sqrt(3);
gaussPts = [-gp, gp];
KE = zeros(8, 8);
for xi = gaussPts
    for eta = gaussPts
        dN_dxi  = 0.25 * [-(1-eta),  (1-eta),  (1+eta), -(1+eta)];
        dN_deta = 0.25 * [-(1-xi),  -(1+xi),   (1+xi),   (1-xi)];
        dN_xy = invJ * [dN_dxi; dN_deta];
        dN_dx = dN_xy(1, :);
        dN_dy = dN_xy(2, :);
        B = zeros(3, 8);
        B(1, 1:2:end) = dN_dx;
        B(2, 2:2:end) = dN_dy;
        B(3, 1:2:end) = dN_dy;
        B(3, 2:2:end) = dN_dx;
        KE = KE + (B' * D * B) * detJ;
    end
end
end

function ME = localQ4Mass(hx, hy)
area = hx * hy;
Ms = (area / 36) * [4, 2, 1, 2;
                    2, 4, 2, 1;
                    1, 2, 4, 2;
                    2, 1, 2, 4];
ME = kron(Ms, eye(2));
end

function [kineticElem, strainElem] = localElementEnergies( ...
    phi, omega, x, edofMat, KE, ME, E0, Emin, rho0, rho_min, penal, massInterp)
nEl = numel(x);
kineticElem = zeros(nEl, 1);
strainElem = zeros(nEl, 1);
for e = 1:nEl
    ue = phi(edofMat(e,:));
    keScale = Emin + x(e)^penal * (E0 - Emin);
    [rhoScale, ~] = our_mass_interpolation(x(e), rho0, rho_min, massInterp.mode, massInterp.pmass);
    strainElem(e) = 0.5 * real(ue' * (keScale * KE) * ue);
    kineticElem(e) = 0.5 * omega^2 * real(ue' * (rhoScale * ME) * ue);
end
end

function row = localDiagnoseMode(k, omega, freqHz, phi, M, K, free, edofMat, macVal, kineticElem, strainElem, ...
    x, elemCenters, component, compStats, largestCompId, lowMask, lowDensityThreshold)
row = localBlankModeRow();
row.mode = k;
row.omega_rad_s = omega;
row.frequency_hz = freqHz;
row.mac_to_solid_reference_mode_1 = macVal;
row.modal_mass = real(phi' * (M * phi));
row.modal_mass_residual = abs(row.modal_mass - 1);
row.stiffness_rayleigh = real(phi' * (K * phi));
freeResidual = K(free,:) * phi - omega^2 * (M(free,:) * phi);
row.eigen_residual_relative = norm(freeResidual) / max(norm(K(free,:) * phi), eps);
row.kinetic_energy_total = sum(kineticElem);
row.strain_energy_total = sum(strainElem);
row.low_density_kinetic_fraction = localFrac(kineticElem(lowMask), kineticElem);
row.low_density_strain_fraction = localFrac(strainElem(lowMask), strainElem);

dispElem = localElementDisplacementEnergy(phi, edofMat);
row.low_density_displacement_fraction = localFrac(dispElem(lowMask), dispElem);

[row.kinetic_localization_index, row.kinetic_effective_element_fraction, ...
    row.kinetic_top_1pct_fraction, row.kinetic_top_5pct_fraction] = localLocalization(kineticElem);
[row.strain_localization_index, row.strain_effective_element_fraction, ...
    row.strain_top_1pct_fraction, row.strain_top_5pct_fraction] = localLocalization(strainElem);

[domId, domK, domS] = localDominantComponent(component, kineticElem, strainElem);
row.dominant_solid_component_id = domId;
row.dominant_solid_component_kinetic_fraction = domK;
row.dominant_solid_component_strain_fraction = domS;
if domId > 0 && domId <= numel(compStats)
    row.dominant_component_touches_left_support = compStats(domId).touches_left;
    row.dominant_component_touches_right_support = compStats(domId).touches_right;
    row.dominant_component_touches_both_supports = compStats(domId).touches_left && compStats(domId).touches_right;
end
if largestCompId > 0
    supportMask = component == largestCompId;
    row.largest_support_component_kinetic_fraction = localFrac(kineticElem(supportMask), kineticElem);
    row.largest_support_component_strain_fraction = localFrac(strainElem(supportMask), strainElem);
end

[row.classification, row.classification_reason] = localClassify(row, lowDensityThreshold);
end

function dispElem = localElementDisplacementEnergy(phi, edofMat)
nEl = size(edofMat, 1);
dispElem = zeros(nEl, 1);
for e = 1:nEl
    ue = phi(edofMat(e,:));
    dispElem(e) = sum(real(ue(:)).^2);
end
end

function [classification, reason] = localClassify(row, ~)
if row.low_density_kinetic_fraction >= 0.35 || row.low_density_strain_fraction >= 0.35
    classification = 'localized low-density mode';
    reason = sprintf('low-density energy fractions are high: kinetic %.3f, strain %.3f', ...
        row.low_density_kinetic_fraction, row.low_density_strain_fraction);
elseif row.kinetic_effective_element_fraction < 0.025 || row.strain_effective_element_fraction < 0.025
    if ~row.dominant_component_touches_both_supports && ...
            (row.dominant_solid_component_kinetic_fraction >= 0.25 || ...
             row.dominant_solid_component_strain_fraction >= 0.25)
        classification = 'disconnected-component mode';
        reason = sprintf(['energy is localized and dominant solid component %d does not touch both supports ', ...
            '(kinetic frac %.3f, strain frac %.3f)'], ...
            row.dominant_solid_component_id, ...
            row.dominant_solid_component_kinetic_fraction, ...
            row.dominant_solid_component_strain_fraction);
    else
        classification = 'ambiguous';
        reason = sprintf('localized by participation metric but not clearly low-density or disconnected: effK %.3f, effS %.3f', ...
            row.kinetic_effective_element_fraction, row.strain_effective_element_fraction);
    end
elseif row.largest_support_component_kinetic_fraction >= 0.70 && ...
        row.largest_support_component_strain_fraction >= 0.70 && ...
        row.dominant_component_touches_both_supports
    classification = 'physical global mode';
    reason = sprintf('energy is mostly on support-connected component: kinetic %.3f, strain %.3f', ...
        row.largest_support_component_kinetic_fraction, ...
        row.largest_support_component_strain_fraction);
else
    classification = 'ambiguous';
    reason = sprintf('mixed energy/component indicators: lowK %.3f, lowS %.3f, supportK %.3f, supportS %.3f', ...
        row.low_density_kinetic_fraction, row.low_density_strain_fraction, ...
        row.largest_support_component_kinetic_fraction, ...
        row.largest_support_component_strain_fraction);
end
end

function [index, effectiveFraction, top1, top5] = localLocalization(e)
e = max(real(e(:)), 0);
n = numel(e);
total = sum(e);
if total <= 0
    index = NaN;
    effectiveFraction = NaN;
    top1 = NaN;
    top5 = NaN;
    return;
end
p = e / total;
index = sum(p.^2);
effectiveFraction = 1 / (n * index);
s = sort(p, 'descend');
top1 = sum(s(1:max(1, ceil(0.01*n))));
top5 = sum(s(1:max(1, ceil(0.05*n))));
end

function f = localFrac(partVals, allVals)
den = sum(max(real(allVals(:)), 0));
if den <= 0
    f = NaN;
else
    f = sum(max(real(partVals(:)), 0)) / den;
end
end

function [component, stats] = localSolidComponents(x, nelx, nely, threshold)
solid = reshape(x(:) >= threshold, nely, nelx);
component = zeros(nely, nelx);
stats = repmat(struct('id', 0, 'size', 0, 'touches_left', false, ...
    'touches_right', false, 'touches_bottom', false, 'touches_top', false), 0, 1);
compId = 0;
for ix = 1:nelx
    for iy = 1:nely
        if ~solid(iy, ix) || component(iy, ix) ~= 0
            continue;
        end
        compId = compId + 1;
        q = zeros(nnz(solid), 2);
        head = 1;
        tail = 1;
        q(tail,:) = [iy, ix];
        component(iy, ix) = compId;
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
            neigh = [cy-1, cx; cy+1, cx; cy, cx-1; cy, cx+1];
            for a = 1:4
                ny = neigh(a,1);
                nx = neigh(a,2);
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

function largestCompId = localLargestSupportConnectedComponent(stats)
largestCompId = 0;
largestSize = -Inf;
for i = 1:numel(stats)
    if stats(i).touches_left && stats(i).touches_right && stats(i).size > largestSize
        largestSize = stats(i).size;
        largestCompId = stats(i).id;
    end
end
end

function [domId, fracK, fracS] = localDominantComponent(component, kineticElem, strainElem)
ids = unique(component(component > 0));
if isempty(ids)
    domId = NaN;
    fracK = NaN;
    fracS = NaN;
    return;
end
scores = zeros(numel(ids), 1);
fracKs = zeros(numel(ids), 1);
fracSs = zeros(numel(ids), 1);
for i = 1:numel(ids)
    mask = component == ids(i);
    fracKs(i) = localFrac(kineticElem(mask), kineticElem);
    fracSs(i) = localFrac(strainElem(mask), strainElem);
    scores(i) = max(fracKs(i), fracSs(i));
end
[~, idx] = max(scores);
domId = ids(idx);
fracK = fracKs(idx);
fracS = fracSs(idx);
end

function localWriteEnergyTable(path, x, elemCenters, component, compStats, kineticElem, strainElem, row)
nEl = numel(x);
compTouchesLeft = false(nEl, 1);
compTouchesRight = false(nEl, 1);
compTouchesBoth = false(nEl, 1);
for e = 1:nEl
    cid = component(e);
    if cid > 0 && cid <= numel(compStats)
        compTouchesLeft(e) = compStats(cid).touches_left;
        compTouchesRight(e) = compStats(cid).touches_right;
        compTouchesBoth(e) = compStats(cid).touches_left && compStats(cid).touches_right;
    end
end
kinTotal = sum(kineticElem);
strTotal = sum(strainElem);
T = table((1:nEl)', elemCenters(:,1), elemCenters(:,2), x(:), component(:), ...
    compTouchesLeft, compTouchesRight, compTouchesBoth, ...
    kineticElem(:), strainElem(:), kineticElem(:)/max(kinTotal, eps), ...
    strainElem(:)/max(strTotal, eps), x(:) < 0.05, ...
    repmat(row.mode, nEl, 1), ...
    'VariableNames', {'element','x_center','y_center','density','solid_component_id', ...
    'component_touches_left_support','component_touches_right_support', ...
    'component_touches_both_supports','kinetic_energy','strain_energy', ...
    'kinetic_energy_fraction','strain_energy_fraction','is_low_density_rho_lt_0_05','mode'});
writetable(T, path);
end

function localSaveModeShape(path, phi, x, nelx, nely, L, H, omega, modeIdx, row)
xGrid = repmat((0:nelx) * (L / nelx), nely + 1, 1);
yGrid = repmat((0:nely)' * (H / nely), 1, nelx + 1);
ux = reshape(phi(1:2:end), nely + 1, nelx + 1);
uy = reshape(phi(2:2:end), nely + 1, nelx + 1);
mag = hypot(ux, uy);
scale = 0.075 * max(L, H) / max(max(mag(:)), eps);
xDef = xGrid + scale * ux;
yDef = yGrid + scale * uy;

fig = figure('Color', 'white', 'Visible', 'off');
ax = axes('Parent', fig);
hold(ax, 'on');
topo = reshape(x(:), nely, nelx);
imagesc(ax, [0, L], [0, H], flipud(topo));
set(ax, 'YDir', 'normal');
colormap(ax, gray);
alpha(0.35);
[rows, cols] = localModePlotLineIndices(nely, nelx);
for r = rows
    plot(ax, xDef(r,:), yDef(r,:), '-', 'Color', [0.05, 0.25, 0.75], 'LineWidth', 0.65);
end
for c = cols
    plot(ax, xDef(:,c), yDef(:,c), '-', 'Color', [0.05, 0.25, 0.75], 'LineWidth', 0.65);
end
plot(ax, [0, L, L, 0, 0], [0, 0, H, H, 0], '-', 'Color', [0.15,0.15,0.15], 'LineWidth', 1);
axis(ax, 'equal');
xlim(ax, [-0.05*L, 1.05*L]);
ylim(ax, [-0.15*H, 1.15*H]);
set(ax, 'XTick', [], 'YTick', []);
box(ax, 'on');
title(ax, sprintf('S1 Exp3 400x50 mode %d | omega %.4f rad/s | %s', ...
    modeIdx, omega, row.classification), 'Interpreter', 'none', 'FontSize', 9);
exportgraphics(fig, path, 'Resolution', 180, 'BackgroundColor', 'white');
close(fig);
end

function [rowIdx, colIdx] = localModePlotLineIndices(nely, nelx)
maxLinesPerDirection = 120;
rowStride = max(1, ceil((nely + 1) / maxLinesPerDirection));
colStride = max(1, ceil((nelx + 1) / maxLinesPerDirection));
commonStride = max(rowStride, colStride);
rowIdx = unique([1:commonStride:(nely + 1), nely + 1]);
colIdx = unique([1:commonStride:(nelx + 1), nelx + 1]);
end

function c = localComponentSummary(stats, largestCompId)
sizes = [stats.size];
c = struct();
c.count = numel(stats);
c.largest_support_connected_component_id = largestCompId;
if isempty(sizes)
    c.largest_component_size = 0;
    c.support_connected_component_count = 0;
else
    c.largest_component_size = max(sizes);
    c.support_connected_component_count = sum([stats.touches_left] & [stats.touches_right]);
end
end

function overall = localOverallConclusion(modeRows)
classes = string({modeRows.classification});
lowCount = sum(classes == "localized low-density mode");
discCount = sum(classes == "disconnected-component mode");
physCount = sum(classes == "physical global mode");
ambCount = sum(classes == "ambiguous");
firstThree = classes(1:min(3, numel(classes)));
if any(firstThree == "localized low-density mode")
    cls = 'likely localized/spurious low-density mode influence';
    reason = 'At least one of the first three modes is classified as localized low-density.';
elseif any(firstThree == "disconnected-component mode")
    cls = 'likely disconnected-component pathology';
    reason = 'At least one of the first three modes is classified as disconnected-component.';
elseif all(firstThree == "physical global mode")
    cls = 'likely physically valid but different topology';
    reason = 'The first three modes are classified as physical global modes on the support-connected component.';
else
    cls = 'ambiguous';
    reason = 'The first modes do not clearly separate localized, disconnected, and global indicators.';
end
overall = struct('classification', cls, 'reason', reason, ...
    'physical_global_count', physCount, ...
    'localized_low_density_count', lowCount, ...
    'disconnected_component_count', discCount, ...
    'ambiguous_count', ambCount);
end

function localWriteMarkdown(path, summary, modeRows)
fid = fopen(path, 'w');
if fid < 0
    error('S1:CannotWriteReport', 'Cannot write report: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '# S1 Exp3 400x50 Mode Diagnosis\n\n');
fprintf(fid, 'Scope: postprocessing-only diagnostic of the saved failed Exp3 400x50 alpha=1.00 case. No optimization rerun, solver-code edit, manuscript edit, or Exp2/Exp3/CR2/P1/A4 rerun was performed.\n\n');
fprintf(fid, '## Conclusion\n\n');
fprintf(fid, '**%s.** %s\n\n', summary.overall.classification, summary.overall.reason);
fprintf(fid, 'Saved Exp3 fine-case status: final tracked MAC `%.12g`, tracked omega `%.12g` rad/s.\n\n', ...
    summary.reference.saved_final_tracked_mac, summary.reference.saved_final_tracked_omega_rad_s);
fprintf(fid, '## Setup\n\n');
fprintf(fid, '| item | value |\n|---|---:|\n');
fprintf(fid, '| mesh | %dx%d |\n', summary.mesh.nelx, summary.mesh.nely);
fprintf(fid, '| domain | %.12g x %.12g |\n', summary.mesh.L, summary.mesh.H);
fprintf(fid, '| element size | %.12g x %.12g |\n', summary.mesh.hx, summary.mesh.hy);
fprintf(fid, '| fixed DOFs | %d |\n', summary.supports.fixed_dof_count);
fprintf(fid, '| low-density threshold | %.12g |\n', summary.thresholds.low_density);
fprintf(fid, '| solid-component threshold | %.12g |\n', summary.thresholds.solid_component);
fprintf(fid, '| solid components | %d |\n', summary.components.count);
fprintf(fid, '| support-connected components | %d |\n', summary.components.support_connected_component_count);
fprintf(fid, '| largest support-connected component id | %d |\n\n', ...
    summary.components.largest_support_connected_component_id);
fprintf(fid, '## Mode Summary\n\n');
fprintf(fid, '| mode | omega rad/s | Hz | MAC(ref mode 1) | mass residual | eig residual | low-density K frac | low-density S frac | eff elem frac K | eff elem frac S | support-comp K frac | support-comp S frac | class |\n');
fprintf(fid, '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n');
for i = 1:numel(modeRows)
    r = modeRows(i);
    fprintf(fid, '| %d | %.12g | %.12g | %.12g | %.3e | %.3e | %.6g | %.6g | %.6g | %.6g | %.6g | %.6g | %s |\n', ...
        r.mode, r.omega_rad_s, r.frequency_hz, r.mac_to_solid_reference_mode_1, ...
        r.modal_mass_residual, r.eigen_residual_relative, ...
        r.low_density_kinetic_fraction, r.low_density_strain_fraction, ...
        r.kinetic_effective_element_fraction, r.strain_effective_element_fraction, ...
        r.largest_support_component_kinetic_fraction, r.largest_support_component_strain_fraction, ...
        r.classification);
end
fprintf(fid, '\n## Classification Reasons\n\n');
for i = 1:numel(modeRows)
    fprintf(fid, '- Mode %d: `%s` - %s\n', modeRows(i).mode, ...
        modeRows(i).classification, modeRows(i).classification_reason);
end
fprintf(fid, '\n## Artifacts\n\n');
fprintf(fid, '- JSON summary: `%s`\n', summary.artifacts.summary_json);
fprintf(fid, '- mode summary CSV: `%s`\n', summary.artifacts.mode_summary_csv);
fprintf(fid, '- manifest: `%s`\n', summary.artifacts.manifest_json);
fprintf(fid, '- per-mode energy CSVs: `s1_exp3_400x50_mode_##_energy.csv`\n');
fprintf(fid, '- first six mode-shape figures: `s1_exp3_400x50_mode_##_shape.png`\n');
fprintf(fid, '\n## Interpretation\n\n');
fprintf(fid, 'The classification is based on final-topology K/M reassembly, the first 10 eigenmodes, MAC to the saved solid-reference target mode, elementwise kinetic and strain energy fractions in `rho < %.3g` regions, energy participation/localization metrics, and association with solid components touching the support lines.\n', ...
    summary.thresholds.low_density);
end

function localWriteManifest(path, summary, paths, repoRoot, nModes)
manifest = struct();
manifest.study = summary.study;
manifest.scope = summary.scope;
manifest.created_utc = summary.created_utc;
manifest.inputs = summary.input;
manifest.overall = summary.overall;
manifest.outputs = localArtifactStruct(paths, repoRoot, nModes);
manifest.no_optimization_rerun = true;
manifest.no_solver_code_edit = true;
manifest.no_manuscript_edit = true;
localWriteJson(path, manifest);
end

function artifacts = localArtifactStruct(paths, repoRoot, nModes)
artifacts = struct();
artifacts.log = localRel(paths.log, repoRoot);
artifacts.report_md = localRel(paths.report_md, repoRoot);
artifacts.summary_json = localRel(paths.summary_json, repoRoot);
artifacts.mode_summary_csv = localRel(paths.mode_summary_csv, repoRoot);
artifacts.manifest_json = localRel(paths.manifest_json, repoRoot);
artifacts.energy_csv = cell(nModes, 1);
for k = 1:nModes
    artifacts.energy_csv{k} = localRel(paths.energy_csv{k}, repoRoot);
end
artifacts.mode_shape_png = cell(min(6, nModes), 1);
for k = 1:min(6, nModes)
    artifacts.mode_shape_png{k} = localRel(paths.mode_png{k}, repoRoot);
end
end

function localWriteJson(path, data)
txt = jsonencode(data, 'PrettyPrint', true);
fid = fopen(path, 'w');
if fid < 0
    error('S1:CannotWriteJson', 'Cannot write JSON: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, txt, 'char');
fwrite(fid, newline, 'char');
end

function rel = localRel(path, repoRoot)
path = char(path);
repoRoot = char(repoRoot);
if startsWith(path, [repoRoot, filesep])
    rel = strrep(path(numel(repoRoot)+2:end), filesep, '/');
else
    rel = strrep(path, filesep, '/');
end
end
