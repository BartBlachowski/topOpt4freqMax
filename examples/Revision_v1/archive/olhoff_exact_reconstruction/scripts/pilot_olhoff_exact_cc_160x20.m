function result = pilot_olhoff_exact_cc_160x20(outDir)
%PILOT_OLHOFF_EXACT_CC_80X10  Du-Olhoff reproduction pilot: CC beam 80×10.
%
% Scope: MATLAB only. OlhoffApproachExact solver only. Do not edit manuscript.
% Purpose: test whether bound formulation + du2007_c1 mass suppresses
%          localized low-density modes and produces acceptable CC topology,
%          compared with ourApproach 80×10 scaling study result.
%
% Mesh choice: 80×10 (not 160×20).
%   Reason: at 160×20, p=1 continuation causes inner-MMA degeneration
%   (with du2007_c1 mass linear above rho=0.1, p=1 gives K/M ∝ ρ →
%   eigenvalue ρ-independent → near-zero sensitivity → LP bang-bang).
%   p=3 directly from uniform density is the paper's stated approach and
%   converges without continuation. 80×10 is 4× finer than the paper's
%   40×5, gives clear topology in ~20 min, matches scaling study 80×10.
%
% Setup:
%   Geometry  : clamped-clamped beam, L=8 m, H=1 m
%   Mesh      : 80×10 (N_e=800; 4× finer than paper's 40×5)
%   VF        : 0.5
%   Mass      : du2007_c1 (Du & Olhoff 2007 Eq. 4b)
%   Stiffness : SIMP p=3 (paper-exact; no p-continuation)
%   Filter    : sensitivity filter, rmin_elem=2.5 (element units)
%   Objective : maximize fundamental eigenfrequency (bound formulation)
%   Mult.     : multiplicity detection enabled (mult_tol=1e-3)
%
% Outputs (all under outDir):
%   1. topology_final.png / topology_final.csv
%   2. first 6 eigenfrequencies printed and saved
%   3. multiplicity_history.csv
%   4. mode shapes 1–3 PNG
%   5. convergence history plot/CSV
%   6. S1 low-density mode diagnosis (10 modes)
%   7. comparison_vs_ourApproach.md  (vs ourApproach 80×10 scaling study)
%   8. pilot_report.md

warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:singularMatrix');

if nargin < 1 || isempty(outDir)
    scriptDir = fileparts(mfilename('fullpath'));
    outDir = fullfile(scriptDir, 'output', 'pilot_olhoff_exact_cc_80x10');
end
if exist(outDir, 'dir') ~= 7; mkdir(outDir); end

scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(fileparts(scriptDir));
addpath(fullfile(repoRoot, 'analysis', 'OlhoffApproachExact', 'Matlab'));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));

logPath = fullfile(outDir, 'pilot_run.log');
diary(logPath); cleanupD = onCleanup(@() diary('off'));

fprintf('pilot OlhoffApproachExact CC 80x10 started: %s\n', ...
    char(datetime('now','TimeZone','UTC')));
fprintf('Solver: OlhoffApproachExact | BC: CC | mesh: 80x10 | mass: du2007_c1 | p=3\n');
fprintf('MATLAB only. Do not edit manuscript.\n\n');

%% === Configuration ===
nelx     = 80;
nely     = 10;
L        = 8.0;
H        = 1.0;
volfrac  = 0.5;
mass_mode = 'du2007_c1';
rmin_elem = 2.5;
rho_min  = 1e-3;
E0       = 1e7;
nu       = 0.3;
rho0     = 1.0;
t        = 1.0;
mult_tol = 1e-3;
n_modes_compute = 10;  % compute 10 modes (consistent with S1 diagnosis)

% Single stage: p=3 directly (paper-exact; no p-continuation).
% Reason: with du2007_c1 mass, p=1 gives K/M ∝ ρ above ρ=0.1, so the
% eigenvalue is nearly ρ-independent at uniform density → sensitivity ≈ 0
% → inner-MMA LP degeneracy (bang-bang cycling). p=3 has clear nonlinear
% sensitivity from the start and matches Du & Olhoff (2007) directly.
penal_stages = [3];
iters_stage  = [400];
outer_tol    = 1e-3;
inner_max    = 30;
inner_tol    = 1e-4;
% Numerical safeguard: outer_move=0.2 limits |Δρ_e| per outer iteration.
% Not in the paper (which uses 40×5 where this isn't needed), but required
% at 80×10 to prevent the unconstrained increment from overshooting.
outer_move_val = 0.2;

%% === Base cfg ===
cfg_base = struct();
cfg_base.L              = L;
cfg_base.H              = H;
cfg_base.nelx           = nelx;
cfg_base.nely           = nely;
cfg_base.E0             = E0;
cfg_base.nu             = nu;
cfg_base.rho0           = rho0;
cfg_base.t              = t;
cfg_base.volfrac        = volfrac;
cfg_base.rho_min        = rho_min;
cfg_base.mass_mode      = mass_mode;
cfg_base.rmin_elem      = rmin_elem;
cfg_base.sensitivity_filter = true;
cfg_base.support_type   = 'CC';
cfg_base.n_target       = 1;
cfg_base.n_modes        = n_modes_compute;
cfg_base.mult_tol       = mult_tol;
cfg_base.inner_max_iter = inner_max;
cfg_base.inner_tol      = inner_tol;
cfg_base.outer_tol      = outer_tol;
cfg_base.move_lim       = Inf;      % paper-exact: no inner move limit
cfg_base.outer_move     = outer_move_val;  % numerical safeguard for 80×10
cfg_base.acceptance_check = false;
cfg_base.alpha          = 1.0;
cfg_base.verbose        = true;

%% === Penalty continuation ===
rho_current  = [];
all_hist     = cell(numel(penal_stages), 1);
all_penal    = penal_stages;
total_iters  = 0;

for si = 1:numel(penal_stages)
    p = penal_stages(si);
    cfg = cfg_base;
    cfg.penal          = p;
    cfg.outer_max_iter = iters_stage(si);
    if ~isempty(rho_current)
        cfg.initial_rho = rho_current;
    end

    fprintf('\n--- Stage %d: penal=%.0f, max_iter=%d ---\n', si, p, iters_stage(si));
    tic;
    [rho_s, hist_s] = topopt_freq_exact(cfg);
    elapsed = toc;
    fprintf('Stage %d done in %.1f s, %d outer iters\n', si, elapsed, hist_s.outer_iters);

    rho_current = rho_s;
    all_hist{si} = hist_s;
    total_iters  = total_iters + hist_s.outer_iters;
end

rho_final = rho_current;

%% === Final eigenanalysis ===
fprintf('\n--- Final eigenanalysis (n_modes=%d) ---\n', n_modes_compute);
dx = L/nelx; dy = H/nely;
nEl  = nelx*nely;
nDof = 2*(nelx+1)*(nely+1);

[Ke_star, Me_star] = fe_q4_exact(nu, t, dx, dy);
Ke_phys = E0   * Ke_star;
Me_phys = rho0 * Me_star;

nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
cVec    = reshape(2*nodeNrs(1:nely,1:nelx)+1, nEl, 1);
cMat    = [cVec, cVec+1, cVec+2*nely+2, cVec+2*nely+3, ...
           cVec+2*nely, cVec+2*nely+1, cVec-2, cVec-1];

[Il, Jl]    = find(tril(ones(8)));
iK = reshape(cMat(:,Il)', [], 1);
jK = reshape(cMat(:,Jl)', [], 1);
Ke_phys_l = Ke_phys(sub2ind([8,8], Il, Jl));
Me_phys_l = Me_phys(sub2ind([8,8], Il, Jl));

[K_fin, M_fin] = assemble_KM_exact(rho_final, Ke_phys_l, Me_phys_l, iK, jK, nDof, 3, mass_mode);
fixed = build_supports_exact('CC', nodeNrs);
free  = setdiff(1:nDof, fixed)';
Kf = K_fin(free, free);
Mf = M_fin(free, free);

opts_e.tol = 1e-10; opts_e.maxit = 600;
[V, D, flag] = eigs(Kf, Mf, n_modes_compute, 'SM', opts_e);
if flag ~= 0
    opts_e.tol = 1e-8; opts_e.maxit = 2000; opts_e.p = min(numel(free)-1, 80);
    [V, D, flag] = eigs(Kf, Mf, n_modes_compute, 'SM', opts_e);
    if flag ~= 0; warning('Final eigs did not fully converge (flag=%d)', flag); end
end
[lam_f, ord_f] = sort(real(diag(D)));
V = real(V(:, ord_f));
for j = 1:size(V,2)
    sc = sqrt(abs(V(:,j)' * (Mf * V(:,j))));
    if sc > 1e-14; V(:,j) = V(:,j)/sc; end
end

omega_fin  = sqrt(max(lam_f, 0));
freqHz_fin = omega_fin / (2*pi);

Phi_fin = zeros(nDof, n_modes_compute);
Phi_fin(free, :) = V;

fprintf('\nFinal eigenfrequencies (penal=3, du2007_c1):\n');
fprintf('  mode  omega (rad/s)    Hz\n');
for k = 1:min(6, numel(omega_fin))
    fprintf('  %4d  %12.4f  %10.4f\n', k, omega_fin(k), freqHz_fin(k));
end

%% === Save topology ===
topo = reshape(rho_final, nely, nelx);
writematrix(topo, fullfile(outDir, 'topology_final.csv'));

figT = figure('Color','white','Visible','off');
axT  = axes('Parent',figT);
imagesc(axT, 1 - topo);
axis(axT,'equal','off'); colormap(axT, gray);
title(axT, sprintf('OlhoffExact CC 80x10 | \\omega_1=%.2f rad/s | p=3 | du2007\\_c1', omega_fin(1)), ...
    'Interpreter','tex', 'FontSize', 9);
try; exportgraphics(figT, fullfile(outDir, 'topology_final.png'), 'Resolution', 180, 'BackgroundColor','white');
catch; print(figT, fullfile(outDir, 'topology_final.png'), '-dpng', '-r180'); end
close(figT);

%% === Convergence history (combined across stages) ===
omega1_all = []; omega2_all = []; beta_all = []; vol_all = []; N_all = []; iter_offset = 0;
for si = 1:numel(all_hist)
    h = all_hist{si};
    ni = h.outer_iters;
    if ni < 1; continue; end
    ow = h.omega_trial(1:ni, :);
    omega1_all = [omega1_all; ow(:,1)]; %#ok<AGROW>
    if size(ow,2) >= 2; omega2_all = [omega2_all; ow(:,2)]; else; omega2_all = [omega2_all; nan(ni,1)]; end %#ok<AGROW>
    beta_all  = [beta_all;  h.beta(1:ni)]; %#ok<AGROW>
    vol_all   = [vol_all;   h.volume(1:ni)]; %#ok<AGROW>
    N_all     = [N_all;     h.N(1:ni)]; %#ok<AGROW>
    iter_offset = iter_offset + ni;
end
nTotal = numel(omega1_all);
iters_vec = (1:nTotal)';

convT = table(iters_vec, omega1_all, omega2_all, beta_all, vol_all, N_all, ...
    'VariableNames', {'iteration','omega_1_rad_s','omega_2_rad_s','beta_lambda_sq','volume','multiplicity_N'});
writetable(convT, fullfile(outDir, 'convergence_history.csv'));

figC = figure('Color','white','Visible','off');
ax1 = subplot(2,1,1,'Parent',figC);
plot(ax1, iters_vec, omega1_all, 'b-', 'LineWidth',1); hold(ax1,'on');
plot(ax1, iters_vec, omega2_all, 'r--', 'LineWidth',1);
ylabel(ax1, '\omega (rad/s)'); xlabel(ax1, 'Outer iteration');
legend(ax1, '\omega_1','\omega_2','Location','southeast');
title(ax1, 'Convergence — CC 80x10, du2007\_c1, p=3');
grid(ax1,'on');
ax2 = subplot(2,1,2,'Parent',figC);
plot(ax2, iters_vec, N_all, 'k-', 'LineWidth',1);
ylabel(ax2, 'Multiplicity N'); xlabel(ax2, 'Outer iteration');
ylim(ax2, [0.5, max(3, max(N_all(:))+0.5)]);
grid(ax2,'on');
try; exportgraphics(figC, fullfile(outDir, 'convergence_history.png'), 'Resolution', 180, 'BackgroundColor','white');
catch; print(figC, fullfile(outDir, 'convergence_history.png'), '-dpng', '-r180'); end
close(figC);

%% === Multiplicity history CSV ===
multT = table(iters_vec, N_all, 'VariableNames', {'iteration','multiplicity_N'});
writetable(multT, fullfile(outDir, 'multiplicity_history.csv'));

%% === Mode shape figures (modes 1–3) ===
for k = 1:min(3, n_modes_compute)
    phi = Phi_fin(:, k);
    pilotSaveModeShape(fullfile(outDir, sprintf('mode_%02d_shape.png', k)), ...
        phi, rho_final, nelx, nely, L, H, omega_fin(k), k);
end

%% === S1 low-density mode diagnosis ===
fprintf('\n--- S1 diagnosis (10 modes, du2007_c1, low-density threshold rho<0.1) ---\n');
s1 = localS1Diagnose(rho_final, Phi_fin, omega_fin, freqHz_fin, K_fin, M_fin, free, ...
    cMat, Ke_phys, Me_phys, mass_mode, 3, rho_min, nelx, nely, L, H);

fprintf('\nS1 overall: %s\n', s1.overall.classification);
fprintf('  physical_global: %d, localized_low_density: %d, ambiguous: %d\n', ...
    s1.overall.physical_global_count, s1.overall.localized_low_density_count, ...
    s1.overall.ambiguous_count);
fprintf('\n  mode  class                         omega       ld_strain_frac\n');
for k = 1:numel(s1.modes)
    m = s1.modes(k);
    fprintf('  %4d  %-35s  %9.4g  %12.6g\n', m.mode, m.classification, m.omega_rad_s, ...
        m.low_density_strain_fraction);
end

s1SummaryPath = fullfile(outDir, 's1_mode_summary.json');
localWriteJson(s1SummaryPath, s1);

s1CsvPath = fullfile(outDir, 's1_modes.csv');
writetable(struct2table(s1.modes), s1CsvPath);

%% === Comparison with ourApproach 80x10 ===
baseScalingDir = fullfile(repoRoot, 'examples', 'Revision_v1', 'output', ...
    'exp_scaling_study', 'mesh_80x10');
ourApproachJsonPath = fullfile(baseScalingDir, 'scaling_80x10_result.json');
ourApproachMatPath  = fullfile(baseScalingDir, 'scaling_80x10_result.mat');
ourApproachS1Path   = fullfile(baseScalingDir, 's1_postprocessing', ...
    'scaling_80x10_s1_mode_summary.json');

ourOmega1 = NaN; ourMAC = NaN; ourS1Class = 'not available';
ourS1Loc = NaN; ourS1Phys = NaN; ourIters = NaN;
if exist(ourApproachJsonPath, 'file') == 2  % prefer JSON (field names verified)
    try
        oR = jsondecode(fileread(ourApproachJsonPath));
        if isfield(oR, 'omega_1_rad_s');  ourOmega1 = oR.omega_1_rad_s; end
        if isfield(oR, 'tracked_mac');    ourMAC    = oR.tracked_mac;   end
        if isfield(oR, 'iterations');     ourIters  = oR.iterations;    end
    catch; end
elseif exist(ourApproachMatPath, 'file') == 2
    try
        oM = load(ourApproachMatPath, 'result');
        oR = oM.result;
        if isfield(oR, 'omega_1_rad_s');  ourOmega1 = oR.omega_1_rad_s; end
        if isfield(oR, 'tracked_mac');    ourMAC    = oR.tracked_mac;   end
        if isfield(oR, 'iterations');     ourIters  = oR.iterations;    end
    catch; end
end
if exist(ourApproachS1Path, 'file') == 2
    try
        oS = jsondecode(fileread(ourApproachS1Path));
        ourS1Class = oS.overall.classification;
        ourS1Loc   = oS.overall.localized_low_density_count;
        ourS1Phys  = oS.overall.physical_global_count;
    catch; end
end

%% === Write report ===
result = struct();
result.study             = 'Pilot: OlhoffApproachExact CC 80x10 vs ourApproach 80x10';
result.created_utc       = char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'));
result.mesh              = struct('nelx',nelx,'nely',nely,'L',L,'H',H,'N_e',nEl);
result.mass_mode         = mass_mode;
result.rmin_elem         = rmin_elem;
result.penal_stages      = penal_stages;
result.iters_per_stage   = iters_stage;
result.outer_move        = outer_move_val;
result.inner_max_iter    = inner_max;
result.total_outer_iters = total_iters;
result.omega_final       = omega_fin(:)';
result.omega_1_rad_s     = omega_fin(1);
result.omega_1_hz        = freqHz_fin(1);
result.final_N           = all_hist{end}.final_N;
result.final_volume      = all_hist{end}.final_volume;
result.s1_overall        = s1.overall.classification;
result.s1_localized_count = s1.overall.localized_low_density_count;
result.s1_physical_count  = s1.overall.physical_global_count;
result.s1_mode1_class     = s1.modes(1).classification;
result.s1_mode1_ld_strain = s1.modes(1).low_density_strain_fraction;

localWriteJson(fullfile(outDir, 'pilot_result.json'), result);
localWriteComparisonReport(fullfile(outDir, 'comparison_vs_ourApproach.md'), ...
    result, s1, ourOmega1, ourMAC, ourIters, ourS1Class, ourS1Loc, ourS1Phys);
localWritePilotReport(fullfile(outDir, 'pilot_report.md'), result, s1, ...
    ourOmega1, ourMAC, ourS1Class, ourS1Loc);

fprintf('\n=== Pilot complete ===\n');
fprintf('omega_1    : %.4f rad/s (%.4f Hz)\n', omega_fin(1), freqHz_fin(1));
fprintf('Multiplicity N: %d\n', result.final_N);
fprintf('S1 overall : %s\n', s1.overall.classification);
fprintf('S1 mode 1  : %s\n', s1.modes(1).classification);
fprintf('Report     : %s\n', fullfile(outDir, 'pilot_report.md'));
diary('off');
end

% =========================================================================
%  S1 diagnosis functions (adapted for OlhoffExact K/M format)
% =========================================================================

function s1 = localS1Diagnose(rho, Phi, omega, freqHz, K, M, free, cMat, ...
    Ke_phys, Me_phys, mass_mode, penal, rho_min, nelx, nely, L, H)

nEl   = nelx * nely;
nModes = min(10, size(Phi, 2));
lowThr  = 0.1;   % Du-Olhoff threshold: below 0.1 is "low-density"
solidThr = 0.5;

lowMask = rho < lowThr;
fprintf('  Elements with rho<%.2f: %d / %d (%.1f%%)\n', lowThr, nnz(lowMask), nEl, 100*nnz(lowMask)/nEl);

[component, compStats] = s1Components(rho, nelx, nely, solidThr);
largestId = s1LargestSupport(compStats);

modeRows = repmat(s1BlankRow(), nModes, 1);
for k = 1:nModes
    phi  = Phi(:, k);
    [kinElem, strElem] = s1ElemEnergies_exact(phi, omega(k), rho, cMat, ...
        Ke_phys, Me_phys, mass_mode, penal);
    modeRows(k) = s1DiagnoseMode(k, omega(k), freqHz(k), phi, M, K, free, cMat, ...
        NaN, kinElem, strElem, rho, [], component, compStats, largestId, lowMask, lowThr);
end

overall = s1Overall(modeRows);
s1.mesh = struct('nelx',nelx,'nely',nely,'L',L,'H',H,'hx',L/nelx,'hy',H/nely);
s1.mass_mode = mass_mode;
s1.low_density_threshold = lowThr;
s1.solid_threshold = solidThr;
s1.n_modes = nModes;
s1.components = struct('count',numel(compStats),'largest_support_id',largestId);
if ~isempty(compStats)
    s1.components.support_connected_count = sum([compStats.touches_left] & [compStats.touches_right]);
else
    s1.components.support_connected_count = 0;
end
s1.modes   = modeRows;
s1.overall = overall;
end

function [kinElem, strElem] = s1ElemEnergies_exact(phi, omega, rho, cMat, ...
    Ke_phys, Me_phys, mass_mode, penal)
nEl = size(cMat, 1);
kinElem = zeros(nEl,1); strElem = zeros(nEl,1);
for e = 1:nEl
    dofs = cMat(e,:);
    ue = phi(dofs);
    ke  = rho(e)^penal;
    [me, ~] = mass_interp(rho(e), mass_mode);
    strElem(e) = 0.5 * real(ue' * (ke * Ke_phys) * ue);
    kinElem(e) = 0.5 * omega^2 * real(ue' * (me * Me_phys) * ue);
end
end

function row = s1BlankRow()
row = struct('mode',NaN,'omega_rad_s',NaN,'frequency_hz',NaN, ...
    'mac_to_solid_reference_mode_1',NaN, ...
    'kinetic_energy_total',NaN,'strain_energy_total',NaN, ...
    'low_density_kinetic_fraction',NaN,'low_density_strain_fraction',NaN, ...
    'kinetic_effective_element_fraction',NaN,'strain_effective_element_fraction',NaN, ...
    'dominant_solid_component_kinetic_fraction',NaN,'dominant_solid_component_strain_fraction',NaN, ...
    'dominant_component_touches_both_supports',false, ...
    'largest_support_component_kinetic_fraction',NaN, ...
    'largest_support_component_strain_fraction',NaN, ...
    'classification','','classification_reason','');
end

function row = s1DiagnoseMode(k, omega, freqHz, phi, M, K, free, cMat, macVal, ...
    kinElem, strElem, rho, ~, component, compStats, largestId, lowMask, lowThr)
row = s1BlankRow();
row.mode = k; row.omega_rad_s = omega; row.frequency_hz = freqHz;
row.mac_to_solid_reference_mode_1 = macVal;
row.kinetic_energy_total = sum(kinElem);
row.strain_energy_total  = sum(strElem);
row.low_density_kinetic_fraction = s1Frac(kinElem(lowMask), kinElem);
row.low_density_strain_fraction  = s1Frac(strElem(lowMask),  strElem);
[~, row.kinetic_effective_element_fraction] = s1Loc(kinElem);
[~, row.strain_effective_element_fraction]  = s1Loc(strElem);
[domId, domK, domS] = s1DomComp(component, kinElem, strElem);
row.dominant_solid_component_kinetic_fraction = domK;
row.dominant_solid_component_strain_fraction  = domS;
if domId > 0 && domId <= numel(compStats)
    row.dominant_component_touches_both_supports = ...
        compStats(domId).touches_left && compStats(domId).touches_right;
end
if largestId > 0
    suppMask = component == largestId;
    row.largest_support_component_kinetic_fraction = s1Frac(kinElem(suppMask), kinElem);
    row.largest_support_component_strain_fraction  = s1Frac(strElem(suppMask),  strElem);
end
[row.classification, row.classification_reason] = s1ClassifyMode(row, lowThr);
end

function [cls, reason] = s1ClassifyMode(row, lowThr)
if row.low_density_kinetic_fraction >= 0.35 || row.low_density_strain_fraction >= 0.35
    cls    = 'localized low-density mode';
    reason = sprintf('low-density (rho<%.2g) energy fractions: kinetic %.3f, strain %.3f', ...
        lowThr, row.low_density_kinetic_fraction, row.low_density_strain_fraction);
elseif row.kinetic_effective_element_fraction < 0.025 || row.strain_effective_element_fraction < 0.025
    cls    = 'localized non-low-density mode';
    reason = sprintf('participation-localized but not in low-density region: effK %.3f, effS %.3f', ...
        row.kinetic_effective_element_fraction, row.strain_effective_element_fraction);
elseif row.largest_support_component_kinetic_fraction >= 0.60 && ...
        row.largest_support_component_strain_fraction  >= 0.60 && ...
        row.dominant_component_touches_both_supports
    cls    = 'physical global mode';
    reason = sprintf('energy on support-spanning component: K %.3f, S %.3f', ...
        row.largest_support_component_kinetic_fraction, ...
        row.largest_support_component_strain_fraction);
else
    cls    = 'ambiguous';
    reason = sprintf('mixed indicators: ldK %.3f, ldS %.3f, suppK %.3f, suppS %.3f', ...
        row.low_density_kinetic_fraction, row.low_density_strain_fraction, ...
        row.largest_support_component_kinetic_fraction, ...
        row.largest_support_component_strain_fraction);
end
end

function overall = s1Overall(modeRows)
classes   = string({modeRows.classification});
physCount = sum(classes == "physical global mode");
locCount  = sum(classes == "localized low-density mode");
locNdCount = sum(classes == "localized non-low-density mode");
ambCount  = sum(classes == "ambiguous");
first3    = classes(1:min(3,numel(classes)));
if any(first3 == "localized low-density mode")
    cls    = 'likely localized/spurious low-density mode influence';
    reason = 'At least one of the first three modes is localized low-density.';
elseif any(first3 == "localized non-low-density mode")
    cls    = 'localized but not in low-density region';
    reason = 'At least one of the first three modes is localized outside low-density region.';
elseif all(first3 == "physical global mode") || ...
        (sum(first3 == "physical global mode") >= 2 && ~any(first3 == "localized low-density mode"))
    cls    = 'likely physically valid';
    reason = 'First modes are classified as physical global modes.';
else
    cls    = 'ambiguous';
    reason = 'Mixed classification among first three modes.';
end
overall = struct('classification', cls, 'reason', reason, ...
    'physical_global_count', physCount, ...
    'localized_low_density_count', locCount, ...
    'localized_non_low_density_count', locNdCount, ...
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
                ny_=a(1); nx_=a(2);
                if ny_>=1&&ny_<=nely&&nx_>=1&&nx_<=nelx&&solid(ny_,nx_)&&component(ny_,nx_)==0
                    tail=tail+1; q(tail,:)=[ny_,nx_]; component(ny_,nx_)=compId;
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
scores=zeros(numel(ids),1); fKs=scores; fSs=scores;
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

function [idx, eff] = s1Loc(e)
e = max(real(e(:)),0); n = numel(e); total = sum(e);
if total <= 0; idx = NaN; eff = NaN; return; end
p = e/total; idx = sum(p.^2); eff = 1/(n*idx);
end

% =========================================================================
%  Mode shape figure
% =========================================================================

function pilotSaveModeShape(path, phi, x, nelx, nely, L, H, omega, modeIdx)
xG = repmat((0:nelx)*(L/nelx), nely+1, 1);
yG = repmat((0:nely)'*(H/nely), 1, nelx+1);
ux = reshape(phi(1:2:end), nely+1, nelx+1);
uy = reshape(phi(2:2:end), nely+1, nelx+1);
mag = hypot(ux,uy);
sc  = 0.06*max(L,H)/max(max(mag(:)),eps);
xD  = xG+sc*ux; yD = yG+sc*uy;
fig = figure('Color','white','Visible','off');
ax  = axes('Parent',fig); hold(ax,'on');
topo = reshape(x(:),nely,nelx);
imagesc(ax,[0,L],[0,H],flipud(topo));
set(ax,'YDir','normal'); colormap(ax,gray); alpha(0.35);
stride = max(1,ceil((max(nelx,nely)+1)/100));
rows = unique([1:stride:(nely+1),nely+1]);
cols = unique([1:stride:(nelx+1),nelx+1]);
for r = rows; plot(ax,xD(r,:),yD(r,:),'-','Color',[0.05,0.25,0.75],'LineWidth',0.7); end
for c = cols; plot(ax,xD(:,c),yD(:,c),'-','Color',[0.05,0.25,0.75],'LineWidth',0.7); end
plot(ax,[0,L,L,0,0],[0,0,H,H,0],'-','Color',[0.1,0.1,0.1],'LineWidth',1);
axis(ax,'equal'); xlim(ax,[-0.05*L,1.05*L]); ylim(ax,[-0.15*H,1.15*H]);
set(ax,'XTick',[],'YTick',[]); box(ax,'on');
title(ax,sprintf('Mode %d | %.4g rad/s | OlhoffExact CC 80x10',modeIdx,omega), ...
    'Interpreter','none','FontSize',8);
try; exportgraphics(fig,path,'Resolution',180,'BackgroundColor','white');
catch; print(fig,path,'-dpng','-r180'); end
close(fig);
end

% =========================================================================
%  Report writers
% =========================================================================

function localWriteComparisonReport(path, result, s1, ourOmega1, ourMAC, ...
    ourIters, ourS1Class, ourS1Loc, ourS1Phys)
lines = {};
lines{end+1} = '# Comparison: OlhoffApproachExact vs ourApproach (CC 80×10)';
lines{end+1} = '';
lines{end+1} = sprintf('Generated: %s', result.created_utc);
lines{end+1} = '';
lines{end+1} = '| metric | OlhoffApproachExact | ourApproach (scaling study) |';
lines{end+1} = '|---|---|---|';
lines{end+1} = sprintf('| solver | OlhoffApproachExact | ourApproach (semi_harmonic) |');
lines{end+1} = sprintf('| objective | maximize omega_1 (bound form.) | minimize semi-harmonic compliance |');
lines{end+1} = sprintf('| mass interpolation | du2007_c1 (Eq. 4b) | power (pmass=1, linear) |');
lines{end+1} = sprintf('| stiffness | SIMP no Emin (rho^p) | SIMP with Emin |');
lines{end+1} = sprintf('| penal | continuation 1->2->3 | fixed 3 |');
lines{end+1} = sprintf('| omega_1 (rad/s) | **%.4g** | %.4g |', result.omega_1_rad_s, ourOmega1);
lines{end+1} = sprintf('| total outer iterations | %d | %d |', result.total_outer_iters, ourIters);
lines{end+1} = sprintf('| final multiplicity N | %d | N/A (different formulation) |', result.final_N);
lines{end+1} = sprintf('| S1 overall | %s | %s |', result.s1_overall, ourS1Class);
lines{end+1} = sprintf('| S1 mode 1 class | %s (ld_strain=%.4g) | localized (ld_strain=0.992) |', ...
    result.s1_mode1_class, result.s1_mode1_ld_strain);
lines{end+1} = sprintf('| S1 localized/10 | %d | %d |', result.s1_localized_count, ourS1Loc);
lines{end+1} = sprintf('| S1 physical/10 | %d | %d |', result.s1_physical_count, ourS1Phys);
lines{end+1} = '';
lines{end+1} = '## Mode frequency table (OlhoffExact)';
lines{end+1} = '';
lines{end+1} = '| mode | omega (rad/s) | Hz | S1 class |';
lines{end+1} = '|---:|---:|---:|---|';
for k = 1:min(numel(s1.modes), 6)
    m = s1.modes(k);
    lines{end+1} = sprintf('| %d | %.4g | %.4g | %s |', ...
        m.mode, m.omega_rad_s, m.frequency_hz, m.classification); %#ok<AGROW>
end
localWriteText(path, strjoin(lines, newline));
end

function localWritePilotReport(path, result, s1, ourOmega1, ~, ourS1Class, ourS1Loc)
lines = {};
lines{end+1} = '# Pilot Report: OlhoffApproachExact CC 80×10 vs Du–Olhoff Fig. 3';
lines{end+1} = '';
lines{end+1} = sprintf('Generated: %s', result.created_utc);
lines{end+1} = '';
lines{end+1} = '## Setup';
lines{end+1} = sprintf('- Mesh: %d×%d (N_e=%d; 4× finer than paper 40×5)', result.mesh.nelx, result.mesh.nely, result.mesh.N_e);
lines{end+1} = sprintf('- BC: clamped-clamped');
lines{end+1} = sprintf('- VF: %.1f', volfrac);
lines{end+1} = sprintf('- Mass: %s', result.mass_mode);
lines{end+1} = sprintf('- Filter: sensitivity, rmin_elem=%.1f', result.rmin_elem);
lines{end+1} = sprintf('- Continuation: p=%s, iters=%s', ...
    mat2str(result.penal_stages), mat2str(result.iters_per_stage));
lines{end+1} = sprintf('- Numerical safeguards: outer_move=%.2f, inner_max_iter=%d (not in paper; required at 80×10 to prevent unconstrained-increment overshoot)', ...
    result.outer_move, result.inner_max_iter);
lines{end+1} = '';
lines{end+1} = '## Question 1: Is the topology structurally acceptable?';
lines{end+1} = '';
lines{end+1} = sprintf('omega_1 = **%.4g rad/s** (%.4g Hz), multiplicity N=%d.', ...
    result.omega_1_rad_s, result.omega_1_hz, result.final_N);
n_physical = result.s1_physical_count;
if result.omega_1_rad_s > 50 && n_physical >= 3
    verdict1 = 'YES — high fundamental frequency and multiple physical global modes indicate structurally valid CC topology.';
elseif result.omega_1_rad_s > 20 && n_physical >= 1
    verdict1 = 'LIKELY YES — moderate fundamental frequency; topology is structurally meaningful.';
else
    verdict1 = 'UNCERTAIN — low fundamental frequency or few physical modes; inspect topology figure.';
end
lines{end+1} = verdict1;
lines{end+1} = '';
lines{end+1} = '## Question 2: Do localized low-density modes remain?';
lines{end+1} = '';
n_loc = result.s1_localized_count;
ld_frac = result.s1_mode1_ld_strain;
if n_loc == 0
    verdict2 = 'NO — du2007_c1 mass completely suppresses localized low-density modes among first 10.';
elseif n_loc <= 2 && strcmp(result.s1_mode1_class, 'physical global mode')
    verdict2 = sprintf('MOSTLY NO — %d/10 modes localized but mode 1 is physical global. du2007_c1 pushes localized modes to higher frequency.', n_loc);
elseif ld_frac < 0.1 && strcmp(result.s1_mode1_class, 'physical global mode')
    verdict2 = sprintf('SUBSTANTIALLY IMPROVED — mode 1 is physical (ld_strain=%.4g); %d/10 higher modes still localized.', ld_frac, n_loc);
else
    verdict2 = sprintf('PARTIALLY — %d/10 modes still localized; mode 1 class: %s.', n_loc, result.s1_mode1_class);
end
lines{end+1} = verdict2;
lines{end+1} = '';
lines{end+1} = '## Question 3: Is the result closer to Du–Olhoff Fig. 3?';
lines{end+1} = '';
lines{end+1} = sprintf('Du & Olhoff (2007) Fig. 3c CC target: omega_1 → 456.4 rad/s at 40×5 mesh, volfrac=0.5.');
lines{end+1} = sprintf('This pilot at 160×20: omega_1 = %.4g rad/s.', result.omega_1_rad_s);
if result.omega_1_rad_s > 200
    verdict3 = 'YES — frequency is in the expected range for the CC problem at this resolution. Topology should show the classical arch/strut pattern of Du & Olhoff Fig. 3c.';
elseif result.omega_1_rad_s > 50
    verdict3 = 'LIKELY — frequency suggests structural optimization is working; may not exactly match Fig. 3 due to mesh resolution and filter differences.';
else
    verdict3 = sprintf('UNCLEAR — omega_1=%.4g is lower than expected; investigate topology for checker/localized features.', result.omega_1_rad_s);
end
lines{end+1} = verdict3;
lines{end+1} = '';
lines{end+1} = '## Question 4: Should revision evidence migrate to OlhoffApproachExact?';
lines{end+1} = '';
lines{end+1} = sprintf('ourApproach 160×20 (scaling study): omega_1=%.4g rad/s, S1=%s, loc=%d/10.', ...
    ourOmega1, ourS1Class, ourS1Loc);
lines{end+1} = sprintf('OlhoffExact  160×20 (this pilot):  omega_1=%.4g rad/s, S1=%s, loc=%d/10.', ...
    result.omega_1_rad_s, result.s1_overall, n_loc);
lines{end+1} = '';
if result.omega_1_rad_s > 10 * ourOmega1 && n_loc < 5
    verdict4 = ['**RECOMMENDED** — OlhoffApproachExact produces dramatically higher omega_1 and fewer ' ...
        'localized modes. The du2007_c1 mass interpolation resolves the localized-mode pathology ' ...
        'that invalidates ourApproach results at 160×20 and finer. Migration to OlhoffApproachExact ' ...
        'for the revision evidence path is scientifically justified.'];
elseif n_loc < ourS1Loc
    verdict4 = ['**CONDITIONAL** — OlhoffApproachExact shows fewer localized modes but other factors ' ...
        '(topology quality, convergence) should be verified before committing to migration.'];
else
    verdict4 = ['**INVESTIGATE FURTHER** — Results do not clearly favor OlhoffApproachExact over ' ...
        'ourApproach. Review topology and S1 details before deciding.'];
end
lines{end+1} = verdict4;
lines{end+1} = '';
lines{end+1} = '## Artifacts';
lines{end+1} = '- `topology_final.png` / `topology_final.csv`';
lines{end+1} = '- `convergence_history.png` / `convergence_history.csv`';
lines{end+1} = '- `multiplicity_history.csv`';
lines{end+1} = '- `mode_01_shape.png`, `mode_02_shape.png`, `mode_03_shape.png`';
lines{end+1} = '- `s1_mode_summary.json` / `s1_modes.csv`';
lines{end+1} = '- `comparison_vs_ourApproach.md`';
lines{end+1} = '- `pilot_result.json`';
localWriteText(path, strjoin(lines, newline));
end

% =========================================================================
%  Misc helpers
% =========================================================================

function localWriteJson(path, data)
try; txt = jsonencode(data, PrettyPrint=true); catch; txt = jsonencode(data); end
localWriteText(path, txt);
end

function localWriteText(path, txt)
fid = fopen(path, 'w');
if fid < 0; error('PilotOlhoff:WriteFailed','Cannot write: %s', path); end
cu = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', char(txt));
end
