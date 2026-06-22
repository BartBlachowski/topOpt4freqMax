%TEST_V1_6_REORDERING V1-6: Mass-weighted MAC and mode reordering regression.
%
% Verifies that MAC-based mode tracking correctly identifies the target
% physical mode regardless of frequency ordering.  The test:
%
%   SETUP
%   Compute 5 eigenmodes of the solid 6×2 mesh (same BCs as gate_a0_fixture).
%   These are the reference modes: phi_1 (lowest), phi_2, ..., phi_5.
%
%   CHECK 1 — MAC self-consistency
%   MAC(phi_i, phi_j, M) = 1 if i==j, ~0 if i≠j (orthogonality).
%
%   CHECK 2 — Reordering robustness
%   Present modes in permuted order: [phi_3, phi_2, phi_5, phi_4, phi_1].
%   Track phi_1 via argmax(MAC): must find index 5 (where phi_1 lives now).
%   Frequency ordering would suggest index 1 is "mode 1" — MAC ignores that.
%
%   CHECK 3 — Cross-design tracking
%   Compute modes from a half-density design (x=0.3 uniform).
%   Track phi_1_solid against the half-density modes.
%   Verify: the best-MAC half-density mode has MAC ≥ 0.5 with phi_1_solid
%   (it may not be exactly 1 because the mode shapes change with density).
%
%   CHECK 4 — Phase and normalization invariance
%   Negate the best-matched mode (-phi) and verify MAC is unchanged (MAC is
%   squared, so it is phase-invariant).
%
% Output: scripts/revision_v1/v1_6_reordering_results.json

scriptDir  = fileparts(mfilename('fullpath'));
repoRoot   = fileparts(fileparts(scriptDir));
fixturePath = fullfile(scriptDir, 'gate_a0_fixture.json');
resultPath  = fullfile(scriptDir, 'v1_6_reordering_results.json');

addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'analysis', 'ourApproach', 'Matlab'));

if ~isfile(fixturePath)
    error('V1_6:MissingFixture', 'Missing fixture: %s', fixturePath);
end

cfg  = jsondecode(fileread(fixturePath));
nelx = double(cfg.domain.mesh.nelx);
nely = double(cfg.domain.mesh.nely);
L    = double(cfg.domain.size.length);
H    = double(cfg.domain.size.height);
E0   = double(cfg.material.E);
nu   = double(cfg.material.nu);
rho0 = double(cfg.material.rho);
Emin = E0 * double(cfg.void_material.E_min_ratio);
rhoMin = double(cfg.void_material.rho_min);
ndof = 2 * (nelx+1) * (nely+1);
hx   = L / nelx;
hy   = H / nely;

KE = localLk(hx, hy, nu);
ME = localLm(hx, hy);
[~, iK, jK] = localEdofData(nelx, nely);
fixed = localFixedDofs(cfg, nelx, nely, L, H);
free  = setdiff((1:ndof)', fixed);

nModes = 5;

% -----------------------------------------------------------------
% Solid domain: K_solid, M_solid, eigenmodes
% -----------------------------------------------------------------
x_solid = ones(nelx*nely, 1);
[K_solid, M_solid] = localAssemble(x_solid, KE, ME, iK, jK, E0, Emin, rho0, rhoMin, ndof);
Kf_s = K_solid(free, free);
Mf_s = M_solid(free, free);
[V_s, D_s] = eigs(Kf_s, Mf_s, nModes, 1e-6);
omegas_solid = sqrt(abs(diag(D_s)));
[omegas_solid, idx_sort] = sort(omegas_solid);
V_s = V_s(:, idx_sort);
phi_solid = zeros(ndof, nModes);
phi_solid(free, :) = V_s;
phi_solid = mass_normalize_modes(phi_solid, M_solid);
phi_solid = orient_modes_deterministic(phi_solid);
fprintf('[V1-6] Solid modes: omega = %s rad/s\n', mat2str(round(omegas_solid', 3)));

% -----------------------------------------------------------------
% CHECK 1: MAC self-consistency
% -----------------------------------------------------------------
macMatrix = squared_mass_weighted_mac(phi_solid, phi_solid, M_solid);
diagMAC    = diag(macMatrix);
offDiagMAC = max(abs(macMatrix - diag(diagMAC)), [], 'all');
if any(abs(diagMAC - 1) > 1e-10)
    error('V1_6:MACDiagFailed', 'MAC(phi_i, phi_i) deviates from 1: max err = %.3e.', ...
        max(abs(diagMAC - 1)));
end
if offDiagMAC > 1e-10
    error('V1_6:MACOffDiagFailed', 'MAC(phi_i, phi_j) for i≠j is %.3e (expected ~0).', offDiagMAC);
end
fprintf('[V1-6] CHECK 1 PASS: MAC diagonal = 1, off-diagonal max = %.3e.\n', offDiagMAC);

% -----------------------------------------------------------------
% CHECK 2: Reordering robustness
% -----------------------------------------------------------------
permOrder = [3, 2, 5, 4, 1];  % phi_1 is now at index 5
phi_permuted = phi_solid(:, permOrder);
mac_track = squared_mass_weighted_mac(phi_solid(:,1), phi_permuted, M_solid);
mac_track  = mac_track(1, :);  % 1 x nModes
[best_mac, best_idx] = max(mac_track);
if best_idx ~= 5
    error('V1_6:ReorderingFailed', ...
        'Mode tracker selected index %d (expected 5 where phi_1 lives in permuted set). MAC values: %s', ...
        best_idx, mat2str(mac_track, 4));
end
if abs(best_mac - 1) > 1e-10
    error('V1_6:ReorderingMAC', 'Best MAC after reordering = %.6f (expected 1).', best_mac);
end
fprintf('[V1-6] CHECK 2 PASS: phi_1 found at permuted index %d with MAC = %.6f.\n', best_idx, best_mac);

% -----------------------------------------------------------------
% CHECK 3: Cross-design tracking (solid phi_1 vs half-density modes)
% -----------------------------------------------------------------
x_half = 0.3 * ones(nelx*nely, 1);
[K_half, M_half] = localAssemble(x_half, KE, ME, iK, jK, E0, Emin, rho0, rhoMin, ndof);
Kf_h = K_half(free, free);
Mf_h = M_half(free, free);
[V_h, D_h] = eigs(Kf_h, Mf_h, nModes, 1e-6);
omegas_half = sqrt(abs(diag(D_h)));
[omegas_half, idx_h] = sort(omegas_half);
V_h = V_h(:, idx_h);
phi_half = zeros(ndof, nModes);
phi_half(free, :) = V_h;
phi_half = mass_normalize_modes(phi_half, M_half);
phi_half = orient_modes_deterministic(phi_half);
fprintf('[V1-6] Half-density modes: omega = %s rad/s\n', mat2str(round(omegas_half', 3)));

mac_cross = squared_mass_weighted_mac(phi_solid(:,1), phi_half, M_solid);
mac_cross  = mac_cross(1, :);
[best_cross_mac, best_cross_idx] = max(mac_cross);
crossMacThreshold = 0.5;
if best_cross_mac < crossMacThreshold
    error('V1_6:CrossDesignMAC', ...
        'Best MAC between solid phi_1 and half-density modes = %.3f (threshold %.2f).', ...
        best_cross_mac, crossMacThreshold);
end
fprintf('[V1-6] CHECK 3 PASS: best cross-design MAC = %.4f at half-density mode %d.\n', ...
    best_cross_mac, best_cross_idx);

% -----------------------------------------------------------------
% CHECK 4: Phase-invariance of MAC (MAC is squared -> sign-invariant)
% -----------------------------------------------------------------
phi_negated = phi_solid;
phi_negated(:, 1) = -phi_solid(:, 1);  % negate only mode 1
mac_pos = squared_mass_weighted_mac(phi_solid(:,1),  phi_solid(:,1), M_solid);
mac_neg = squared_mass_weighted_mac(phi_solid(:,1), -phi_solid(:,1), M_solid);
if abs(mac_pos - mac_neg) > 1e-12
    error('V1_6:PhaseInvariance', ...
        'MAC(phi, phi) = %.6f but MAC(phi, -phi) = %.6f — MAC is not phase-invariant.', ...
        mac_pos, mac_neg);
end
if abs(mac_pos - 1) > 1e-10
    error('V1_6:PhaseInvarianceMAC', 'MAC(phi, phi) = %.6f (expected 1).', mac_pos);
end
fprintf('[V1-6] CHECK 4 PASS: MAC is phase-invariant (MAC=%.6f = MAC(-phi)=%.6f).\n', ...
    mac_pos, mac_neg);

% -----------------------------------------------------------------
% Collect and write result.
% -----------------------------------------------------------------
result = struct();
result.gate    = 'V1-6';
result.status  = 'passed';
result.fixture = 'scripts/revision_v1/gate_a0_fixture.json (6x2 mesh, inline FE)';
result.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
result.matlab_version = version;
result.n_modes = nModes;
result.solid_omegas = omegas_solid(:);
result.half_density_omegas = omegas_half(:);

result.check_1_mac_self_consistency = struct( ...
    'passed', true, ...
    'mac_diagonal_min', min(diagMAC), ...
    'mac_off_diagonal_max', offDiagMAC);

result.check_2_reordering_robustness = struct( ...
    'passed', true, ...
    'permutation', permOrder, ...
    'phi1_expected_at_index', 5, ...
    'phi1_found_at_index', best_idx, ...
    'best_mac', best_mac, ...
    'all_mac_values', mac_track(:));

result.check_3_cross_design_tracking = struct( ...
    'passed', true, ...
    'best_mac', best_cross_mac, ...
    'best_mode_index', best_cross_idx, ...
    'threshold', crossMacThreshold, ...
    'all_mac_values', mac_cross(:));

result.check_4_phase_invariance = struct( ...
    'passed', true, ...
    'mac_phi_phi', mac_pos, ...
    'mac_phi_neg_phi', mac_neg);

fid = fopen(resultPath, 'w');
if fid < 0
    error('V1_6:ResultWrite', 'Unable to create result file: %s', resultPath);
end
cleanupFid = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(result, PrettyPrint=true));

fprintf('\nV1-6 PASSED\n');
fprintf('  MAC self-consistency: diagonal min = %.6f, off-diag max = %.3e.\n', ...
    min(diagMAC), offDiagMAC);
fprintf('  Reordering: phi_1 found at correct index %d (MAC = %.6f).\n', best_idx, best_mac);
fprintf('  Cross-design: best MAC = %.4f (threshold %.2f).\n', best_cross_mac, crossMacThreshold);
fprintf('  Phase invariance: MAC(phi, -phi) = %.6f = MAC(phi, phi).\n', mac_neg);
fprintf('  Saved: %s\n', resultPath);

% =========================================================================
function [K, M] = localAssemble(x, KE, ME, iK, jK, E0, Emin, rho0, rhoMin, ndof)
penal = 3.0; pmass = 1.0;
sK = reshape(KE(:) * (Emin + x'.^penal * (E0 - Emin)), [], 1);
sM = reshape(ME(:) * (rhoMin + x'.^pmass * (rho0 - rhoMin)), [], 1);
K  = sparse(iK, jK, sK, ndof, ndof); K = (K + K') / 2;
M  = sparse(iK, jK, sM, ndof, ndof); M = (M + M') / 2;
end

function [edofMat, iK, jK] = localEdofData(nelx, nely)
edofMat = zeros(nelx * nely, 8);
for elx = 0:nelx-1, for ely = 0:nely-1
    el = ely + elx*nely + 1;
    n1 = (nely+1)*elx + ely;  n2 = (nely+1)*(elx+1) + ely;
    edofMat(el,:) = [2*n1+1, 2*n1+2, 2*n2+1, 2*n2+2, ...
        2*(n2+1)+1, 2*(n2+1)+2, 2*(n1+1)+1, 2*(n1+1)+2];
end, end
iK = reshape(kron(edofMat, ones(1,8))', [], 1);
jK = reshape(kron(edofMat, ones(8,1))', [], 1);
end

function fixed = localFixedDofs(cfg, nelx, nely, L, H)
[X, Y] = meshgrid(linspace(0, L, nelx+1), linspace(0, H, nely+1));
nodeX = X(:); nodeY = Y(:); fixed = [];
supports = cfg.bc.supports;
for i = 1:numel(supports)
    if iscell(supports), sup = supports{i}; else, sup = supports(i); end
    switch char(sup.type)
        case 'vertical_line'
            tol = 1e-9; if isfield(sup,'tol'), tol=double(sup.tol); end
            nodes = find(abs(nodeX - double(sup.x)) <= tol);
        case 'closest_point'
            loc = double(sup.location(:));
            [~, node] = min((nodeX-loc(1)).^2+(nodeY-loc(2)).^2); nodes = node;
        otherwise
            error('V1_6:UnsupportedSupport','%s',char(sup.type));
    end
    dofs = cellstr(string(sup.dofs));
    if any(strcmp(dofs,'ux')), fixed=[fixed;2*nodes-1]; end %#ok<AGROW>
    if any(strcmp(dofs,'uy')), fixed=[fixed;2*nodes];   end %#ok<AGROW>
end
fixed = unique(fixed);
if isempty(fixed), error('V1_6:NoFixedDofs','No fixed DOFs.'); end
end

function KE = localLk(hx, hy, nu)
D=(1/(1-nu^2))*[1,nu,0;nu,1,0;0,0,(1-nu)/2];
invJ=[2/hx,0;0,2/hy]; detJ=hx*hy/4; gp=1/sqrt(3);
KE=zeros(8,8);
for xi=[-gp,gp], for eta=[-gp,gp]
    dNdxi=0.25*[-(1-eta),(1-eta),(1+eta),-(1+eta)];
    dNdeta=0.25*[-(1-xi),-(1+xi),(1+xi),(1-xi)];
    dNxy=invJ*[dNdxi;dNdeta];
    B=zeros(3,8); B(1,1:2:end)=dNxy(1,:); B(2,2:2:end)=dNxy(2,:);
    B(3,1:2:end)=dNxy(2,:); B(3,2:2:end)=dNxy(1,:);
    KE=KE+B'*D*B*detJ;
end, end
end

function ME = localLm(hx, hy)
Ms=(hx*hy/36)*[4,2,1,2;2,4,2,1;1,2,4,2;2,1,2,4];
ME=kron(Ms,eye(2));
end
