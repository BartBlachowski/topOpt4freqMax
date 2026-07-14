function ep = a4_endpoint_eval(endpoint, nModes)
%A4_ENDPOINT_EVAL  Endpoint response variables for A4 (spec V3 §4.1, §4.2).
%
%   ep = A4_ENDPOINT_EVAL(endpoint)          % nModes = 20 (spec default)
%   ep = A4_ENDPOINT_EVAL(endpoint, nModes)
%
%   endpoint = info.a4_endpoint from topopt_freq (runCfg.a4_endpoint_export=true).
%   It carries ALREADY-ASSEMBLED K/M and the FE operators, so this function does
%   NOT re-implement FE assembly and cannot drift from the solver.
%
%   THE SURROGATE MAY NOT JUDGE ITSELF (spec §4).  The primary endpoint is the
%   TRUE fundamental frequency of the tracked physical mode, from an independent
%   exact eigensolve of the final design -- never the surrogate compliance.
%
%   ep.omega1_tracked   PRIMARY. Frequency of the Phi1-type mode: the eigenvector
%                       with max mass-weighted MAC to the solid reference Phi0.
%   ep.mode_index_jstar Index of that mode in the sorted spectrum (B1 detector).
%   ep.mac_to_phi0      Its MAC to Phi0.
%   ep.omega1_min       Lowest eigenvalue, whatever mode it is (the conventional
%                       omega1, reported for honesty).
%   ep.omega1_thresholded  omega1_tracked recomputed on a volume-preserving
%                       thresholded design (gray-material artifact detector).
%   ep.omega1_omega2_gap   Separation of the two lowest modes (coalescence, R2).
%   ep.screen           §4.3.1 mode-admissibility screen on the final design.
%   ep.n_components     Disconnected solid components in the final design.

if nargin < 2 || isempty(nModes), nModes = 20; end

ep = struct();
free = endpoint.free;
K = endpoint.K_final;
M = endpoint.M_final;

[omegas, Phi] = localModes(K(free, free), M(free, free), free, endpoint.ndof, nModes);

ep.omegas = omegas(:);
finite = omegas(isfinite(omegas));
if isempty(finite)
    error('a4_endpoint_eval:NoModes', 'Endpoint eigensolve produced no finite modes.');
end
ep.omega1_min = min(finite);

if numel(finite) >= 2
    s = sort(finite);
    ep.omega1_omega2_gap = s(2) - s(1);
else
    ep.omega1_omega2_gap = NaN;
end

% ---- tracked (Phi1-type) mode: max mass-weighted MAC to the solid Phi0 ----
phi0 = endpoint.phi0_solid;
if isempty(phi0)
    error('a4_endpoint_eval:NoSolidReference', ...
        'phi0_solid is empty: the run did not use a semi_harmonic solid reference.');
end
macs = nan(numel(omegas), 1);
for k = 1:numel(omegas)
    if isfinite(omegas(k))
        macs(k) = a4_mac(Phi(:, k), phi0, M);
    end
end
[bestMac, jstar] = max(macs);
ep.mac_all = macs;
ep.mac_to_phi0 = bestMac;
ep.mode_index_jstar = jstar;
ep.omega1_tracked = omegas(jstar);

% ---- §4.3.1 screen on the final design -----------------------------------
ctx = localScreenCtx(endpoint, M);
ep.screen = a4_mode_screen(Phi, omegas, endpoint.xPhys, ctx, phi0);
ep.n_components = ep.screen.nComponents;

% ---- volume-preserving thresholded design --------------------------------
% Gray material is where the localized modes are reported to live. If
% omega1_tracked ~= omega1_thresholded, the result is not a gray-material
% artifact. If they diverge, it is.
xt = localVolumePreservingThreshold(endpoint.xPhys, endpoint.volfrac);
try
    [Kt, Mt] = localAssembleFrom(endpoint, xt);
    [omT, PhiT] = localModes(Kt(free, free), Mt(free, free), free, endpoint.ndof, nModes);
    macT = nan(numel(omT), 1);
    for k = 1:numel(omT)
        if isfinite(omT(k))
            macT(k) = a4_mac(PhiT(:, k), phi0, Mt);
        end
    end
    [~, jt] = max(macT);
    ep.omega1_thresholded = omT(jt);
catch ME
    ep.omega1_thresholded = NaN;
    ep.omega1_thresholded_error = ME.message;
end

ep.grayness = mean(4 * endpoint.xPhys .* (1 - endpoint.xPhys));
ep.feasibility = abs(mean(endpoint.xPhys) - endpoint.volfrac) / max(endpoint.volfrac, eps);
end

% =========================================================================

function ctx = localScreenCtx(endpoint, M)
ctx = struct( ...
    'nelx', endpoint.nelx, 'nely', endpoint.nely, ...
    'edofMat', endpoint.edofMat, 'KE', endpoint.KE, 'ME', endpoint.ME, ...
    'M', M, 'free', endpoint.free, ...
    'Emax', endpoint.Emax, 'Emin', endpoint.Emin, ...
    'rho0', endpoint.rho0, 'rho_min', endpoint.rho_min, ...
    'penal', endpoint.penal, 'massInterp', endpoint.massInterp);
end

function [K, M] = localAssembleFrom(endpoint, x)
% Rebuild K/M from the SAME element operators the solver used (endpoint.KE /
% endpoint.ME / endpoint.edofMat). No independent FE implementation.
nEl = numel(x);
edof = endpoint.edofMat;
iK = reshape(kron(edof, ones(8, 1))', 64 * nEl, 1);
jK = reshape(kron(edof, ones(1, 8))', 64 * nEl, 1);

sK = reshape(endpoint.KE(:) * (endpoint.Emin + x(:)'.^endpoint.penal * ...
    (endpoint.Emax - endpoint.Emin)), [], 1);
K = sparse(iK, jK, sK, endpoint.ndof, endpoint.ndof);
K = (K + K') / 2;

[rhoPhys, ~] = our_mass_interpolation(x(:), endpoint.rho0, endpoint.rho_min, ...
    endpoint.massInterp.mode, endpoint.massInterp.pmass);
sM = reshape(endpoint.ME(:) * rhoPhys(:)', [], 1);
M = sparse(iK, jK, sM, endpoint.ndof, endpoint.ndof);
M = (M + M') / 2;
end

function xt = localVolumePreservingThreshold(x, volfrac)
% Threshold at the density that preserves the volume fraction exactly.
x = x(:);
xs = sort(x, 'descend');
nKeep = max(1, min(numel(x), round(volfrac * numel(x))));
thr = xs(nKeep);
xt = double(x >= thr);
xt(xt == 0) = 1e-3;   % keep the declared lower bound; avoid singular K
end

function [omegas, Phi] = localModes(Kf, Mf, free, ndof, nModes)
nf = size(Kf, 1);
nModes = min(nModes, max(1, nf - 2));
omegas = nan(nModes, 1);
Phi = zeros(ndof, nModes);
opts.tol = 1e-10;
try
    [V, D] = eigs(Kf, Mf, nModes, 'sm', opts);
catch
    try
        [V, D] = eigs(Kf, Mf, nModes, 1e-6);
    catch ME
        error('a4_endpoint_eval:EigsFailed', 'Endpoint eigensolve failed: %s', ME.message);
    end
end
lam = diag(D);
[lam, idx] = sort(real(lam), 'ascend');
V = V(:, idx);
for k = 1:numel(lam)
    if lam(k) > 0
        omegas(k) = sqrt(lam(k));
    end
    v = zeros(ndof, 1);
    v(free) = real(V(:, k));
    Phi(:, k) = v;
end
end
