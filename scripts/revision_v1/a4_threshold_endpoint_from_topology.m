function ep = a4_threshold_endpoint_from_topology(x, cfg, nModes)
%A4_THRESHOLD_ENDPOINT_FROM_TOPOLOGY  Recompute only A4's threshold endpoint.
%
%   Recovery Phase 1 helper. It assembles the thresholded and solid-reference
%   eigenproblems from an existing stored topology. It does not run optimization
%   and does not recompute or replace the original tracked endpoint.

if nargin < 3 || isempty(nModes), nModes = 20; end

repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repoRoot, 'analysis', 'ourApproach', 'Matlab'));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));

nelx = double(cfg.domain.mesh.nelx);
nely = double(cfg.domain.mesh.nely);
L = double(cfg.domain.size.length);
H = double(cfg.domain.size.height);
nu = double(cfg.material.nu);
Emax = double(cfg.material.E);
Emin = Emax * double(cfg.void_material.E_min_ratio);
rho0 = double(cfg.material.rho);
rhoMin = double(cfg.void_material.rho_min);
penal = double(cfg.optimization.penalization);
volfrac = double(cfg.optimization.volume_fraction);
pmass = double(cfg.optimization.pmass);

x = double(x(:));
if numel(x) ~= nelx*nely
    error('a4_threshold_endpoint_from_topology:TopologySize', ...
        'Expected %d topology values, got %d.', nelx*nely, numel(x));
end

hx = L / nelx;
hy = H / nely;
ndof = 2*(nelx+1)*(nely+1);
KE = localKE(hx, hy, nu);
ME = localME(hx, hy);

edofMat = zeros(nelx*nely, 8);
for elx = 0:nelx-1
    for ely = 0:nely-1
        el = ely + elx*nely + 1;
        n1 = (nely+1)*elx + ely;
        n2 = (nely+1)*(elx+1) + ely;
        n3 = n2 + 1;
        n4 = n1 + 1;
        edofMat(el,:) = [2*n1+1,2*n1+2,2*n2+1,2*n2+2, ...
            2*n3+1,2*n3+2,2*n4+1,2*n4+2];
    end
end
iK = reshape(kron(edofMat, ones(1,8))', [], 1);
jK = reshape(kron(edofMat, ones(8,1))', [], 1);

supports = cfg.bc.supports;
for k = 1:numel(supports)
    t = lower(strtrim(char(supports(k).type)));
    if any(strcmp(t, {'hinge','clamp'}))
        error('a4_threshold_endpoint_from_topology:UnsupportedSupport', ...
            'Recovery helper expects A4 closest_point supports, not %s.', t);
    end
end
fixed = supportsToFixedDofs(supports, nelx, nely, L, H);
free = setdiff(1:ndof, fixed(:)');

massInterp = struct('mode', 'power', 'pmass', pmass);
[Ksolid, Msolid] = localAssemble(ones(size(x)), KE, ME, iK, jK, ndof, ...
    Emax, Emin, rho0, rhoMin, penal, massInterp);
[~, phiSolid] = localModes(Ksolid(free,free), Msolid(free,free), free, ndof, 1);

xt = a4_volume_preserving_threshold(x, volfrac, rhoMin);
[Kt, Mt] = localAssemble(xt, KE, ME, iK, jK, ndof, ...
    Emax, Emin, rho0, rhoMin, penal, massInterp);
[omegas, Phi] = localModes(Kt(free,free), Mt(free,free), free, ndof, nModes);

macs = nan(numel(omegas),1);
for k = 1:numel(omegas)
    if isfinite(omegas(k))
        macs(k) = a4_mac(Phi(:,k), phiSolid(:,1), Mt);
    end
end
[bestMac, jstar] = max(macs);

xs = sort(x, 'descend');
nKeep = max(1, min(numel(x), round(volfrac*numel(x))));
ep = struct( ...
    'omega1_thresholded', omegas(jstar), ...
    'mode_index', jstar, ...
    'mac_to_solid', bestMac, ...
    'configured_floor', rhoMin, ...
    'threshold_value', xs(nKeep), ...
    'n_solid', nnz(xt == 1), ...
    'n_floor', nnz(xt == rhoMin));
end

function [K,M] = localAssemble(x, KE, ME, iK, jK, ndof, ...
        Emax, Emin, rho0, rhoMin, penal, massInterp)
nEl = numel(x);
sK = reshape(KE(:)*(Emin + x(:)'.^penal*(Emax-Emin)), 64*nEl, 1);
[rho,~] = our_mass_interpolation(x(:), rho0, rhoMin, ...
    massInterp.mode, massInterp.pmass);
sM = reshape(ME(:)*rho(:)', 64*nEl, 1);
K = sparse(iK,jK,sK,ndof,ndof); K = (K+K')/2;
M = sparse(iK,jK,sM,ndof,ndof); M = (M+M')/2;
end

function [omegas,Phi] = localModes(Kf,Mf,free,ndof,nModes)
nf = size(Kf,1);
nModes = min(nModes,max(1,nf-2));
opts.tol = 1e-10;
try
    [V,D] = eigs(Kf,Mf,nModes,'sm',opts);
catch
    [V,D] = eigs(Kf,Mf,nModes,1e-6);
end
[lam,idx] = sort(real(diag(D)),'ascend');
V = V(:,idx);
omegas = nan(nModes,1);
Phi = zeros(ndof,nModes);
for k = 1:numel(lam)
    if lam(k) > 0, omegas(k) = sqrt(lam(k)); end
    v = zeros(ndof,1); v(free) = real(V(:,k)); Phi(:,k) = v;
end
end

function KE = localKE(hx,hy,nu)
D = (1/(1-nu^2))*[1,nu,0;nu,1,0;0,0,0.5*(1-nu)];
invJ = [2/hx,0;0,2/hy]; detJ = 0.25*hx*hy;
gp = 1/sqrt(3); pts = [-gp,gp]; KE = zeros(8,8);
for xi = pts
    for eta = pts
        q = invJ*[0.25*[-(1-eta),(1-eta),(1+eta),-(1+eta)]; ...
            0.25*[-(1-xi),-(1+xi),(1+xi),(1-xi)]];
        B = zeros(3,8);
        B(1,1:2:end)=q(1,:); B(2,2:2:end)=q(2,:);
        B(3,1:2:end)=q(2,:); B(3,2:2:end)=q(1,:);
        KE = KE + B'*D*B*detJ;
    end
end
end

function ME = localME(hx,hy)
Ms = (hx*hy/36)*[4,2,1,2;2,4,2,1;1,2,4,2;2,1,2,4];
ME = kron(Ms,eye(2));
end
