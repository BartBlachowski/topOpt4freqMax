function postprocess_final_modes()
% POSTPROCESS_FINAL_MODES  Spectral-validity evidence for Gate G4.
%
%   For the FINAL design of every completed run, recompute the lowest modes and
%   measure what fraction of each mode's strain energy sits in low-density
%   (rho <= 0.1) elements.  A "reported objective corresponding to the actual
%   lowest eigenvalue cluster" (G4) requires mode 1 to be a genuine global
%   structural mode, not a spurious void-localized artefact.
%
%   The measure is identical to the one used in the mesh-resolution campaign:
%       ld_strain(j) = sum_{e: rho_e <= 0.1} phi_j,e' K_e phi_j,e
%                      / sum_e            phi_j,e' K_e phi_j,e
%
%   Runs AFTER the optimization on saved designs; changes nothing.
%   Writes results/<tag>/localization.json.

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);
addpath(fullfile(this_dir, '..', '..', 'Matlab'));
addpath(fullfile(this_dir, '..', '..', '..', '..', 'tools', 'Matlab'));

res = fullfile(this_dir, 'results');
dd = dir(res);
dd = dd([dd.isdir] & ~ismember({dd.name}, {'.','..','_logs'}));

for k = 1:numel(dd)
    d = fullfile(res, dd(k).name);
    sj = fullfile(d, 'summary.json');
    rf = fullfile(d, 'rho_final.csv');
    if ~exist(sj,'file') || ~exist(rf,'file'), continue, end
    S = jsondecode(fileread(sj));
    G = readmatrix(rf);
    nely = size(G,1);  nelx = size(G,2);
    rho = G(:);

    M = setup(nelx, nely, S.bc);
    pp = 3.0;
    [om, Phi, lam] = modes(rho, M, pp, 'du2007_c1', 4);
    ld = zeros(1,4);
    lowmask = rho <= 0.1;
    for j = 1:4
        pe  = Phi(:,j);
        pev = pe(M.cMat);                             % nEl x 8
        se  = sum((pev * M.Ke_phys) .* pev, 2) .* (rho .^ pp);
        tot = sum(se);
        if tot > 0, ld(j) = sum(se(lowmask))/tot; else, ld(j) = NaN; end
    end
    o = struct();
    o.tag = dd(k).name;
    o.omega_p3 = om(:)';
    o.lambda_p3 = lam(:)';
    o.mode1_local_frac = ld(1);
    o.mode_local_frac = ld;
    o.low_density_area_frac = mean(lowmask);
    o.g12 = abs(om(2)-om(1))/max(om(1), eps);
    fid = fopen(fullfile(d,'localization.json'),'w');
    fprintf(fid, '%s\n', jsonencode(o, 'PrettyPrint', true));
    fclose(fid);
    fprintf(' %-30s omega1=%9.4f  g12=%.3e  ld_strain(mode1)=%.4e  modes2-4=[%.3e %.3e %.3e]\n', ...
        dd(k).name, om(1), o.g12, ld(1), ld(2), ld(3), ld(4));
end
end

%% ---------------------------------------------------------------------
function M = setup(nelx, nely, bc)
L=8; H=1; E0=1e7; nu=0.3; rho0=1; t=1;
dx=L/nelx; dy=H/nely;
M.nEl=nelx*nely; M.nDof=2*(nelx+1)*(nely+1);
[Ks, Ms] = fe_q4_exact(nu,t,dx,dy);
M.Ke_phys = E0*Ks;  M.Me_phys = rho0*Ms;
nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
cVec = reshape(2*nodeNrs(1:nely,1:nelx)+1, M.nEl, 1);
M.cMat = [cVec, cVec+1, cVec+2*nely+2, cVec+2*nely+3, ...
          cVec+2*nely, cVec+2*nely+1, cVec-2, cVec-1];
[Il,Jl] = find(tril(ones(8)));
M.iK = reshape(M.cMat(:,Il)',[],1);
M.jK = reshape(M.cMat(:,Jl)',[],1);
M.Ke_l = M.Ke_phys(sub2ind([8,8],Il,Jl));
M.Me_l = M.Me_phys(sub2ind([8,8],Il,Jl));
fixed = build_supports_exact(bc, nodeNrs);
M.free = setdiff(1:M.nDof, fixed);
M.opts.tol=1e-10; M.opts.maxit=600;
end

%% ---------------------------------------------------------------------
function [omega, Phi, lam] = modes(rho, M, penal, mass_mode, nm)
[K, Mm] = assemble_KM_exact(rho, M.Ke_l, M.Me_l, M.iK, M.jK, M.nDof, penal, mass_mode);
Kf = K(M.free,M.free);  Mf = Mm(M.free,M.free);
[V,D,fl] = eigs(Kf, Mf, nm, 'SM', M.opts);
if fl ~= 0
    o.tol=1e-8; o.maxit=1500; o.p=min(numel(M.free)-1, max(40,4*nm));
    [V,D,fl] = eigs(Kf, Mf, nm, 'SM', o);
end
[lam, ix] = sort(real(diag(D)));  V = real(V(:,ix));
for j=1:nm
    v=V(:,j); sc=sqrt(abs(v'*(Mf*v))); if sc>1e-14, V(:,j)=v/sc; end
end
omega = sqrt(max(lam,0));
Phi = zeros(M.nDof, nm);
for j=1:nm, Phi(M.free,j)=V(:,j); end
end
