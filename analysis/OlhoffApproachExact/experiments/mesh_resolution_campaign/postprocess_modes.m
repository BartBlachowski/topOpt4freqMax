function postprocess_modes(tag)
% POSTPROCESS_MODES  Post-hoc modal instrumentation for one campaign run.
%
%   postprocess_modes(tag)
%
% MEASUREMENT ONLY.  Re-evaluates the eigenproblem on the stored per-iteration
% design snapshots and derives quantities the solver does not itself record:
%
%   - MAC history                between consecutive outer iterations
%   - tracked mode identity      which mode index at iter k continues mode 1
%                                of iter k-1 (max-MAC assignment)
%   - mode-1 tracking loss       1 - MAC(mode1_{k-1}, tracked mode_k)
%   - local-mode dominance       fraction of modal strain energy residing in
%                                low-density elements (rho <= 0.1), the
%                                ld_strain_frac metric
%
% It reuses the production FE routines (assemble_KM_exact, fe_q4_exact,
% build_supports_exact) with the same eigensolver options as the solver.  It
% does not call, and cannot affect, the optimizer: it runs after the fact on
% saved data.
%
% Output: results/<tag>/modes.csv, results/<tag>/mac_history.mat

this_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(this_dir, '..', '..', 'Matlab'));
addpath(fullfile(this_dir, '..', '..', '..', '..', 'tools', 'Matlab'));

run_dir = fullfile(this_dir, 'results', tag);
S = load(fullfile(run_dir, 'run.mat'));
cfg = S.cfg; hist = S.hist;

nelx = cfg.nelx; nely = cfg.nely;
L = 8.0; H = 1.0;
E0 = 1e7; nu = 0.3; rho0 = 1.0; t = 1.0;
penal = 3.0; mass_mode = cfg.mass_mode;
n_modes = cfg.n_modes;
rho_min = 1e-3;

dx = L/nelx; dy = H/nely;
nEl = nelx*nely;
nDof = 2*(nelx+1)*(nely+1);

[Ke_star, Me_star] = fe_q4_exact(nu, t, dx, dy);
Ke_phys = E0*Ke_star;  Me_phys = rho0*Me_star;

nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
cVec = reshape(2*nodeNrs(1:nely,1:nelx)+1, nEl, 1);
cMat = [cVec, cVec+1, cVec+2*nely+2, cVec+2*nely+3, ...
        cVec+2*nely, cVec+2*nely+1, cVec-2, cVec-1];
[Il, Jl] = find(tril(ones(8)));
iK = reshape(cMat(:,Il)', [], 1);
jK = reshape(cMat(:,Jl)', [], 1);
Ke_phys_l = Ke_phys(sub2ind([8,8], Il, Jl));
Me_phys_l = Me_phys(sub2ind([8,8], Il, Jl));

fixed = build_supports_exact(cfg.support_type, nodeNrs);
free  = setdiff(1:nDof, fixed);

opts_eig.tol = 1e-10; opts_eig.maxit = 600;

ns   = hist.rho_snapshot_count;
snap = hist.rho_snapshots(:, 1:ns);
iters = hist.rho_snapshot_iters(1:ns);

omega_s  = nan(ns, n_modes);
ld_frac  = nan(ns, n_modes);
mac_diag = nan(ns, n_modes);   % MAC of mode j at k vs its tracked partner at k-1
track_id = nan(ns, 1);         % index at k of the mode continuing mode 1 at k-1
track_mac = nan(ns, 1);
mac_full = cell(ns, 1);

Phi_prev = [];
fprintf('postprocess_modes(%s): %d snapshots, mesh %dx%d\n', tag, ns, nelx, nely);

for k = 1:ns
    rho = max(rho_min, min(1, snap(:, k)));
    [K, M] = assemble_KM_exact(rho, Ke_phys_l, Me_phys_l, iK, jK, nDof, penal, mass_mode);
    Kf = K(free, free);  Mf = M(free, free);

    [V, D, flag] = eigs(Kf, Mf, n_modes, 'SM', opts_eig);
    if flag ~= 0
        o.tol = 1e-8; o.maxit = 1500; o.p = min(numel(free)-1, max(40, 4*n_modes));
        [V, D, flag] = eigs(Kf, Mf, n_modes, 'SM', o);
    end
    if flag ~= 0, continue; end

    [lam, idx] = sort(real(diag(D)));
    V = real(V(:, idx));
    for j = 1:n_modes
        sc = sqrt(abs(V(:,j)' * (Mf * V(:,j))));
        if sc > 1e-14, V(:,j) = V(:,j)/sc; end
    end
    omega_s(k, :) = sqrt(max(lam, 0))';

    Phi = zeros(nDof, n_modes);
    Phi(free, :) = V;

    % --- local-mode dominance: modal strain energy in low-density elements ---
    Esimp = rho_min + (1 - rho_min) * rho.^penal;   % assemble_KM_exact SIMP law
    low = rho <= 0.1;
    for j = 1:n_modes
        Ue = reshape(Phi(cMat(:), j), nEl, 8);   % element DOF vectors, nEl x 8
        se = Esimp .* sum((Ue * Ke_phys) .* Ue, 2);
        tot = sum(se);
        if tot > 0, ld_frac(k, j) = sum(se(low)) / tot; end
    end

    % --- MAC vs previous iteration (mass-normalised, current M) ---
    if ~isempty(Phi_prev)
        num = (Phi_prev' * (M * Phi)).^2;
        mp  = real(sum(Phi_prev .* (M * Phi_prev), 1))';
        mc  = real(sum(Phi     .* (M * Phi    ), 1));
        mac = num ./ max(mp * mc, eps);
        mac = min(max(mac, 0), 1);
        mac_full{k} = mac;
        [track_mac(k), track_id(k)] = max(mac(1, :));   % partner of previous mode 1
        for j = 1:n_modes
            mac_diag(k, j) = max(mac(j, :));
        end
    end
    Phi_prev = Phi;
end

fid = fopen(fullfile(run_dir, 'modes.csv'), 'w');
hdr = 'iter,track_id,track_mac,mode1_tracking_loss';
for j = 1:n_modes, hdr = [hdr sprintf(',omega_%d', j)]; end
for j = 1:n_modes, hdr = [hdr sprintf(',ld_strain_frac_%d', j)]; end
for j = 1:n_modes, hdr = [hdr sprintf(',mac_best_%d', j)]; end
fprintf(fid, '%s\n', hdr);
for k = 1:ns
    fprintf(fid, '%d,%g,%.10g,%.10g', iters(k), track_id(k), track_mac(k), 1 - track_mac(k));
    for j = 1:n_modes, fprintf(fid, ',%.10g', omega_s(k, j)); end
    for j = 1:n_modes, fprintf(fid, ',%.10g', ld_frac(k, j)); end
    for j = 1:n_modes, fprintf(fid, ',%.10g', mac_diag(k, j)); end
    fprintf(fid, '\n');
end
fclose(fid);

save(fullfile(run_dir, 'mac_history.mat'), 'mac_full', 'iters', 'omega_s', ...
    'ld_frac', 'track_id', 'track_mac', '-v7.3');
fprintf('  -> wrote %s\n', fullfile(run_dir, 'modes.csv'));
end
