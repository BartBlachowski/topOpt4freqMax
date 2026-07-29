function out = recon_solve(cfg)
% RECON_SOLVE  Instrumented Du & Olhoff (2007) outer loop for the faithful
%              reconstruction campaign.
%
%   out = recon_solve(cfg)
%
%   PROVENANCE.  This function re-implements ONLY the outer loop of the
%   production solver analysis/OlhoffApproachExact/Matlab/topopt_freq_exact.m,
%   restricted to its DEFAULT execution path (globalization, forensics,
%   density/projection filters, forced-solid masks, density symmetry and
%   persistent-MMA are all default-disabled in production and are simply
%   absent here).  Every numerical kernel is the PRODUCTION file, called
%   unmodified:
%       fe_q4_exact, build_supports_exact, build_filter, assemble_KM_exact,
%       detect_multiplicity, compute_generalized_gradients,
%       compute_elem_sensitivity, apply_sensitivity_filter, mass_interp
%   The inner MMA is inner_loop_mma_instr.m, an instrumented copy of the
%   production inner_loop_mma.m proven bit-identical by
%   tests/test_inner_equivalence.m.
%
%   With cfg.cont.enabled = false and cfg.fail_closed = false this function
%   reproduces topopt_freq_exact bit-identically; proven by
%   tests/test_outer_equivalence.m.
%
%   NO PRODUCTION FILE IS MODIFIED BY THIS CAMPAIGN.
%
%   ---------------- added, campaign-specific cfg fields ----------------
%     cfg.cont.enabled     logical, SIMP penalization continuation (default false)
%     cfg.cont.p_values    vector of p values, e.g. [1 1.5 2 2.5 3]
%     cfg.cont.mode        'fixed'  -> advance every cfg.cont.stage_len iters
%                          'drho'   -> advance when drho_norm < drho_trigger
%                                      and the stage has run min_stage_len iters
%     cfg.cont.stage_len   outer iterations per stage ('fixed' mode)
%     cfg.cont.drho_trigger  stage-advance threshold ('drho' mode)
%     cfg.cont.min_stage_len minimum iterations before a 'drho' advance
%     cfg.fail_closed      logical, fail-closed inner-MMA acceptance (default false)
%     cfg.fc_vol_tol       absolute tolerance on mean(rho+drho) - volfrac (1e-4)
%     cfg.fc_bound_tol     absolute tolerance on box-bound violation (1e-9)
%     cfg.tag              run label used for output files
%
%   Reference: Du & Olhoff (2007), Struct Multidisc Optim 34:91-110.

cfg = recon_defaults(cfg);

L = cfg.L; H = cfg.H; nelx = cfg.nelx; nely = cfg.nely;
E0 = cfg.E0; nu = cfg.nu; rho0 = cfg.rho0; t = cfg.t;
volfrac = cfg.volfrac; rho_min = cfg.rho_min;
mass_mode = cfg.mass_mode; rmin_elem = cfg.rmin_elem;
n_target = cfg.n_target; n_modes = cfg.n_modes; mult_tol = cfg.mult_tol;
outer_max_iter = cfg.outer_max_iter; outer_tol = cfg.outer_tol;
inner_max_iter = cfg.inner_max_iter; inner_tol = cfg.inner_tol;
move_lim = cfg.move_lim; outer_move = cfg.outer_move; alpha = cfg.alpha;
verbose = cfg.verbose;

%% ---------------- mesh / FE setup (identical to production) -------------
dx = L/nelx;  dy = H/nely;
nEl  = nelx * nely;
nDof = 2 * (nelx+1) * (nely+1);

[Ke_star, Me_star] = fe_q4_exact(nu, t, dx, dy);
Ke_phys = E0   * Ke_star;
Me_phys = rho0 * Me_star;

nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
cVec    = reshape(2*nodeNrs(1:nely,1:nelx)+1, nEl, 1);
cMat    = [cVec, cVec+1, cVec+2*nely+2, cVec+2*nely+3, ...
           cVec+2*nely, cVec+2*nely+1, cVec-2, cVec-1];

[Il, Jl] = find(tril(ones(8)));
iK = reshape(cMat(:,Il)', [], 1);
jK = reshape(cMat(:,Jl)', [], 1);
Ke_phys_l = Ke_phys(sub2ind([8,8], Il, Jl));
Me_phys_l = Me_phys(sub2ind([8,8], Il, Jl));

fixed = build_supports_exact(cfg.support_type, nodeNrs);
free  = setdiff(1:nDof, fixed);
nFree = numel(free);

[h_filt, Hs_filt] = build_filter(nelx, nely, rmin_elem);

%% ---------------- initial design ---------------------------------------
if isfield(cfg,'initial_rho') && ~isempty(cfg.initial_rho)
    rho = max(rho_min, min(1, cfg.initial_rho(:)));
else
    rho = volfrac * ones(nEl, 1);
end

opts_eig.tol = 1e-10;  opts_eig.maxit = 600;

%% ---------------- history allocation -----------------------------------
Z  = @() nan(outer_max_iter, 1);
ZM = @() nan(outer_max_iter, n_modes);
h = struct();
h.iter = (1:outer_max_iter)';
h.penal        = Z();
h.cont_stage   = Z();
h.omega        = ZM();   h.lambda       = ZM();
h.omega_trial  = ZM();   h.lambda_trial = ZM();
h.N            = Z();    h.N_trial      = Z();
h.J_idx        = Z();    h.lambda_bar   = Z();  h.lambda_J = Z();
h.g12          = Z();    h.g12_trial    = Z();  h.g23 = Z();
h.N_rec_1em2   = Z();    h.N_rec_5em3   = Z();
h.N_rec_1em3   = Z();    h.N_rec_1em4   = Z();
h.beta         = Z();
h.pred_dlambda = Z();    h.real_dlambda = Z();  h.pred_ratio = Z();
h.inner_iters  = Z();    h.inner_converged = false(outer_max_iter,1);
h.inner_reason = repmat({''}, outer_max_iter, 1);
h.inner_last_change = Z();  h.inner_singular_warn = Z();
h.inner_cpu    = Z();
h.asym_width_min = Z();  h.asym_width_mean = Z();  h.asym_width_max = Z();
h.asym_beta_low  = Z();  h.asym_beta_upp   = Z();
h.mma_lam_max    = Z();  h.mma_ymma_max    = Z();  h.mma_zmma = Z();
h.frac_at_bound  = Z();  h.frac_near_bound = Z();
h.n_at_lb = Z();  h.n_at_ub = Z();
h.drho_min = Z(); h.drho_max_signed = Z(); h.drho_inf = Z(); h.drho_l2n = Z();
h.pred_vol = Z();  h.vol = Z();  h.vol_before = Z();
h.fsk_norm = Z();  h.fsk_absmax = Z();
h.accepted = false(outer_max_iter,1);
h.reject_reason = repmat({''}, outer_max_iter, 1);
h.step_alpha = Z();
h.d1_inf = Z(); h.d1_rms = Z(); h.d2_inf = Z(); h.d2_rms = Z();
h.rel_domega1 = Z(); h.rel_dbeta = Z(); h.dvol = Z();
h.grayness = Z();
h.mac11 = Z(); h.mac_best_1 = Z(); h.mac_best_idx_1 = Z();
h.macmat = nan(outer_max_iter, n_modes*n_modes);
h.eig_flag = Z();
h.n_top_elems_90 = Z();

rho_snap = nan(nEl, outer_max_iter);
Phi_prev = [];  M_prev = [];
rho_km1 = [];   rho_km2 = [];
omega1_prev = NaN;  beta_prev = NaN;  vol_prev = mean(rho);

stop_status = 'MAX_ITERATIONS';
stop_iter   = outer_max_iter;

% continuation state
cont = cfg.cont;
stage = 1;  stage_start = 1;
if cont.enabled, penal_k = cont.p_values(1); else, penal_k = cfg.penal; end

if verbose
    fprintf('\n %-4s %-6s %-10s %-10s %-3s %-7s %-10s %-5s %-4s %-9s %s\n', ...
        'it','p','omega1','omega2','N','vol','beta(w)','in','cvg','drho','status');
    fprintf(' %s\n', repmat('-', 1, 100));
end

%% ================= OUTER LOOP =========================================
for out_it = 1:outer_max_iter

    % ---- continuation stage selection (paper Section 2.1: p increasing 1->3)
    if cont.enabled
        switch lower(cont.mode)
            case 'fixed'
                stage = min(numel(cont.p_values), floor((out_it-1)/cont.stage_len)+1);
            case 'drho'
                if stage < numel(cont.p_values) && out_it > 1 && ...
                        (out_it - stage_start) >= cont.min_stage_len && ...
                        h.d1_rms(out_it-1) < cont.drho_trigger
                    stage = stage + 1;  stage_start = out_it;
                end
            otherwise
                error('recon_solve: unknown cont.mode %s', cont.mode);
        end
        penal_k = cont.p_values(stage);
    end
    h.penal(out_it) = penal_k;
    h.cont_stage(out_it) = stage;

    rho_phys = max(rho_min, min(1, rho));

    % ---- step 1: assemble + eigensolve (paper Fig.1 step 1) -------------
    [K, M] = assemble_KM_exact(rho_phys, Ke_phys_l, Me_phys_l, iK, jK, nDof, ...
                               penal_k, mass_mode);
    Kf = K(free, free);  Mf = M(free, free);

    [V, D, flag] = eigs(Kf, Mf, n_modes, 'SM', opts_eig);
    if flag ~= 0
        opts_r.tol = 1e-8; opts_r.maxit = 1500;
        opts_r.p = min(nFree-1, max(40, 4*n_modes));
        [V, D, flag] = eigs(Kf, Mf, n_modes, 'SM', opts_r);
    end
    h.eig_flag(out_it) = flag;
    if flag ~= 0
        stop_status = 'EIGS_FAILURE';  stop_iter = out_it;
        h.reject_reason{out_it} = 'eigs_failed';
        break
    end

    [lam, idx] = sort(real(diag(D)));
    V = real(V(:, idx));
    for j = 1:n_modes
        v = V(:,j);  sc = sqrt(abs(v' * (Mf * v)));
        if sc > 1e-14, V(:,j) = v / sc; end
    end
    omega = sqrt(max(lam, 0));
    Phi = zeros(nDof, n_modes);
    for j = 1:n_modes, Phi(free, j) = V(:, j); end

    % ---- multiplicity, solver-reported and independently reconstructed ---
    [N, J_idx, cluster_idx] = detect_multiplicity(omega, n_target, mult_tol);
    lambda_bar = mean(lam(cluster_idx));
    h.N(out_it) = N;  h.J_idx(out_it) = J_idx;  h.lambda_bar(out_it) = lambda_bar;
    h.omega(out_it,:) = omega(:)';  h.lambda(out_it,:) = lam(:)';
    h.g12(out_it) = abs(omega(2)-omega(1)) / max(omega(1), eps);
    if n_modes >= 3, h.g23(out_it) = abs(omega(3)-omega(2))/max(omega(2), eps); end
    h.N_rec_1em2(out_it) = detect_multiplicity(omega, n_target, 1e-2);
    h.N_rec_5em3(out_it) = detect_multiplicity(omega, n_target, 5e-3);
    h.N_rec_1em3(out_it) = detect_multiplicity(omega, n_target, 1e-3);
    h.N_rec_1em4(out_it) = detect_multiplicity(omega, n_target, 1e-4);

    % ---- MAC vs previous outer iterate (Phase 6) ------------------------
    if ~isempty(Phi_prev)
        Mfull = M;
        Cmat = Phi_prev' * (Mfull * Phi);
        nrm_p = sqrt(abs(diag(Phi_prev' * (Mfull * Phi_prev))));
        nrm_c = sqrt(abs(diag(Phi'      * (Mfull * Phi))));
        MACm = (Cmat.^2) ./ max((nrm_p.^2) * (nrm_c.^2)', eps);
        h.macmat(out_it, :) = MACm(:)';
        h.mac11(out_it) = MACm(1,1);
        [bv, bi] = max(MACm(1,:));
        h.mac_best_1(out_it) = bv;  h.mac_best_idx_1(out_it) = bi;
    end
    Phi_prev = Phi;  M_prev = M;

    % ---- step 2: generalized gradients (Eq. 19) + sensitivity filter -----
    Phi_cluster = Phi(:, cluster_idx);
    fsk_raw = compute_generalized_gradients(rho_phys, lambda_bar, Phi_cluster, ...
                  cMat, Ke_phys, Me_phys, penal_k, mass_mode);
    fsk_use = zeros(size(fsk_raw));
    for s = 1:N
        for k = 1:N
            fsk_use(:,s,k) = apply_sensitivity_filter(fsk_raw(:,s,k), rho_phys, ...
                                 h_filt, Hs_filt, nely, nelx);
        end
    end
    h.fsk_norm(out_it)   = norm(reshape(fsk_use, [], 1));
    h.fsk_absmax(out_it) = max(abs(fsk_use(:)));

    if J_idx > 0
        lambda_J = lam(J_idx);
        dlam_J_raw = compute_elem_sensitivity(rho_phys, lambda_J, Phi(:,J_idx), ...
            cMat, Ke_phys, Me_phys, free, nDof, penal_k, mass_mode);
        dlam_J = apply_sensitivity_filter(dlam_J_raw, rho_phys, h_filt, Hs_filt, nely, nelx);
    else
        lambda_J = Inf;  dlam_J = [];
    end
    h.lambda_J(out_it) = lambda_J;

    % ---- step 3: inner loop (Eq. 25) ------------------------------------
    tic_in = tic;
    [drho, beta_fin, ih] = inner_loop_mma_instr(rho, lambda_bar, fsk_use, ...
        lambda_J, dlam_J, volfrac, rho_min, inner_max_iter, inner_tol, ...
        move_lim, outer_move);
    h.inner_cpu(out_it) = toc(tic_in);

    h.beta(out_it) = beta_fin;
    h.inner_iters(out_it) = ih.n_iters;
    h.inner_converged(out_it) = ih.converged;
    h.inner_reason{out_it} = ih.termination_reason;
    h.inner_last_change(out_it) = ih.drho_change(end);
    h.inner_singular_warn(out_it) = ih.n_singular_warn;
    h.asym_width_min(out_it)  = ih.asym_width_min(end);
    h.asym_width_mean(out_it) = ih.asym_width_mean(end);
    h.asym_width_max(out_it)  = ih.asym_width_max(end);
    h.asym_beta_low(out_it)   = ih.asym_beta_low(end);
    h.asym_beta_upp(out_it)   = ih.asym_beta_upp(end);
    h.mma_lam_max(out_it)  = ih.mma_lam_max(end);
    h.mma_ymma_max(out_it) = ih.mma_ymma_max(end);
    h.mma_zmma(out_it)     = ih.mma_zmma(end);
    h.frac_at_bound(out_it) = ih.frac_at_bound(end);
    h.frac_near_bound(out_it) = ih.frac_near_bound(end);
    h.n_at_lb(out_it) = ih.n_at_lb(end);
    h.n_at_ub(out_it) = ih.n_at_ub(end);
    h.drho_min(out_it) = min(drho);
    h.drho_max_signed(out_it) = max(drho);
    h.drho_inf(out_it) = max(abs(drho));
    h.drho_l2n(out_it) = norm(drho)/sqrt(nEl);
    h.pred_vol(out_it) = mean(rho + drho);
    h.pred_dlambda(out_it) = beta_fin - lambda_bar;

    if out_it == 1 || cfg.save_inner_full
        inner_full{min(out_it, outer_max_iter)} = ih; %#ok<AGROW>
    end
    inner_summary(out_it) = pack_inner(ih); %#ok<AGROW>

    % ---- Phase 4: fail-closed acceptance audit --------------------------
    [ok, reason] = fc_check(ih, drho, rho, rho_min, volfrac, outer_move, ...
                            cfg.fc_vol_tol, cfg.fc_bound_tol);
    h.accepted(out_it) = ok;
    h.reject_reason{out_it} = reason;

    if cfg.fail_closed && ~ok
        stop_status = 'INNER_FAILURE';  stop_iter = out_it;
        if verbose
            fprintf(' %-4d %-6.3g %-10s %-10s %-3d %-7.4f %-10s %-5d %-4d %-9s FAIL-CLOSED: %s\n', ...
                out_it, penal_k, '-', '-', N, mean(rho), '-', ih.n_iters, ...
                ih.converged, '-', reason);
        end
        break
    end

    % ---- step 4: outer update  rho := rho + alpha*Delta_rho -------------
    step_alpha = alpha;
    rho_new = max(rho_min, min(1, rho + step_alpha * drho));
    h.step_alpha(out_it) = step_alpha;

    rho_new_phys = max(rho_min, min(1, rho_new));
    [omega_trial, tflag] = recon_eval_omega(rho_new_phys, Ke_phys_l, Me_phys_l, ...
        iK, jK, nDof, free, n_modes, opts_eig, penal_k, mass_mode);
    lam_trial = omega_trial(:).^2;
    h.omega_trial(out_it,:) = omega_trial(:)';
    h.lambda_trial(out_it,:) = lam_trial(:)';
    if all(isfinite(omega_trial))
        h.N_trial(out_it) = detect_multiplicity(omega_trial, n_target, mult_tol);
        h.g12_trial(out_it) = abs(omega_trial(2)-omega_trial(1))/max(omega_trial(1), eps);
        h.real_dlambda(out_it) = lam_trial(n_target) - lam(n_target);
        if abs(h.pred_dlambda(out_it)) > eps
            h.pred_ratio(out_it) = h.real_dlambda(out_it) / h.pred_dlambda(out_it);
        end
    end

    % ---- design-change diagnostics (Q3) ---------------------------------
    d1 = rho_new - rho;
    h.d1_inf(out_it) = max(abs(d1));
    h.d1_rms(out_it) = norm(d1)/sqrt(nEl);
    if ~isempty(rho_km1)
        d2 = rho_new - rho_km1;
        h.d2_inf(out_it) = max(abs(d2));
        h.d2_rms(out_it) = norm(d2)/sqrt(nEl);
    end
    h.vol_before(out_it) = mean(rho);
    h.vol(out_it) = mean(rho_new);
    h.dvol(out_it) = h.vol(out_it) - vol_prev;
    h.grayness(out_it) = mean(4 * rho_new .* (1 - rho_new));
    if isfinite(omega1_prev) && isfinite(omega_trial(n_target))
        h.rel_domega1(out_it) = (omega_trial(n_target) - omega1_prev)/max(omega1_prev, eps);
    end
    if isfinite(beta_prev) && beta_prev ~= 0
        h.rel_dbeta(out_it) = (beta_fin - beta_prev)/abs(beta_prev);
    end
    % how many elements carry 90% of the squared design change
    ad = sort(d1.^2, 'descend');
    if sum(ad) > 0
        cs = cumsum(ad)/sum(ad);
        h.n_top_elems_90(out_it) = find(cs >= 0.9, 1, 'first');
    else
        h.n_top_elems_90(out_it) = 0;
    end

    rho_snap(:, out_it) = rho_new;
    rho_km2 = rho_km1;  rho_km1 = rho;
    omega1_prev = omega_trial(n_target);  beta_prev = beta_fin;
    vol_prev = h.vol(out_it);

    if verbose
        fprintf(' %-4d %-6.3g %-10.4f %-10.4f %-3d %-7.4f %-10.4f %-5d %-4d %-9.3e %s\n', ...
            out_it, penal_k, omega_trial(n_target), omega_trial(min(n_target+1,n_modes)), ...
            h.N_trial(out_it), h.vol(out_it), sqrt(max(beta_fin,0)), ih.n_iters, ...
            ih.converged, h.d1_rms(out_it), reason);
    end

    rho = rho_new;
    stop_iter = out_it;

    if h.d1_rms(out_it) < outer_tol
        stop_status = 'DRHO_TOL_MET';
        break
    end
end

%% ---------------- trim + finalize --------------------------------------
ni = stop_iter;
fn = fieldnames(h);
for k = 1:numel(fn)
    v = h.(fn{k});
    if size(v,1) == outer_max_iter, h.(fn{k}) = v(1:ni, :); end
end
rho_snap = rho_snap(:, 1:ni);

out = struct();
out.cfg = cfg;
out.hist = h;
out.rho_final = max(rho_min, min(1, rho));
out.rho_snapshots = rho_snap;
out.inner_summary = inner_summary(1:min(ni, numel(inner_summary)));
if exist('inner_full','var'), out.inner_full = inner_full; else, out.inner_full = {}; end
out.outer_iters = ni;
out.stop_status = stop_status;
out.nEl = nEl;  out.nDof = nDof;  out.nFree = nFree;
out.penal_final = penal_k;

[fo, ff] = recon_eval_omega(out.rho_final, Ke_phys_l, Me_phys_l, iK, jK, nDof, ...
    free, n_modes, opts_eig, penal_k, mass_mode);
out.final_omega = fo(:)';
out.final_flag  = ff;
if ff == 0
    [out.final_N, out.final_J, ~] = detect_multiplicity(fo, n_target, mult_tol);
    out.final_g12 = abs(fo(2)-fo(1))/max(fo(1), eps);
else
    out.final_N = NaN; out.final_J = NaN; out.final_g12 = NaN;
end
out.final_volume = mean(out.rho_final);

% Final state evaluated at p = 3 as well, so continuation runs are comparable
% with fixed-p runs on a common physical model.
[fo3, ff3] = recon_eval_omega(out.rho_final, Ke_phys_l, Me_phys_l, iK, jK, nDof, ...
    free, n_modes, opts_eig, 3.0, mass_mode);
out.final_omega_p3 = fo3(:)';
out.final_flag_p3  = ff3;
if ff3 == 0
    out.final_N_p3 = detect_multiplicity(fo3, n_target, mult_tol);
    out.final_g12_p3 = abs(fo3(2)-fo3(1))/max(fo3(1), eps);
else
    out.final_N_p3 = NaN;  out.final_g12_p3 = NaN;
end
end

%% ======================================================================
function cfg = recon_defaults(cfg)
    function c = def(c, f, v)
        if ~isfield(c, f) || isempty(c.(f)), c.(f) = v; end
    end
cfg = def(cfg,'L',8.0);        cfg = def(cfg,'H',1.0);
cfg = def(cfg,'nelx',160);     cfg = def(cfg,'nely',20);
cfg = def(cfg,'E0',1e7);       cfg = def(cfg,'nu',0.3);
cfg = def(cfg,'rho0',1.0);     cfg = def(cfg,'t',1.0);
cfg = def(cfg,'volfrac',0.5);  cfg = def(cfg,'rho_min',1e-3);
cfg = def(cfg,'penal',3.0);    cfg = def(cfg,'mass_mode','du2007_c1');
cfg = def(cfg,'rmin_elem',2.5);
cfg = def(cfg,'support_type','CC');
cfg = def(cfg,'n_target',1);   cfg = def(cfg,'n_modes',4);
cfg = def(cfg,'mult_tol',1e-3);
cfg = def(cfg,'outer_max_iter',300);
cfg = def(cfg,'outer_tol',1e-4);
cfg = def(cfg,'inner_max_iter',30);
cfg = def(cfg,'inner_tol',1e-4);
cfg = def(cfg,'move_lim',Inf); cfg = def(cfg,'outer_move',Inf);
cfg = def(cfg,'alpha',1.0);
cfg = def(cfg,'verbose',true);
cfg = def(cfg,'fail_closed',false);
cfg = def(cfg,'fc_vol_tol',1e-4);
cfg = def(cfg,'fc_bound_tol',1e-9);
cfg = def(cfg,'save_inner_full',false);
cfg = def(cfg,'tag','run');
if ~isfield(cfg,'cont') || isempty(cfg.cont), cfg.cont = struct(); end
cfg.cont = def(cfg.cont,'enabled',false);
cfg.cont = def(cfg.cont,'p_values',[1 1.5 2 2.5 3]);
cfg.cont = def(cfg.cont,'mode','fixed');
cfg.cont = def(cfg.cont,'stage_len',25);
cfg.cont = def(cfg.cont,'drho_trigger',1e-3);
cfg.cont = def(cfg.cont,'min_stage_len',10);
end

%% ======================================================================
function [ok, reason] = fc_check(ih, drho, rho, rho_min, volfrac, outer_move, ...
                                 vol_tol, bnd_tol)
% Gate G1/G3 predicate.  Returns ok=false with the FIRST violated condition.
ok = true;  reason = 'ok';
if ~ih.converged
    ok = false;  reason = sprintf('inner_not_converged(%s,%d it)', ...
        ih.termination_reason, ih.n_iters);  return
end
if any(~isfinite(ih.fval(:)))
    ok = false;  reason = 'nonfinite_constraint';  return
end
if any(~isfinite(drho))
    ok = false;  reason = 'nonfinite_increment';  return
end
lb = max(rho_min - rho, -outer_move*ones(size(rho)));
ub = min(1        - rho, +outer_move*ones(size(rho)));
if any(drho < lb - bnd_tol) || any(drho > ub + bnd_tol)
    ok = false;  reason = 'bound_violation';  return
end
if mean(rho + drho) > volfrac + vol_tol
    ok = false;
    reason = sprintf('volume_violation(%.3e)', mean(rho+drho)-volfrac);  return
end
end

%% ======================================================================
function s = pack_inner(ih)
s = struct();
s.n_iters   = ih.n_iters;
s.converged = ih.converged;
s.reason    = ih.termination_reason;
s.drho_change = ih.drho_change(:)';
s.beta        = ih.beta(:)';
s.fval_cluster = ih.fval_cluster(:)';
s.fval_vol     = ih.fval_vol(:)';
s.fval_J       = ih.fval_J(:)';
s.frac_at_bound = ih.frac_at_bound(:)';
s.frac_near_bound = ih.frac_near_bound(:)';
s.drho_inf      = ih.drho_inf(:)';
s.pred_vol      = ih.pred_vol(:)';
s.asym_width_mean = ih.asym_width_mean(:)';
s.mma_lam_max   = ih.mma_lam_max(:)';
s.mma_ymma_max  = ih.mma_ymma_max(:)';
s.n_singular_warn = ih.n_singular_warn;
end

%% ======================================================================
function [omega, flag] = recon_eval_omega(rho, Ke_phys_l, Me_phys_l, iK, jK, ...
                                          nDof, free, n_modes, opts_eig, ...
                                          penal, mass_mode)
[K, M] = assemble_KM_exact(rho, Ke_phys_l, Me_phys_l, iK, jK, nDof, penal, mass_mode);
Kf = K(free, free);  Mf = M(free, free);
nFree = numel(free);
[~, D, flag] = eigs(Kf, Mf, n_modes, 'SM', opts_eig);
if flag ~= 0
    opts_r.tol = 1e-8; opts_r.maxit = 1500;
    opts_r.p = min(nFree-1, max(40, 4*n_modes));
    [~, D, flag] = eigs(Kf, Mf, n_modes, 'SM', opts_r);
end
if flag == 0
    lam = sort(real(diag(D)));
    omega = sqrt(max(lam, 0));
else
    omega = nan(n_modes, 1);
end
end
