function [rho_final, hist] = topopt_freq_exact(cfg)
% TOPOPT_FREQ_EXACT  Olhoff & Du (2014) eigenfrequency topology optimization.
%
%   [rho_final, hist] = topopt_freq_exact(cfg)
%
%   Implements the computational procedure of Olhoff & Du (2014) Fig. 1 exactly:
%
%     0. rho <- volfrac uniform;  choose n
%     1. assemble K(rho), M(rho) from RAW rho;  solve (1b) and orthonormalize
%        per (1c);  detect multiplicity N of omega_n  (and R of omega_{n-1} for
%        the gap problem)
%     2. generalized gradients f_sk (Eq. 13) if N > 1, usual gradients (Eq. 4/5)
%        if N = 1  -- the same code path handles both, since Eq. (14)/(15) make
%        N = 1 a special case of Eq. (13)
%     3. inner loop: solve subproblem (19) (or (20)) for the increments
%        Delta_rho, with the outer-loop eigenpairs, f_sk, N and R held FIXED
%     4. rho := rho + Delta_rho          (no damping; alpha = 1)
%        stop when ||Delta_rho||_inf < outer_tol and the subproblem predicts no
%        further improvement
%
%   The one addition to the paper is the move limit cfg.move on Delta_rho,
%   entering constraint (19f).  It is unavoidable: for N = 1 problem (19) is a
%   linear program (the paper says so in the last paragraph of section 2.5), so
%   its exact optimum over the full box of (19f) is a vertex with essentially
%   every Delta_rho_e at a bound.  Taking that step destroys the design (CC
%   omega_1 145.6 -> 0.06 in one feasible in-bounds step, at every mesh from 492
%   to 14942 DOF).  Sequential LINEAR programming is not defined without a trust
%   region; the paper's own LP reduction cites Krog & Olhoff (1999).  The move
%   limit is therefore a reconstruction of an omitted implementation detail, is
%   the ONLY such parameter in this solver, is reported in hist.cfg, and is
%   calibrated in experiments/step_calibration.
%
%   See analysis/OlhoffApproachExact/PLAN_Olhoff2014_exact.md for the full
%   [E]xplicit / [D]u2007-import / [R]econstructed contract.  Tags below refer
%   to it.
%
%   CONFIG (all fields optional)
%   ----------------------------
%   Geometry / mesh
%     .L .H              design domain, default 8, 1                      [E14]
%     .nelx .nely        mesh, default 160, 20.  nely must be EVEN for SS/CS.
%     .t                 thickness, default 1
%   Material
%     .E0 .nu .rho0      default 1e7, 0.3, 1                              [E14]
%     .penal             SIMP exponent p, default 3                       [D1]
%     .mass_mode         'du2007_c1' (default) [D3] | 'olhoff2014_pow' [E6]
%                        | 'linear' | 'du2007_step' | 'du2007_c0'
%     .mass_q            mass exponent q for olhoff2014_pow, default 1     [D2]
%     .rho_min           default 1e-3                                     [E3]
%     .volfrac           default 0.5                                      [E2]
%   Regularization
%     .sensitivity_filter  default true                                   [D4]
%     .rmin_elem           filter radius in element units, default 2.5    [D4]
%   Boundary conditions / loads
%     .support_type      'SS' | 'CS' | 'CC' (default 'CC')                [E15]
%     .fixed_dofs        explicit override, 1-based
%     .lumped_mass       [] or struct('where','bottom_mid','mass',value)
%                        design-independent concentrated mass (section 3.3)
%     .lumped_mass_frac  alternative: mass as a fraction of the initial
%                        structural mass m_b (0.5 reproduces Fig. 7a)
%   Problem
%     .objective         'nth' (default, Eq. 19) | 'gap' (Eq. 20)
%     .n_target          target mode index n, default 1
%     .n_modes           modes computed each outer iteration,
%                        default n_target + N_max + 3, at least 6          [R5]
%   Multiplicity
%     .mult_tol_join     lambda-relative tolerance to join, default 5e-3   [R3]
%     .mult_tol_leave    lambda-relative tolerance to leave, default 1.5e-2[R3]
%     .N_max             cap on detected multiplicity, default 3
%     .cluster_model     'CA' (default) reference lam~ = one scalar, i.e.
%                        constraint (19c) with omega_j^2 = lam~ for all j
%                        'CC'          reference diag(lam_n..lam_{n+N-1}),
%                        the degenerate-perturbation variant for NEAR-multiple
%                        clusters; identical to CA at exact multiplicity    [R4]
%     .lam_ref_rule      'lowest' (default) | 'mean'  -- which scalar to use as
%                        lam~ inside f_sk (Eq. 13)                          [R4]
%   Step control                                                           [R1]
%     .step_control      'trust_region' (default) | 'fixed'
%     .move              move limit / initial trust-region radius, default 0.05
%     .move_min .move_max  radius bounds, default 1e-4 and 0.2
%     .tr_eta_lo .tr_eta_hi  reject / expand thresholds, default 0.25, 0.75
%     .tr_dec .tr_inc    contraction and expansion factors, default 0.5, 2.0
%   Inner loop
%     .subproblem_solver 'lp' (default, exact) | 'mma' (paper alternative) [R9]
%     .inner             options struct forwarded to the subproblem solver
%     .inner_audit       compute the exact optimality gap of the MMA solution
%                        every outer iteration (default true when solver='mma')
%   Outer loop
%     .outer_max_iter    default 300
%     .outer_tol         convergence on ||Delta_rho||_inf, default 1e-4    [R8]
%     .kkt_tol           relative predicted-improvement tolerance, 1e-6    [R8]
%   Reporting
%     .verbose           default true
%     .rho_snapshot_interval  store rho every k iterations, 0 = off
%
%   OUTPUT
%     rho_final   nEl x 1   FINAL design (paper semantics, Fig. 1) -- not
%                           best-seen.  hist.best_* reports best-seen for
%                           diagnosis only.
%     hist        struct    per-iteration telemetry; see the field list at the
%                           end of this file.  hist.stop_reason is always set.
%
%   Reference: N. Olhoff, J. Du, "Structural Topology Optimization with Respect
%   to Eigenfrequencies of Vibration", CISM Courses and Lectures, Springer 2014,
%   pp. 275-297, DOI 10.1007/978-3-7091-1643-2_11.

if nargin < 1 || isempty(cfg), cfg = struct(); end
cfg = set_defaults(cfg);

is_gap = strcmpi(cfg.objective, 'gap');
n      = cfg.n_target;

%% ---------------------------------------------------------------- mesh / FE
nelx = cfg.nelx;  nely = cfg.nely;
dx = cfg.L/nelx;  dy = cfg.H/nely;
nEl  = nelx * nely;
nDof = 2 * (nelx+1) * (nely+1);

[Ke_star, Me_star] = fe_q4_exact(cfg.nu, cfg.t, dx, dy);
Ke_phys = cfg.E0   * Ke_star;
Me_phys = cfg.rho0 * Me_star;

nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
cVec    = reshape(2*nodeNrs(1:nely,1:nelx)+1, nEl, 1);
cMat    = [cVec, cVec+1, cVec+2*nely+2, cVec+2*nely+3, ...
           cVec+2*nely, cVec+2*nely+1, cVec-2, cVec-1];

[Il, Jl]  = find(tril(ones(8)));
iK        = reshape(cMat(:,Il)', [], 1);
jK        = reshape(cMat(:,Jl)', [], 1);
Ke_phys_l = Ke_phys(sub2ind([8,8], Il, Jl));
Me_phys_l = Me_phys(sub2ind([8,8], Il, Jl));

if ~isempty(cfg.fixed_dofs)
    fixed = unique(double(cfg.fixed_dofs(:)));
else
    fixed = build_supports_exact(cfg.support_type, nodeNrs);
end
free  = setdiff(1:nDof, fixed);
nFree = numel(free);

if cfg.sensitivity_filter
    [h_filt, Hs_filt] = build_filter(nelx, nely, cfg.rmin_elem);
end

%% ------------------------------------------------------- initial density
if ~isempty(cfg.initial_rho)
    rho = max(cfg.rho_min, min(1, cfg.initial_rho(:)));
    if numel(rho) ~= nEl
        error('topopt_freq_exact:InvalidInitialRho', ...
            'initial_rho has %d entries, expected %d.', numel(rho), nEl);
    end
else
    rho = cfg.volfrac * ones(nEl, 1);       % uniform, per section 3.1  [E14]
end

%% -------------------------------------------- concentrated (non-structural) mass
Mc = sparse(nDof, nDof);
lumped_note = 'none';
if ~isempty(cfg.lumped_mass) || ~isempty(cfg.lumped_mass_frac)
    if ~isempty(cfg.lumped_mass_frac)
        % m_b = total structural mass of the INITIAL design (section 3.3).
        m_b  = sum(mass_interp(rho, cfg.mass_mode, cfg.mass_q)) * cfg.rho0 * dx*dy*cfg.t;
        mval = cfg.lumped_mass_frac * m_b;
        where = 'bottom_mid';
        if ~isempty(cfg.lumped_mass) && isfield(cfg.lumped_mass,'where')
            where = cfg.lumped_mass.where;
        end
    else
        mval  = cfg.lumped_mass.mass;
        where = cfg.lumped_mass.where;
    end
    [Mc, mnode] = lumped_mass(nodeNrs, nDof, where, mval);
    lumped_note = sprintf('%s, m_c = %.6g at node %d', where, mval, mnode);
end

%% --------------------------------------------------------------- history
K_it = cfg.outer_max_iter;
hist = struct();
hist.omega        = nan(K_it, cfg.n_modes);
hist.beta         = nan(K_it, 1);
hist.beta1        = nan(K_it, 1);
hist.obj          = nan(K_it, 1);
hist.volume       = nan(K_it, 1);
hist.N            = nan(K_it, 1);
hist.R            = nan(K_it, 1);
hist.J_idx        = nan(K_it, 1);
hist.cluster_spread = nan(K_it, 1);   % (lam_max-lam_min)/lam_n within cluster
hist.eigengap     = nan(K_it, 1);     % (lam_{J} - lam_{n+N-1})/lam_n
hist.drho_inf     = nan(K_it, 1);
hist.drho_norm    = nan(K_it, 1);
hist.frac_at_bound= nan(K_it, 1);
hist.inner_iters  = nan(K_it, 1);
hist.inner_cuts   = nan(K_it, 1);
hist.inner_gap_rel= nan(K_it, 1);
hist.inner_viol   = nan(K_it, 1);
hist.inner_stop   = cell(K_it, 1);
hist.fd_audit     = nan(K_it, 1);     % predicted vs realised cluster dlambda
hist.move         = nan(K_it, 1);     % trust-region radius in force at this step
hist.ratio        = nan(K_it, 1);     % realised / predicted objective increase
hist.accepted     = nan(K_it, 1);     % 1 = step taken, 0 = rejected by the ratio test
hist.mac_mode_n   = nan(K_it, 1);
hist.subspace_ang = nan(K_it, 1);
hist.n_components = nan(K_it, 1);
hist.grey_frac    = nan(K_it, 1);
if cfg.rho_snapshot_interval > 0
    nSnap = ceil(K_it / cfg.rho_snapshot_interval) + 1;
    hist.rho_snapshot_iters = nan(nSnap,1);
    hist.rho_snapshots      = nan(nEl, nSnap);
    hist.rho_snapshot_count = 0;
end

opts_eig       = struct('tol', 1e-12, 'maxit', 800);
opts_eig.v0    = make_v0(nFree);

N_prev   = 1;
R_prev   = 1;
Phi_prev = [];
best_obj = -Inf;  best_rho = rho;  best_omega = [];

stop_reason = 'max_iter';
out_it_done = 0;

if cfg.verbose
    fprintf('\n Olhoff & Du (2014) %s  |  n = %d  |  %s  |  %dx%d  |  move = %g  |  solver = %s\n', ...
        upper(cfg.objective), n, cfg.support_type, nelx, nely, cfg.move, cfg.subproblem_solver);
    fprintf(' mass = %s, p = %g, filter = %d (rmin %g), cluster model %s/%s, lumped mass: %s\n', ...
        cfg.mass_mode, cfg.penal, cfg.sensitivity_filter, cfg.rmin_elem, ...
        cfg.cluster_model, cfg.lam_ref_rule, lumped_note);
    fprintf(' step control: %s (m0 = %g, [%g, %g], eta %g/%g, gamma %g/%g)\n', ...
        cfg.step_control, cfg.move, cfg.move_min, cfg.move_max, ...
        cfg.tr_eta_lo, cfg.tr_eta_hi, cfg.tr_dec, cfg.tr_inc);
    fprintf(' %4s %11s %11s %11s %3s %3s %7s %10s %9s %7s %6s %s%5s\n', ...
        'it','omega_n','omega_n+1','beta^1/2','N','R','vol','|drho|inf','fd_err', ...
        'move','ratio','','comp');
    fprintf(' %s\n', repmat('-', 1, 112));
end

%% ------------------------------------------------- initial eigen state
[lam, omega, Phi, M, eig_ok] = eig_state(rho);
if ~eig_ok
    error('topopt_freq_exact:InitialEigenFailure', ...
        'The eigensolver failed on the initial design.');
end

move_k = cfg.move;                 % current trust-region radius (Fig. 1 has none)

%% ============================================================ MAIN LOOP
for out_it = 1:cfg.outer_max_iter
    out_it_done = out_it;

    % ---- Fig. 1 step 1 (cont.): multiplicity detection ------------------
    [N, J_idx, cl_up] = detect_multiplicity(lam, n, cfg.mult_tol_join, ...
                                            cfg.mult_tol_leave, N_prev);
    N      = min(N, cfg.N_max);
    cl_up  = n : n+N-1;
    J_idx  = n + N;  if J_idx > cfg.n_modes, J_idx = 0; end
    N_prev = N;

    R = 0; cl_lo = []; Jm_idx = 0;
    if is_gap
        [R, Jm_idx, cl_lo] = detect_multiplicity_below(lam, n, cfg.mult_tol_join, ...
                                                       cfg.mult_tol_leave, R_prev);
        R      = min(R, cfg.N_max);
        cl_lo  = (n-R) : (n-1);
        Jm_idx = n - R - 1;  if Jm_idx < 1, Jm_idx = 0; end
        R_prev = R;
    end

    % ---- Fig. 1 step 2: generalized gradients (Eq. 13) ------------------
    sp = struct();
    sp.mode    = cfg.objective;
    sp.rho     = rho;
    sp.volfrac = cfg.volfrac;
    sp.rho_min = cfg.rho_min;
    sp.move    = move_k;

    sp.up = build_block(lam, Phi, cl_up, J_idx);
    if is_gap
        sp.lo = build_block(lam, Phi, cl_lo, Jm_idx);
    end

    % ---- Fig. 1 step 3: inner loop -------------------------------------
    switch lower(cfg.subproblem_solver)
        case 'lp'
            sol = subproblem_lp(sp, cfg.inner);
            gap_rel = 0;                            % exact by construction
        case 'mma'
            sol = subproblem_mma(sp, cfg.inner);
            if cfg.inner_audit
                q = subproblem_kkt(sp, sol);
                gap_rel = q.gap_rel;
            else
                gap_rel = NaN;
            end
        otherwise
            error('topopt_freq_exact:UnknownSolver', ...
                'subproblem_solver must be ''lp'' or ''mma''.');
    end

    drho = sol.drho;

    % Predicted cluster increments (Eq. 19d) from the TRUE (unfiltered)
    % linearization -- see build_block for why the filtered one must not be used
    % here.
    pred_up = predict_cluster(sp.up, drho);      % predicted NEW cluster eigenvalues
    pred_lo = [];
    if is_gap
        pred_lo = predict_cluster(sp.lo, drho);
    end

    % ---- Fig. 1 step 4: trial update -----------------------------------
    rho_trial = max(cfg.rho_min, min(1, rho + drho));   % alpha = 1, no damping
    drho_eff  = rho_trial - rho;

    [lam_t, omega_t, Phi_t, M_t, eig_ok] = eig_state(rho_trial);
    if ~eig_ok
        stop_reason = 'eigensolver_failure';
        if cfg.verbose
            fprintf(' STOP iter %d: eigensolver failed on the trial design.\n', out_it);
        end
        break
    end

    % ---- Ratio test (standard SLP globalization) -----------------------
    % predicted = what subproblem (19) claims the bound can be raised to;
    % realised  = what the eigenproblem actually delivers.  Their ratio is the
    % classical trust-region test.  Constants are Powell's textbook values, not
    % fitted to this problem.  Fig. 1 has no step control at all -- see the
    % header and PLAN_Olhoff2014_exact.md [R1].
    % The objective of (19) is the LOWEST constrained eigenvalue, so the
    % predicted increase is min(predicted new cluster eigenvalues) - lam_n.
    % It is NOT a per-mode difference min(pred_new_i - lam_i): under cluster
    % model CA the reference is lam_n for every member, so that expression is
    % mu_2 - (lam_{n+1} - lam_n) for the second mode and turns negative as soon
    % as the cluster has any spread.  That made pred_inc <= 0 -- and hence the
    % ratio undefined -- on EVERY iteration with N >= 2, i.e. the ratio test
    % switched itself off exactly when the multiple-eigenvalue path engaged.
    if is_gap
        obj_now  = lam(n) - lam(n-1);
        obj_new  = lam_t(n) - lam_t(n-1);
        pred_inc = (min(pred_up) - max(pred_lo)) - obj_now;
    else
        obj_now  = lam(n);
        obj_new  = lam_t(n);
        pred_inc = min(pred_up) - obj_now;
    end
    act_inc = obj_new - obj_now;

    tiny = 1e-12 * max(1, abs(obj_now));
    if pred_inc > tiny
        ratio = act_inc / pred_inc;
    else
        ratio = NaN;             % the model predicts no gain: ratio undefined
    end

    % FD audit: predicted vs realised cluster eigenvalues, both sorted.
    fd_err = max(abs(pred_up(:) - lam_t(cl_up(:)))) / max(abs(lam(n)), eps);

    accepted = true;
    if strcmpi(cfg.step_control, 'trust_region')
        if isnan(ratio)
            % Undefined ratio must NOT mean "accept unconditionally" -- that is
            % how an unmodelled step gets taken.  Fall back to the objective
            % itself: keep the step only if it actually improved something.
            if act_inc <= 0
                accepted = false;
                move_k   = max(cfg.move_min, cfg.tr_dec * move_k);
            end
        elseif ratio < cfg.tr_eta_lo
            accepted = false;
            move_k   = max(cfg.move_min, cfg.tr_dec * move_k);
        elseif ratio > cfg.tr_eta_hi && max(abs(drho)) >= 0.99*move_k
            move_k   = min(cfg.move_max, cfg.tr_inc * move_k);
        end
    end

    % ---- Telemetry ------------------------------------------------------
    hist.omega(out_it, :)     = omega(:)';      % state BEFORE this step
    hist.move(out_it)         = move_k;
    hist.ratio(out_it)        = ratio;
    hist.accepted(out_it)     = accepted;
    hist.fd_audit(out_it)     = fd_err;

    if accepted
        rho_new = rho_trial;
    else
        rho_new  = rho;
        drho_eff = zeros(nEl,1);
    end
    hist.beta(out_it)         = sol.beta;
    hist.beta1(out_it)        = sol.beta1;
    hist.obj(out_it)          = sol.obj;
    hist.volume(out_it)       = mean(rho_new);
    hist.N(out_it)            = N;
    hist.R(out_it)            = R;
    hist.J_idx(out_it)        = J_idx;
    hist.cluster_spread(out_it) = (lam(cl_up(end)) - lam(cl_up(1))) / max(lam(n), eps);
    if J_idx > 0
        hist.eigengap(out_it) = (lam(J_idx) - lam(cl_up(end))) / max(lam(n), eps);
    end
    hist.drho_inf(out_it)     = max(abs(drho_eff));
    hist.drho_norm(out_it)    = norm(drho_eff)/sqrt(nEl);
    hist.frac_at_bound(out_it)= sol.frac_at_bound;
    if isfield(sol,'n_iters'), hist.inner_iters(out_it) = sol.n_iters;
    else,                      hist.inner_iters(out_it) = sol.n_lp; end
    hist.inner_cuts(out_it)   = sol.n_cuts;
    hist.inner_gap_rel(out_it)= gap_rel;
    hist.inner_viol(out_it)   = sol.lmi_violation;
    hist.inner_stop{out_it}   = sol.stop_reason;
    hist.grey_frac(out_it)    = mean(rho_new > 0.1 & rho_new < 0.9);
    hist.n_components(out_it) = count_components(rho_new, nely, nelx, 0.5, 0.005);

    if ~isempty(Phi_prev)
        % Phi_prev holds the PREVIOUS cluster, so column 1 is mode n.
        hist.mac_mode_n(out_it) = mac(Phi(:,n), Phi_prev(:,1), M);
        a = Phi(free, cl_up);  b = Phi_prev(free, :);
        if size(a,2) == size(b,2) && size(a,2) >= 1
            hist.subspace_ang(out_it) = subspace(a, b);
        end
    end
    Phi_prev = Phi(:, cl_up);

    if cfg.rho_snapshot_interval > 0 && ...
       (out_it == 1 || mod(out_it, cfg.rho_snapshot_interval) == 0)
        hist.rho_snapshot_count = hist.rho_snapshot_count + 1;
        si = hist.rho_snapshot_count;
        hist.rho_snapshot_iters(si) = out_it;
        hist.rho_snapshots(:, si)   = rho_new;
    end

    if is_gap, cur_obj = omega(n) - omega(n-1); else, cur_obj = omega(n); end
    if cur_obj > best_obj
        best_obj = cur_obj;  best_rho = rho;  best_omega = omega;
    end

    if cfg.verbose
        if accepted, amark = ' '; else, amark = 'x'; end
        fprintf(' %4d %11.4f %11.4f %11.4f %3d %3d %7.4f %10.3e %9.2e %7.4f %6.2f %s%5d\n', ...
            out_it, omega(n), omega(min(n+1, cfg.n_modes)), sqrt(max(sol.beta,0)), ...
            N, R, mean(rho_new), hist.drho_inf(out_it), fd_err, move_k, ratio, ...
            amark, hist.n_components(out_it));
    end

    % ---- Carry the trial state forward when the step was accepted -------
    rho = rho_new;
    if accepted
        lam = lam_t;  omega = omega_t;  Phi = Phi_t;  M = M_t;
    end

    % ---- Convergence (Fig. 1: ||Delta_rho|| < eps) ----------------------
    pred_improve = pred_inc / max(abs(lam(n)), eps);
    small_step   = max(abs(drho)) < cfg.outer_tol;
    tr_exhausted = strcmpi(cfg.step_control,'trust_region') && ...
                   move_k <= cfg.move_min*(1+1e-12);
    if small_step || tr_exhausted
        if abs(pred_improve) < cfg.kkt_tol
            stop_reason = 'kkt_converged';
        elseif tr_exhausted
            stop_reason = 'trust_region_exhausted';
        else
            stop_reason = 'increment_small';
        end
        if cfg.verbose
            fprintf([' STOP iter %d: %s (|drho|inf = %.3e, move = %.3e, ' ...
                     'predicted improvement = %.3e)\n'], out_it, stop_reason, ...
                    max(abs(drho)), move_k, pred_improve);
        end
        break
    end
end

%% ------------------------------------------------------------- finalize
ni = out_it_done;
fns = fieldnames(hist);
for k = 1:numel(fns)
    v = hist.(fns{k});
    if (isnumeric(v) || iscell(v)) && size(v,1) == K_it
        hist.(fns{k}) = v(1:ni, :);
    end
end

rho_final = rho;
[final_omega, final_flag] = eval_omega(rho_final);
hist.final_omega = final_omega(:)';
hist.final_flag  = final_flag;
if final_flag == 0
    lamf = final_omega.^2;
    [hist.final_N, hist.final_J_idx, ~] = detect_multiplicity(lamf, n, ...
        cfg.mult_tol_join, cfg.mult_tol_leave, 1);
    hist.final_N = min(hist.final_N, cfg.N_max);
    if hist.final_N >= 2
        hist.final_cluster_spread_omega = ...
            (final_omega(n+hist.final_N-1) - final_omega(n)) / final_omega(n);
    else
        hist.final_cluster_spread_omega = NaN;
    end
else
    hist.final_N = NaN;  hist.final_J_idx = NaN;
    hist.final_cluster_spread_omega = NaN;
end
hist.final_volume     = mean(rho_final);
hist.final_components = count_components(rho_final, nely, nelx, 0.5, 0.005);
hist.outer_iters      = ni;
hist.stop_reason      = stop_reason;
hist.best_obj         = best_obj;
hist.best_rho         = best_rho;
hist.best_omega       = best_omega;
hist.cfg              = cfg;
hist.mesh             = struct('nelx',nelx,'nely',nely,'nEl',nEl,'nDof',nDof, ...
                               'nFree',nFree,'fixed',fixed);

if cfg.verbose
    fprintf('\n stop_reason = %s after %d outer iterations\n', stop_reason, ni);
    fprintf(' final omega(1:%d) = %s\n', min(4,cfg.n_modes), ...
        mat2str(round(final_omega(1:min(4,cfg.n_modes))',3)));
    fprintf(' final N = %g, volume = %.4f, structural components = %d\n\n', ...
        hist.final_N, hist.final_volume, hist.final_components);
end

%% =================================================== nested helpers
    function blk = build_block(lam_all, Phi_all, cl, guard_idx)
        % Generalized gradients + reference eigenvalues for one cluster,
        % plus the adjacent simple-mode guard constraint.
        Nb   = numel(cl);
        switch lower(cfg.lam_ref_rule)
            case 'mean',   lam_tilde = mean(lam_all(cl));
            case 'lowest', lam_tilde = lam_all(cl(1));
            otherwise, error('topopt_freq_exact:BadLamRefRule', ...
                    'lam_ref_rule must be ''lowest'' or ''mean''.');
        end

        Fe_raw = generalized_gradients(rho, lam_tilde, Phi_all(:,cl), cMat, ...
                     Ke_phys, Me_phys, cfg.penal, cfg.mass_mode, cfg.mass_q);
        Fe = Fe_raw;

        if cfg.sensitivity_filter
            for s = 1:Nb
                for kk = s:Nb
                    f = apply_sensitivity_filter(Fe_raw(:,s,kk), rho, h_filt, ...
                                                 Hs_filt, nely, nelx);
                    Fe(:,s,kk) = f;
                    Fe(:,kk,s) = f;      % keep F exactly symmetric
                end
            end
        end

        switch upper(cfg.cluster_model)
            case 'CA', L = lam_tilde * ones(Nb,1);
            case 'CC', L = lam_all(cl);
            otherwise, error('topopt_freq_exact:BadClusterModel', ...
                    'cluster_model must be ''CA'' or ''CC''.');
        end

        % Fe     : filtered, used by the subproblem (the [D4] regularization)
        % Fe_raw : unfiltered, the TRUE linearization of Eq. (13).  Sensitivity
        %          filtering is not a consistent gradient of anything, so the
        %          trust-region ratio test and the FD audit must be measured
        %          against Fe_raw or they compare a filtered prediction with an
        %          unfiltered reality and floor at a constant ratio (measured:
        %          0.22 on the SS beam, independent of the step length).
        blk = struct('L', L(:), 'Fe', Fe, 'Fe_raw', Fe_raw, 'guard', []);

        if guard_idx > 0 && guard_idx <= numel(lam_all)
            g = compute_elem_sensitivity(rho, lam_all(guard_idx), ...
                    Phi_all(:,guard_idx), cMat, Ke_phys, Me_phys, free, nDof, ...
                    cfg.penal, cfg.mass_mode, cfg.mass_q);
            if cfg.sensitivity_filter
                g = apply_sensitivity_filter(g, rho, h_filt, Hs_filt, nely, nelx);
            end
            blk.guard = struct('lam', lam_all(guard_idx), 'grad', g);
        end
    end

    function [lm, om, Ph, Mx, ok] = eig_state(r)
    % Fig. 1 step 1: assemble from RAW rho, solve (1b), M-orthonormalize (1c).
        [Kx, Mx] = assemble_KM_exact(r, Ke_phys_l, Me_phys_l, iK, jK, nDof, ...
                                     cfg.penal, cfg.mass_mode, cfg.mass_q);
        Mx = Mx + Mc;                  % non-structural mass, drho-independent
        Kf = Kx(free, free);
        Mf = Mx(free, free);

        [Vx, Dx, fl] = eigs(Kf, Mf, cfg.n_modes, 'SM', opts_eig);
        if fl ~= 0
            o2 = struct('tol', 1e-9, 'maxit', 2000, ...
                        'p', min(nFree-1, max(40, 4*cfg.n_modes)), 'v0', opts_eig.v0);
            [Vx, Dx, fl] = eigs(Kf, Mf, cfg.n_modes, 'SM', o2);
        end
        ok = (fl == 0);
        if ~ok
            lm = nan(cfg.n_modes,1); om = lm; Ph = zeros(nDof, cfg.n_modes);
            return
        end

        [lm, ix] = sort(real(diag(Dx)));
        Vx = real(Vx(:, ix));
        for j = 1:cfg.n_modes
            s = sqrt(abs(Vx(:,j)' * (Mf * Vx(:,j))));
            if s > 1e-300, Vx(:,j) = Vx(:,j)/s; end
        end
        om = sqrt(max(lm, 0));
        Ph = zeros(nDof, cfg.n_modes);
        Ph(free, :) = Vx;
    end

    function [om, fl] = eval_omega(r)
        [~, om, ~, ~, okx] = eig_state(r);
        if okx, fl = 0; else, fl = 1; end
    end
end

%% =========================================================== free helpers
function cfg = set_defaults(cfg)
    cfg = def(cfg, 'L', 8.0);            cfg = def(cfg, 'H', 1.0);
    cfg = def(cfg, 'nelx', 160);         cfg = def(cfg, 'nely', 20);
    cfg = def(cfg, 'E0', 1e7);           cfg = def(cfg, 'nu', 0.3);
    cfg = def(cfg, 'rho0', 1.0);         cfg = def(cfg, 't', 1.0);
    cfg = def(cfg, 'volfrac', 0.5);      cfg = def(cfg, 'rho_min', 1e-3);
    cfg = def(cfg, 'penal', 3.0);
    cfg = def(cfg, 'mass_mode', 'du2007_c1');
    cfg = def(cfg, 'mass_q', 1.0);
    cfg = def(cfg, 'sensitivity_filter', true);
    cfg = def(cfg, 'rmin_elem', 2.5);
    cfg = def(cfg, 'support_type', 'CC');
    cfg = def(cfg, 'fixed_dofs', []);
    cfg = def(cfg, 'lumped_mass', []);
    cfg = def(cfg, 'lumped_mass_frac', []);
    cfg = def(cfg, 'objective', 'nth');
    cfg = def(cfg, 'n_target', 1);
    cfg = def(cfg, 'N_max', 3);
    cfg = def(cfg, 'n_modes', max(cfg.n_target + cfg.N_max + 3, 6));
    % Multiplicity tolerances are on LAMBDA (= omega^2), relative.  Natural
    % coalescence in these beams is 0.3-1.3 % wide on omega, i.e. 0.6-2.6 % on
    % lambda, so a join tolerance of 2e-2 on lambda (= 1 % on omega) is the
    % smallest value that actually fires -- and 1 % on omega is the value that
    % was empirically found to activate N = 2 on the CC beam.  tol_leave is
    % 2.5x larger to give the Schmitt hysteresis of detect_multiplicity.
    cfg = def(cfg, 'mult_tol_join',  2e-2);
    cfg = def(cfg, 'mult_tol_leave', 5e-2);
    cfg = def(cfg, 'cluster_model', 'CA');
    cfg = def(cfg, 'lam_ref_rule', 'lowest');
    % ---- Step control [R1] ---------------------------------------------
    % 'fixed'         : constant move limit m, added to (19f).  Measured to be
    %                   unable to both preserve the basin and converge: at
    %                   m >= 0.05 the SS beam exits the basin at iteration 12-13
    %                   (linearization error reaches lambda_1 itself) and
    %                   oscillates; at m <= 0.02 it is stable but |drho|_inf
    %                   stays pinned at m forever, because the optimum of an LP
    %                   subproblem is always a move-limit vertex, so the Fig. 1
    %                   stopping test ||drho|| < eps can never be met.
    % 'trust_region'  : the standard SLP globalization -- accept/reject on the
    %                   ratio of realised to predicted objective increase, and
    %                   contract or expand m accordingly.  Constants below are
    %                   Powell's textbook values, NOT fitted to this problem.
    cfg = def(cfg, 'step_control', 'trust_region');
    cfg = def(cfg, 'move',      0.05);   % initial radius
    cfg = def(cfg, 'move_min',  1e-4);
    cfg = def(cfg, 'move_max',  0.2);
    cfg = def(cfg, 'tr_eta_lo', 0.25);
    cfg = def(cfg, 'tr_eta_hi', 0.75);
    cfg = def(cfg, 'tr_dec',    0.5);
    cfg = def(cfg, 'tr_inc',    2.0);
    cfg = def(cfg, 'subproblem_solver', 'lp');
    cfg = def(cfg, 'inner', struct());
    cfg = def(cfg, 'inner_audit', strcmpi(cfg.subproblem_solver,'mma'));
    cfg = def(cfg, 'outer_max_iter', 300);
    cfg = def(cfg, 'outer_tol', 1e-4);
    cfg = def(cfg, 'kkt_tol', 1e-6);
    cfg = def(cfg, 'verbose', true);
    cfg = def(cfg, 'rho_snapshot_interval', 0);
    cfg = def(cfg, 'initial_rho', []);
    if strcmpi(cfg.objective,'gap') && cfg.n_target < 2
        error('topopt_freq_exact:GapNeedsN2', ...
            'The gap problem (Eq. 20) requires n_target >= 2.');
    end
end

function s = def(s, f, v)
    if ~isfield(s, f) || isempty(s.(f)), s.(f) = v; end
end

function p = predict_cluster(blk, drho)
% Predicted NEW eigenvalues of a cluster after the increment Delta_rho, i.e.
% the eigenvalues of G = diag(L) + sum_e Delta_rho_e F_e, which is what the
% sub-eigenvalue problem (12)/(19d) defines.  Built from the TRUE (unfiltered)
% generalized gradients -- see build_block for why the filtered ones must not
% be used for model-accuracy measurements.  Returned sorted ascending.
    Nb = numel(blk.L);
    F2 = reshape(blk.Fe_raw, numel(drho), Nb*Nb);
    G  = diag(blk.L(:)) + reshape(F2' * drho, Nb, Nb);
    p  = sort(real(eig((G+G')/2)));
end

function v0 = make_v0(nFree)
    % Deterministic eigs start vector so runs are bit-reproducible.
    v0 = ones(nFree,1);
    v0(2:2:end) = -1;
    v0 = v0 / norm(v0);
end

function val = mac(pa, pb, M)
    num = (pa' * (M * pb))^2;
    den = (pa' * (M * pa)) * (pb' * (M * pb));
    if den > 0, val = num/den; else, val = NaN; end
end

function nc = count_components(rho, nely, nelx, thr, min_frac)
% Number of 8-connected solid components holding at least min_frac of the mesh.
    A   = reshape(rho, nely, nelx) >= thr;
    lab = zeros(nely, nelx);
    cur = 0;
    sizes = [];
    stack = zeros(nely*nelx, 2);
    for j = 1:nelx
        for i = 1:nely
            if A(i,j) && lab(i,j) == 0
                cur = cur + 1;
                sp  = 1; stack(1,:) = [i j]; lab(i,j) = cur; cnt = 1;
                while sp > 0
                    ci = stack(sp,1); cj = stack(sp,2); sp = sp - 1;
                    for di = -1:1
                        for dj = -1:1
                            ni = ci+di; nj = cj+dj;
                            if ni >= 1 && ni <= nely && nj >= 1 && nj <= nelx ...
                               && A(ni,nj) && lab(ni,nj) == 0
                                lab(ni,nj) = cur;
                                sp = sp + 1; stack(sp,:) = [ni nj];
                                cnt = cnt + 1;
                            end
                        end
                    end
                end
                sizes(cur) = cnt; %#ok<AGROW>
            end
        end
    end
    nc = sum(sizes >= min_frac * nely * nelx);
end
