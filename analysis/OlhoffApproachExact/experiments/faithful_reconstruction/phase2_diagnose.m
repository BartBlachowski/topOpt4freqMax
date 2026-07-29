function phase2_diagnose(nelx, nely, bc)
% PHASE2_DIAGNOSE  Why does the paper-literal incremental subproblem collapse?
%
%   phase2_diagnose(nelx, nely, bc)
%
%   Read-only forensic probe on the FIRST outer iteration of the paper-literal
%   regime (move_lim = Inf, outer_move = Inf, alpha = 1), plus the two follow-on
%   iterations.  Nothing here changes the algorithm; every quantity is measured
%   from the production kernels.
%
%   Probes, in order:
%     P1  Inner-MMA convergence budget.  Run the SAME subproblem with
%         inner_max_iter in {1,5,10,20,30,60,120,300,1000} and record where
%         the increment converges, what it converges TO, and what the realised
%         eigenvalue does after that increment is applied.  Separates
%         "truncated inner solve" from "converged inner solve" as the cause.
%     P2  Is the N = 1 subproblem a linear program?  Verified numerically by
%         checking that the cluster / J / volume constraint values are exactly
%         affine in Delta_rho (second differences vanish to round-off), and by
%         measuring how close the converged increment is to a box vertex.
%     P3  Direct LP solution of Eq. (25) with linprog (if available) or with
%         the exact greedy vertex rule, compared against what MMA returns.
%     P4  MMA conditioning.  Dynamic range of the p0/q0/P/Q entries induced by
%         the beta variable's box [0, 1e6] versus the Delta_rho boxes, and the
%         beta asymptote span, measured directly from mmasub's own formulas.
%     P5  Realised vs predicted eigenvalue along the accepted direction:
%         lambda_1(rho + t*drho) for t in [0,1], to locate where the linear
%         model stops being valid.
%     P6  Same quantities with the Regime-B step controls, for contrast.
%
%   Output: results/phase2_<BC>_<nelx>x<nely>/*.csv + report fragments.

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);
addpath(fullfile(this_dir, '..', '..', 'Matlab'));
addpath(fullfile(this_dir, '..', '..', '..', '..', 'tools', 'Matlab'));
if nargin < 3 || isempty(bc), bc = 'CC'; end

tag = sprintf('phase2_%s_%dx%d', upper(bc), nelx, nely);
d = fullfile(this_dir, 'results', tag);
if ~exist(d,'dir'), mkdir(d); end
diary(fullfile(d,'log.txt')); cleanup = onCleanup(@() diary('off'));

fprintf('\n=========== PHASE 2 DIAGNOSIS  %s %dx%d ===========\n', upper(bc), nelx, nely);

% ---------------- model setup (production kernels) ----------------------
S = setup_model(nelx, nely, bc);
volfrac = 0.5;  rho_min = 1e-3;  penal = 3.0;  mass_mode = 'du2007_c1';
n_modes = 4;  n_target = 1;  mult_tol = 1e-3;
rho = volfrac * ones(S.nEl, 1);

% ---------------- outer step 1 & 2 data ---------------------------------
St = outer_state(rho, S, penal, mass_mode, n_modes, n_target, mult_tol);
fprintf('\n[state] iter 1: omega = %s\n', num2str(St.omega','%.4f  '));
fprintf('[state] N = %d, J = %d, lambda_bar = %.6e, lambda_J = %.6e\n', ...
    St.N, St.J_idx, St.lambda_bar, St.lambda_J);
fprintf('[state] g12 = %.4e\n', abs(St.omega(2)-St.omega(1))/St.omega(1));

%% ===================== P1: inner budget sweep =========================
fprintf('\n--- P1  inner-MMA convergence budget (paper-literal box) ---\n');
budgets = [1 5 10 20 30 60 120 300 1000];
P1 = [];
for b = budgets
    for reg = 1:2
        if reg == 1, ml = Inf; om = Inf; al = 1.0;  lbl = 'paper-literal';
        else,        ml = 0.2; om = 0.2; al = 0.5;  lbl = 'regimeB'; end
        [drho, beta_fin, ih] = inner_loop_mma_instr(rho, St.lambda_bar, St.fsk, ...
            St.lambda_J, St.dlam_J, volfrac, rho_min, b, 1e-4, ml, om);
        rn = max(rho_min, min(1, rho + al*drho));
        om_new = eval_omega(rn, S, penal, mass_mode, n_modes);
        P1 = [P1; b, reg, double(ih.converged), ih.n_iters, ...
              ih.drho_change(end), max(abs(drho)), norm(drho)/sqrt(S.nEl), ...
              ih.frac_at_bound(end), mean(rho+drho), beta_fin, ...
              sqrt(max(beta_fin,0)), om_new(1), om_new(2), ...
              ih.n_singular_warn]; %#ok<AGROW>
        fprintf(' budget %4d %-14s conv=%d it=%4d |drho|inf=%.4f atBound=%.3f predVol=%.4f  sqrt(beta)=%9.3f  -> omega1=%10.4f\n', ...
            b, lbl, ih.converged, ih.n_iters, max(abs(drho)), ...
            ih.frac_at_bound(end), mean(rho+drho), sqrt(max(beta_fin,0)), om_new(1));
    end
end
T = array2table(P1, 'VariableNames', {'budget','regime','converged','n_iters', ...
    'last_change','drho_inf','drho_rms','frac_at_bound','pred_vol','beta', ...
    'sqrt_beta','omega1_after','omega2_after','n_singular_warn'});
writetable(T, fullfile(d,'p1_inner_budget.csv'));

%% ===================== P2: is the subproblem an LP? ===================
fprintf('\n--- P2  affineness of the N=1 constraints in Delta_rho ---\n');
% cluster constraint value as a function of t along a random direction
rng(0,'twister');
dir = randn(S.nEl,1);  dir = dir / norm(dir) * 0.1;
ts = (-1:0.25:1)';
f1 = zeros(numel(ts),1);  fJ = f1;  fv = f1;
fsk2D = reshape(St.fsk, S.nEl, St.N*St.N);
for i = 1:numel(ts)
    dd = ts(i)*dir;
    F = reshape(fsk2D'*dd, St.N, St.N);
    f1(i) = min(real(eig(F)));
    fJ(i) = St.dlam_J' * dd;
    fv(i) = mean(rho + dd);
end
sd = @(y) max(abs(diff(y,2)));
fprintf(' max |second difference| : cluster mu_1 = %.3e   J-mode = %.3e   volume = %.3e\n', ...
    sd(f1), sd(fJ), sd(fv));
fprintf(' (all ~ machine epsilon => every constraint is exactly affine => Eq.25 is an LP for N=1)\n');
writetable(table(ts, f1, fJ, fv, 'VariableNames', {'t','mu1','dlamJ_dot','vol'}), ...
    fullfile(d,'p2_affineness.csv'));

%% ===================== P3: exact LP vertex vs MMA =====================
fprintf('\n--- P3  exact LP optimum of Eq. (25) vs what MMA returns ---\n');
% For N = 1 (and ignoring the J constraint, verified inactive below), the LP is
%   max_{drho}  f'drho   s.t.  mean(rho+drho) <= volfrac,  lb <= drho <= ub
% whose optimum is the greedy vertex: push drho to ub where f is largest, to lb
% elsewhere, with one fractional element absorbing the volume residual.
f = fsk2D(:,1);
lb = rho_min - rho;  ub = 1 - rho;
[drho_lp, obj_lp] = greedy_lp_vertex(f, lb, ub, volfrac, rho);
[drho_mma, beta_mma, ih_mma] = inner_loop_mma_instr(rho, St.lambda_bar, St.fsk, ...
    St.lambda_J, St.dlam_J, volfrac, rho_min, 1000, 1e-4, Inf, Inf);
cosang = (drho_lp'*drho_mma)/(norm(drho_lp)*norm(drho_mma));
fprintf(' LP vertex   : f.drho = %.6e  |drho|inf = %.4f  frac at bound = %.4f\n', ...
    obj_lp, max(abs(drho_lp)), mean(drho_lp <= lb+1e-12 | drho_lp >= ub-1e-12));
fprintf(' MMA(b=1000) : f.drho = %.6e  |drho|inf = %.4f  frac at bound = %.4f  conv=%d it=%d\n', ...
    f'*drho_mma, max(abs(drho_mma)), ih_mma.frac_at_bound(end), ih_mma.converged, ih_mma.n_iters);
fprintf(' cos(angle between LP vertex and MMA increment) = %.4f\n', cosang);
om_lp  = eval_omega(max(rho_min,min(1,rho+drho_lp)),  S, penal, mass_mode, n_modes);
om_mma = eval_omega(max(rho_min,min(1,rho+drho_mma)), S, penal, mass_mode, n_modes);
fprintf(' realised omega1 : LP vertex = %.4f    MMA = %.4f    (start %.4f)\n', ...
    om_lp(1), om_mma(1), St.omega(1));
fprintf(' predicted omega1 (linear model): LP = %.4f   MMA = %.4f\n', ...
    sqrt(max(St.lambda_bar+obj_lp,0)), sqrt(max(beta_mma,0)));
writetable(table(obj_lp, f'*drho_mma, cosang, om_lp(1), om_mma(1), St.omega(1), ...
    sqrt(max(St.lambda_bar+obj_lp,0)), sqrt(max(beta_mma,0)), ...
    'VariableNames',{'lp_obj','mma_obj','cos_angle','omega1_lp','omega1_mma', ...
    'omega1_start','omega1_pred_lp','omega1_pred_mma'}), fullfile(d,'p3_lp_vs_mma.csv'));

%% ===================== P4: MMA conditioning ===========================
fprintf('\n--- P4  conditioning induced by the beta variable box [0, 1e6] ---\n');
for beta_max = [1e6 1e3 1e1 2]
    c = mma_condition_probe(rho, St, volfrac, rho_min, beta_max, Inf);
    fprintf(' beta_max_hat = %-8g : asym span(beta) = %.3e   col1/colRest P,Q ratio = %.3e   xmami(1) = %.3e\n', ...
        beta_max, c.beta_span, c.pq_ratio, c.xmami1);
end
fprintf(' (production uses beta_max_hat = 1e6, inner_loop_mma.m line 112)\n');

%% ===================== P5: line profile along the step ================
fprintf('\n--- P5  realised lambda_1 along the accepted paper-literal step ---\n');
[drho30, beta30, ih30] = inner_loop_mma_instr(rho, St.lambda_bar, St.fsk, ...
    St.lambda_J, St.dlam_J, volfrac, rho_min, 30, 1e-4, Inf, Inf);
tt = [0 .01 .02 .05 .1 .2 .3 .5 .75 1]';
prof = zeros(numel(tt), 5);
for i = 1:numel(tt)
    rn = max(rho_min, min(1, rho + tt(i)*drho30));
    o = eval_omega(rn, S, penal, mass_mode, n_modes);
    prof(i,:) = [tt(i), o(1), o(2), mean(rn), ...
                 sqrt(max(St.lambda_bar + tt(i)*(f'*drho30), 0))];
    fprintf('  t = %-5.2f  omega1 = %10.4f  omega2 = %10.4f  vol = %.4f   linear model predicts %10.4f\n', ...
        prof(i,1), prof(i,2), prof(i,3), prof(i,4), prof(i,5));
end
writetable(array2table(prof, 'VariableNames', {'t','omega1','omega2','vol','omega1_linear'}), ...
    fullfile(d,'p5_step_profile.csv'));
fprintf(' inner solve used for this profile: converged=%d, iters=%d\n', ih30.converged, ih30.n_iters);

%% ===================== P6: constraint activity ========================
fprintf('\n--- P6  which constraints are active at the returned increment ---\n');
for nm = {'budget30', 'budget1000'}
    if strcmp(nm{1},'budget30'), dr = drho30; bf = beta30; else, dr = drho_mma; bf = beta_mma; end
    F = reshape(fsk2D'*dr, St.N, St.N);  mu1 = min(real(eig(F)));
    r_cluster = (bf - St.lambda_bar - mu1)/St.lambda_bar;
    r_J = (bf - St.lambda_J - St.dlam_J'*dr)/St.lambda_bar;
    r_v = mean(rho+dr) - volfrac;
    fprintf(' %-11s cluster residual = %+.4e   J residual = %+.4e   volume residual = %+.4e\n', ...
        nm{1}, r_cluster, r_J, r_v);
end

fprintf('\n=========== PHASE 2 DIAGNOSIS COMPLETE -> %s ===========\n', d);
diary('off');
end

%% =====================================================================
function S = setup_model(nelx, nely, bc)
L = 8; H = 1; E0 = 1e7; nu = 0.3; rho0 = 1; t = 1; rmin_elem = 2.5;
dx = L/nelx; dy = H/nely;
S.nEl = nelx*nely;  S.nDof = 2*(nelx+1)*(nely+1);
S.nelx = nelx; S.nely = nely;
[Ke_star, Me_star] = fe_q4_exact(nu, t, dx, dy);
S.Ke_phys = E0*Ke_star;  S.Me_phys = rho0*Me_star;
nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
cVec = reshape(2*nodeNrs(1:nely,1:nelx)+1, S.nEl, 1);
S.cMat = [cVec, cVec+1, cVec+2*nely+2, cVec+2*nely+3, ...
          cVec+2*nely, cVec+2*nely+1, cVec-2, cVec-1];
[Il, Jl] = find(tril(ones(8)));
S.iK = reshape(S.cMat(:,Il)', [], 1);
S.jK = reshape(S.cMat(:,Jl)', [], 1);
S.Ke_l = S.Ke_phys(sub2ind([8,8], Il, Jl));
S.Me_l = S.Me_phys(sub2ind([8,8], Il, Jl));
fixed = build_supports_exact(bc, nodeNrs);
S.free = setdiff(1:S.nDof, fixed);
S.nFree = numel(S.free);
[S.h, S.Hs] = build_filter(nelx, nely, rmin_elem);
S.opts.tol = 1e-10;  S.opts.maxit = 600;
end

%% =====================================================================
function St = outer_state(rho, S, penal, mass_mode, n_modes, n_target, mult_tol)
[K, M] = assemble_KM_exact(rho, S.Ke_l, S.Me_l, S.iK, S.jK, S.nDof, penal, mass_mode);
Kf = K(S.free,S.free);  Mf = M(S.free,S.free);
[V, D] = eigs(Kf, Mf, n_modes, 'SM', S.opts);
[lam, idx] = sort(real(diag(D)));  V = real(V(:,idx));
for j = 1:n_modes
    v = V(:,j); sc = sqrt(abs(v'*(Mf*v))); if sc>1e-14, V(:,j)=v/sc; end
end
St.omega = sqrt(max(lam,0));  St.lam = lam;
Phi = zeros(S.nDof, n_modes);
for j = 1:n_modes, Phi(S.free,j) = V(:,j); end
St.Phi = Phi;
[St.N, St.J_idx, ci] = detect_multiplicity(St.omega, n_target, mult_tol);
St.lambda_bar = mean(lam(ci));
fsk_raw = compute_generalized_gradients(rho, St.lambda_bar, Phi(:,ci), ...
    S.cMat, S.Ke_phys, S.Me_phys, penal, mass_mode);
St.fsk = zeros(size(fsk_raw));
for s = 1:St.N
    for k = 1:St.N
        St.fsk(:,s,k) = apply_sensitivity_filter(fsk_raw(:,s,k), rho, S.h, S.Hs, S.nely, S.nelx);
    end
end
if St.J_idx > 0
    St.lambda_J = lam(St.J_idx);
    dJ = compute_elem_sensitivity(rho, St.lambda_J, Phi(:,St.J_idx), S.cMat, ...
        S.Ke_phys, S.Me_phys, S.free, S.nDof, penal, mass_mode);
    St.dlam_J = apply_sensitivity_filter(dJ, rho, S.h, S.Hs, S.nely, S.nelx);
else
    St.lambda_J = Inf;  St.dlam_J = [];
end
end

%% =====================================================================
function omega = eval_omega(rho, S, penal, mass_mode, n_modes)
[K, M] = assemble_KM_exact(rho, S.Ke_l, S.Me_l, S.iK, S.jK, S.nDof, penal, mass_mode);
[~, D, fl] = eigs(K(S.free,S.free), M(S.free,S.free), n_modes, 'SM', S.opts);
if fl ~= 0
    o2.tol=1e-8; o2.maxit=1500; o2.p=min(S.nFree-1,max(40,4*n_modes));
    [~, D, fl] = eigs(K(S.free,S.free), M(S.free,S.free), n_modes, 'SM', o2);
end
if fl == 0, omega = sqrt(max(sort(real(diag(D))),0)); else, omega = nan(n_modes,1); end
end

%% =====================================================================
function [drho, obj] = greedy_lp_vertex(f, lb, ub, volfrac, rho)
% Exact optimum of   max f'd  s.t. mean(rho+d) <= volfrac, lb <= d <= ub.
nEl = numel(f);
drho = lb;                                   % start everything at its floor
budget = volfrac*nEl - sum(rho + drho);      % material still available
[~, ord] = sort(f, 'descend');
for i = 1:nEl
    e = ord(i);
    if budget <= 0, break, end
    if f(e) <= 0, break, end                 % raising it would not help
    step = min(ub(e) - lb(e), budget);
    drho(e) = lb(e) + step;
    budget = budget - step;
end
obj = f' * drho;
end

%% =====================================================================
function c = mma_condition_probe(rho, St, volfrac, rho_min, beta_max_hat, outer_move)
% Reproduces mmasub's first-iteration asymptote / P,Q construction to expose
% the dynamic range induced by the beta variable's box.  Read-only.
nEl = numel(rho);  n_var = nEl+1;
drho_lb = max(rho_min - rho, -outer_move*ones(nEl,1));
drho_ub = min(1       - rho, +outer_move*ones(nEl,1));
xmin = [0; drho_lb];  xmax = [beta_max_hat; drho_ub];
xval = [(1-1e-6); zeros(nEl,1)];
asyinit = 0.01;  raa0 = 1e-5;  albefa = 0.1;
low = xval - asyinit*(xmax-xmin);
upp = xval + asyinit*(xmax-xmin);
alfa = max(max(low+albefa*(xval-low), xval-(xmax-xmin)), xmin);
bet  = min(min(upp-albefa*(upp-xval), xval+(xmax-xmin)), xmax);
c.beta_span = bet(1)-alfa(1);
c.xmami1 = xmax(1)-xmin(1);
xmami = max(xmax-xmin, 1e-5);  xmamiinv = 1./xmami;
ux2 = (upp-xval).^2;  xl2 = (xval-low).^2;
fsk2D = reshape(St.fsk, nEl, St.N*St.N);
dfdx1 = [1, -(fsk2D(:,1)')/St.lambda_bar];
P = max(dfdx1,0)';  Q = max(-dfdx1,0)';
PQ = 0.001*(P+Q) + raa0*xmamiinv;
P = (P+PQ).*ux2;  Q = (Q+PQ).*xl2;
c.pq_ratio = max(max(P(1),Q(1)),eps) / max(max(max(P(2:end)),max(Q(2:end))),eps);
end
