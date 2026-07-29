function phase1_trace(nelx, nely, bc, regime, n_iters)
% PHASE1_TRACE  Explicit, narrated algorithmic trace of the outer/inner loops.
%
%   phase1_trace(nelx, nely, bc, regime, n_iters)
%
%   Prints, for each outer iteration: the inputs, the construction of the
%   incremental problem (Eq. 25), the exact arrays handed to mmasub, the inner
%   convergence status, the returned increment, any volume correction, the
%   accepted outer update, the recomputed eigenvalues and the updated
%   multiplicity N.  Read-only; uses the production kernels.
%
%   regime : 'A' paper-literal (move_lim Inf, outer_move Inf, alpha 1)
%            'B' stabilized    (move_lim 0.2, outer_move 0.2, alpha 0.5)
%
%   Writes results/phase1_trace_<BC>_<mesh>_<regime>.txt

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);
addpath(fullfile(this_dir, '..', '..', 'Matlab'));
addpath(fullfile(this_dir, '..', '..', '..', '..', 'tools', 'Matlab'));
if nargin < 3 || isempty(bc), bc = 'CC'; end
if nargin < 4 || isempty(regime), regime = 'A'; end
if nargin < 5 || isempty(n_iters), n_iters = 3; end

if upper(regime) == 'A'
    move_lim = Inf;  outer_move = Inf;  alpha = 1.0;  inner_max = 30;
else
    move_lim = 0.2;  outer_move = 0.2;  alpha = 0.5;  inner_max = 30;
end

f = fullfile(this_dir, 'results', ...
    sprintf('phase1_trace_%s_%dx%d_regime%s.txt', upper(bc), nelx, nely, upper(regime)));
diary(f);  cleanup = onCleanup(@() diary('off'));

volfrac = 0.5; rho_min = 1e-3; penal = 3.0; mass_mode = 'du2007_c1';
n_modes = 4; n_target = 1; mult_tol = 1e-3; inner_tol = 1e-4; rmin_elem = 2.5;

fprintf('================================================================\n');
fprintf(' ALGORITHMIC TRACE  Du & Olhoff (2007) Fig. 1 procedure\n');
fprintf(' %s  %dx%d  regime %s  (move_lim=%g outer_move=%g alpha=%g)\n', ...
    upper(bc), nelx, nely, upper(regime), move_lim, outer_move, alpha);
fprintf(' penal=%g mass_mode=%s rmin_elem=%g volfrac=%g rho_min=%g\n', ...
    penal, mass_mode, rmin_elem, volfrac, rho_min);
fprintf(' n_modes=%d n_target=%d mult_tol=%g inner_max_iter=%d inner_tol=%g\n', ...
    n_modes, n_target, mult_tol, inner_max, inner_tol);
fprintf('================================================================\n');

L=8; H=1; E0=1e7; nu=0.3; rho0m=1; th=1;
dx=L/nelx; dy=H/nely; nEl=nelx*nely; nDof=2*(nelx+1)*(nely+1);
[Ks,Ms] = fe_q4_exact(nu,th,dx,dy);
Ke_phys = E0*Ks;  Me_phys = rho0m*Ms;
nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
cVec = reshape(2*nodeNrs(1:nely,1:nelx)+1, nEl, 1);
cMat = [cVec, cVec+1, cVec+2*nely+2, cVec+2*nely+3, cVec+2*nely, cVec+2*nely+1, cVec-2, cVec-1];
[Il,Jl] = find(tril(ones(8)));
iK = reshape(cMat(:,Il)',[],1);  jK = reshape(cMat(:,Jl)',[],1);
Ke_l = Ke_phys(sub2ind([8,8],Il,Jl));  Me_l = Me_phys(sub2ind([8,8],Il,Jl));
fixed = build_supports_exact(bc, nodeNrs);
free = setdiff(1:nDof, fixed);
[hf, Hsf] = build_filter(nelx, nely, rmin_elem);
opts.tol=1e-10; opts.maxit=600;

fprintf('\n[MESH]  nEl=%d  nDof=%d  fixed DOF=%d  free DOF=%d  dx=%g dy=%g\n', ...
    nEl, nDof, numel(fixed), numel(free), dx, dy);

rho = volfrac*ones(nEl,1);
fprintf('[INIT]  rho_e = volfrac = %g for all %d elements (paper Section 4.1)\n', volfrac, nEl);

for it = 1:n_iters
fprintf('\n\n################ OUTER ITERATION %d ################\n', it);
fprintf('\n-- INPUT STATE --\n');
fprintf('   rho : min=%.6f  max=%.6f  mean=%.6f  ||rho||=%.4f\n', ...
    min(rho), max(rho), mean(rho), norm(rho));

fprintf('\n-- STEP 1 (Fig.1 box 1): assemble K,M from Eq.(3); solve Eq.(7b) --\n');
rho_phys = max(rho_min, min(1, rho));
fprintf('   physical density = design density (sensitivity filter => no density filter)\n');
fprintf('   K_e = rho_e^%g * Ke*    M_e = m_%s(rho_e) * Me*\n', penal, mass_mode);
[K,M] = assemble_KM_exact(rho_phys, Ke_l, Me_l, iK, jK, nDof, penal, mass_mode);
Kf = K(free,free); Mf = M(free,free);
[V,D,flag] = eigs(Kf, Mf, n_modes, 'SM', opts);
[lam, ix] = sort(real(diag(D))); V = real(V(:,ix));
for j=1:n_modes, v=V(:,j); sc=sqrt(abs(v'*(Mf*v))); if sc>1e-14, V(:,j)=v/sc; end, end
omega = sqrt(max(lam,0));
Phi = zeros(nDof,n_modes); for j=1:n_modes, Phi(free,j)=V(:,j); end
fprintf('   eigs flag = %d\n', flag);
fprintf('   omega  = [%s]  (rad/s)\n', sprintf('%.4f  ', omega));
fprintf('   lambda = [%s]\n', sprintf('%.6e  ', lam));
fprintf('   M-orthonormality check max|Phi''M Phi - I| = %.3e\n', ...
    max(max(abs(V'*(Mf*V) - eye(n_modes)))));

fprintf('\n-- STEP 1b: multiplicity detection (Section 3.5.1) --\n');
[N, J_idx, ci] = detect_multiplicity(omega, n_target, mult_tol);
g12 = abs(omega(2)-omega(1))/max(omega(1),eps);
fprintf('   relative gap |w2-w1|/w1 = %.6e     mult_tol = %g\n', g12, mult_tol);
fprintf('   => N = %d   cluster indices = [%s]   J = n+N = %d\n', N, num2str(ci), J_idx);
lambda_bar = mean(lam(ci));
fprintf('   lambda_bar (cluster mean) = %.6e   [paper Eq.(25c) uses the individual w_j^2;\n', lambda_bar);
fprintf('     the reconstruction substitutes their mean, identical for N=1]\n');

fprintf('\n-- STEP 2 (Fig.1 box 2): generalized gradients, Eq.(19) --\n');
fsk_raw = compute_generalized_gradients(rho_phys, lambda_bar, Phi(:,ci), cMat, ...
    Ke_phys, Me_phys, penal, mass_mode);
fprintf('   f_sk[e] = phi_s''(K''_e - lambda_bar M''_e) phi_k ,  size = %s\n', mat2str(size(fsk_raw)));
fprintf('   raw   : min=%.4e max=%.4e ||.||=%.4e\n', min(fsk_raw(:)), max(fsk_raw(:)), norm(fsk_raw(:)));
fsk = zeros(size(fsk_raw));
for s=1:N, for k=1:N
    fsk(:,s,k) = apply_sensitivity_filter(fsk_raw(:,s,k), rho_phys, hf, Hsf, nely, nelx);
end, end
fprintf('   filtered (Sigmund 1997, applied to sensitivities as the paper states):\n');
fprintf('           min=%.4e max=%.4e ||.||=%.4e\n', min(fsk(:)), max(fsk(:)), norm(fsk(:)));
if J_idx > 0
    lambda_J = lam(J_idx);
    dJr = compute_elem_sensitivity(rho_phys, lambda_J, Phi(:,J_idx), cMat, Ke_phys, ...
        Me_phys, free, nDof, penal, mass_mode);
    dlam_J = apply_sensitivity_filter(dJr, rho_phys, hf, Hsf, nely, nelx);
    fprintf('   J-mode: lambda_J = %.6e  (lambda_J/lambda_bar = %.4f)\n', lambda_J, lambda_J/lambda_bar);
    fprintf('           grad lambda_J filtered: min=%.4e max=%.4e\n', min(dlam_J), max(dlam_J));
else
    lambda_J = Inf; dlam_J = [];
    fprintf('   J-mode: cluster reaches the end of the computed window; constraint (25b) omitted\n');
end

fprintf('\n-- STEP 3 (Fig.1 box 3): incremental subproblem Eq.(25), solved by MMA --\n');
fprintf('   variables x = [beta_hat ; Delta_rho]   n_var = %d\n', nEl+1);
fprintf('   objective  min -beta_hat            (Eq. 25a: max beta)\n');
fprintf('   constraints m = %d:\n', N + 1 + double(isfinite(lambda_J) && ~isempty(dlam_J)));
fprintf('     [1..%d] cluster (25c): beta_hat - 1 - mu_i(F(Delta_rho))/lambda_bar <= 0\n', N);
if J_idx > 0
    fprintf('     [%d]    J-mode  (25b): beta_hat - lambda_J/lambda_bar - grad_J''Drho/lambda_bar <= 0\n', N+1);
end
fprintf('     [%d]    volume  (25e): mean(rho + Delta_rho) - volfrac <= 0\n', N + 1 + double(J_idx>0));
lbv = max(rho_min - rho, -outer_move*ones(nEl,1));
ubv = min(1        - rho, +outer_move*ones(nEl,1));
fprintf('   bounds  (25f): Delta_rho in [%.4f, %.4f] elementwise\n', min(lbv), max(ubv));
fprintf('           beta_hat in [0, 1e6]   <-- RECONSTRUCTION: the paper imposes no\n');
fprintf('           bound on beta; 1e6 is a "large inactive" surrogate.  Because mmasub\n');
fprintf('           derives its asymptotes from (xmax-xmin), this choice sets the beta\n');
fprintf('           asymptote span to 0.01*1e6 = 1e4 and is NOT numerically inert.\n');
fprintf('   MMA constants passed to mmasub: a0=1, a=0, c=1e3, d=1 (all reconstruction)\n');
fprintf('   asymptotes are REINITIALISED every outer iteration (mmasub iter counter =\n');
fprintf('     inner iteration index), i.e. no MMA state persists across outer steps.\n');

[drho, beta_fin, ih] = inner_loop_mma_instr(rho, lambda_bar, fsk, lambda_J, dlam_J, ...
    volfrac, rho_min, inner_max, inner_tol, move_lim, outer_move);
fprintf('\n   inner loop: %d iterations, converged = %d, reason = %s\n', ...
    ih.n_iters, ih.converged, ih.termination_reason);
fprintf('   inner convergence test: ||Delta_rho_new - Delta_rho_old|| < inner_tol*sqrt(nEl) = %.4e\n', ...
    inner_tol*sqrt(nEl));
if ih.converged, cvg_txt = 'MET'; else, cvg_txt = 'NOT MET'; end
fprintf('   last change = %.4e  ->  %s\n', ih.drho_change(end), cvg_txt);
fprintf('   asymptote span over Delta_rho vars at last inner iter: [%.4e, %.4e]\n', ...
    ih.asym_width_min(end), ih.asym_width_max(end));
fprintf('   beta asymptotes at last inner iter: low=%.4e upp=%.4e\n', ...
    ih.asym_beta_low(end), ih.asym_beta_upp(end));
fprintf('   MMA multipliers: max lam = %.4e   max y_i = %.4e   z = %.4e\n', ...
    ih.mma_lam_max(end), ih.mma_ymma_max(end), ih.mma_zmma(end));
fprintf('   singular / RCOND warnings inside mmasub this outer step: %d\n', ih.n_singular_warn);

fprintf('\n   returned increment Delta_rho:\n');
fprintf('     min=%+.6f  max=%+.6f  ||.||inf=%.6f  ||.||/sqrt(nEl)=%.6f\n', ...
    min(drho), max(drho), max(abs(drho)), norm(drho)/sqrt(nEl));
fprintf('     fraction at a box bound (1e-6 rel) = %.4f   within 1%% of a bound = %.4f\n', ...
    ih.frac_at_bound(end), ih.frac_near_bound(end));
fprintf('     beta = %.6e  ->  sqrt(beta) = %.4f rad/s   (predicted new omega_1)\n', ...
    beta_fin, sqrt(max(beta_fin,0)));
fprintf('     predicted dlambda = beta - lambda_bar = %.6e\n', beta_fin - lambda_bar);

fprintf('\n-- VOLUME HANDLING --\n');
fprintf('   predicted mean(rho + Delta_rho) = %.8f   (volfrac = %.4f, residual %+.3e)\n', ...
    mean(rho+drho), volfrac, mean(rho+drho)-volfrac);
fprintf('   NO explicit volume projection or correction is applied anywhere: the volume\n');
fprintf('   constraint is enforced only inside the MMA subproblem, and only up to MMA''s\n');
fprintf('   artificial-variable relaxation (c = 1e3).\n');

fprintf('\n-- STEP 4 (Fig.1 box 4): outer update --\n');
fprintf('   paper: rho := rho + Delta_rho.  Reconstruction applies rho := rho + alpha*Delta_rho\n');
fprintf('   with alpha = %g, then clamps to [rho_min, 1].\n', alpha);
rho_new = max(rho_min, min(1, rho + alpha*drho));
fprintf('   ||rho_new - rho||inf = %.6f   ||rho_new - rho||/sqrt(nEl) = %.6e\n', ...
    max(abs(rho_new-rho)), norm(rho_new-rho)/sqrt(nEl));
fprintf('   actual mean(rho_new) = %.8f\n', mean(rho_new));

fprintf('\n-- ACCEPTANCE --\n');
fprintf('   Production topopt_freq_exact accepts this update UNCONDITIONALLY on the\n');
fprintf('   default path (acceptance_check = false, globalization_enabled = false).\n');
fprintf('   The inner convergence flag is recorded in hist but never consulted.\n');
fc_ok = ih.converged && all(isfinite(drho)) && ...
    all(drho >= lbv-1e-9 & drho <= ubv+1e-9) && mean(rho+drho) <= volfrac+1e-4;
if fc_ok, fc_txt = 'WOULD ACCEPT'; else, fc_txt = 'WOULD REJECT'; end
fprintf('   fail-closed predicate on this step: inner_converged=%d, increment finite=%d,\n', ...
    ih.converged, all(isfinite(drho)));
fprintf('     bounds respected=%d, predicted volume within 1e-4=%d  ==> %s\n', ...
    all(drho >= lbv-1e-9 & drho <= ubv+1e-9), mean(rho+drho) <= volfrac+1e-4, fc_txt);

fprintf('\n-- RECOMPUTED SPECTRUM AFTER THE UPDATE --\n');
[K2,M2] = assemble_KM_exact(rho_new, Ke_l, Me_l, iK, jK, nDof, penal, mass_mode);
[~,D2,f2] = eigs(K2(free,free), M2(free,free), n_modes, 'SM', opts);
lam2 = sort(real(diag(D2)));  om2 = sqrt(max(lam2,0));
[N2,~,~] = detect_multiplicity(om2, n_target, mult_tol);
fprintf('   eigs flag = %d\n', f2);
fprintf('   omega_new = [%s]\n', sprintf('%.4f  ', om2));
fprintf('   realised dlambda = %.6e   predicted = %.6e   ratio = %.4f\n', ...
    lam2(1)-lam(1), beta_fin-lambda_bar, (lam2(1)-lam(1))/(beta_fin-lambda_bar));
fprintf('   updated multiplicity N = %d   new gap = %.6e\n', N2, ...
    abs(om2(2)-om2(1))/max(om2(1),eps));

fprintf('\n-- STOPPING TEST (Fig.1 box: ||Delta_rho|| < eps) --\n');
fprintf('   ||rho_new - rho||/sqrt(nEl) = %.6e   vs outer_tol\n', norm(rho_new-rho)/sqrt(nEl));
fprintf('   RECONSTRUCTION: the paper writes ||Delta_rho|| < eps with neither the norm\n');
fprintf('   nor eps specified; the code uses the RMS norm and eps = 1e-4 (regime A) or\n');
fprintf('   1e-6 (regime B).\n');

rho = rho_new;
end

fprintf('\n================ TRACE COMPLETE ================\n');
diary('off');
end
