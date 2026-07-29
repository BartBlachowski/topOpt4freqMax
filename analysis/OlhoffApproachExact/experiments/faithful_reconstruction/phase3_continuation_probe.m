function phase3_continuation_probe(nelx, nely, bc)
% PHASE3_CONTINUATION_PROBE  Can penalization continuation avert the
%                            paper-literal collapse?  (question C1)
%
%   For each SIMP penalization power p on the continuation path 1 -> 3, take
%   the uniform rho = 0.5 initial design, build the paper-literal Eq.(25)
%   subproblem at that p, solve the inner loop to full convergence (budget
%   4000, so the returned increment is genuinely the converged one), apply the
%   full paper update rho := rho + Delta_rho, and measure the realised omega_1.
%
%   If continuation were the missing ingredient, the collapse should disappear
%   at the low-p end of the path.  Read-only; nothing is modified.
%
%   Also profiles omega_1(rho + t*Delta_rho) for t in [0,1] at each p, which
%   locates the largest step for which the linear model still holds.

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);
addpath(fullfile(this_dir, '..', '..', 'Matlab'));
addpath(fullfile(this_dir, '..', '..', '..', '..', 'tools', 'Matlab'));
if nargin < 3 || isempty(bc), bc = 'CC'; end

tag = sprintf('phase3_%s_%dx%d', upper(bc), nelx, nely);
d = fullfile(this_dir, 'results', tag);
if ~exist(d,'dir'), mkdir(d); end
diary(fullfile(d,'log.txt')); cleanup = onCleanup(@() diary('off'));

fprintf('\n====== PHASE 3 CONTINUATION PROBE  %s %dx%d ======\n', upper(bc), nelx, nely);

S = setup_model(nelx, nely, bc);
volfrac = 0.5; rho_min = 1e-3; mass_mode = 'du2007_c1';
n_modes = 4;  n_target = 1;  mult_tol = 1e-3;
rho0 = volfrac*ones(S.nEl,1);

p_list = [1 1.5 2 2.5 3];
rows = [];
prof_rows = [];
fprintf('\n%-5s %-10s %-4s %-10s | converged paper-literal step (move=Inf, alpha=1)\n', ...
    'p', 'omega1(0)', 'N', 'g12');
for p = p_list
    St = outer_state(rho0, S, p, mass_mode, n_modes, n_target, mult_tol);
    g12 = abs(St.omega(2)-St.omega(1))/St.omega(1);

    % paper-literal, fully converged inner solve
    [drA, bA, ihA] = inner_loop_mma_instr(rho0, St.lambda_bar, St.fsk, St.lambda_J, ...
        St.dlam_J, volfrac, rho_min, 4000, 1e-4, Inf, Inf);
    rA = max(rho_min, min(1, rho0 + drA));
    oA = eval_omega(rA, S, p, mass_mode, n_modes);
    % same design re-evaluated at p = 3 for a common physical yardstick
    oA3 = eval_omega(rA, S, 3.0, mass_mode, n_modes);

    % Regime-B move-limited, fully converged inner solve, for contrast
    [drB, bB, ihB] = inner_loop_mma_instr(rho0, St.lambda_bar, St.fsk, St.lambda_J, ...
        St.dlam_J, volfrac, rho_min, 4000, 1e-4, 0.2, 0.2);
    rB = max(rho_min, min(1, rho0 + 0.5*drB));
    oB = eval_omega(rB, S, p, mass_mode, n_modes);

    fprintf('%-5.2g %-10.4f %-4d %-10.3e | paper-literal: conv=%d it=%4d  |drho|inf=%.3f  sqrt(beta)=%9.2f -> omega1=%10.4f (at p=3: %8.3f)\n', ...
        p, St.omega(1), St.N, g12, ihA.converged, ihA.n_iters, max(abs(drA)), ...
        sqrt(max(bA,0)), oA(1), oA3(1));
    fprintf('%-5s %-10s %-4s %-10s | move-limited : conv=%d it=%4d  |drho|inf=%.3f  sqrt(beta)=%9.2f -> omega1=%10.4f\n', ...
        '', '', '', '', ihB.converged, ihB.n_iters, max(abs(drB)), sqrt(max(bB,0)), oB(1));

    rows = [rows; p, St.omega(1), St.N, g12, St.lambda_bar, ...
            double(ihA.converged), ihA.n_iters, max(abs(drA)), bA, oA(1), oA(2), oA3(1), ...
            double(ihB.converged), ihB.n_iters, max(abs(drB)), bB, oB(1), oB(2)]; %#ok<AGROW>

    % step-length profile along the paper-literal direction
    fsk2D = reshape(St.fsk, S.nEl, St.N*St.N);
    for t = [0 .05 .1 .15 .2 .25 .3 .4 .5 .75 1]
        rn = max(rho_min, min(1, rho0 + t*drA));
        o = eval_omega(rn, S, p, mass_mode, n_modes);
        lin = sqrt(max(St.lambda_bar + t*(fsk2D(:,1)'*drA), 0));
        prof_rows = [prof_rows; p, t, o(1), o(2), lin, mean(rn)]; %#ok<AGROW>
    end
end

writetable(array2table(rows, 'VariableNames', {'p','omega1_start','N','g12', ...
    'lambda_bar','A_converged','A_iters','A_drho_inf','A_beta','A_omega1', ...
    'A_omega2','A_omega1_at_p3','B_converged','B_iters','B_drho_inf','B_beta', ...
    'B_omega1','B_omega2'}), fullfile(d,'continuation_first_step.csv'));
writetable(array2table(prof_rows, 'VariableNames', {'p','t','omega1','omega2', ...
    'omega1_linear','vol'}), fullfile(d,'continuation_step_profiles.csv'));

fprintf('\n--- largest step t for which the linear model is within 5%% ---\n');
for p = p_list
    m = prof_rows(:,1) == p;
    tt = prof_rows(m,2);  oo = prof_rows(m,3);  ll = prof_rows(m,5);
    ok = abs(oo-ll) <= 0.05*max(ll, eps);
    fprintf(' p = %-4.2g : t_max = %.2f   (omega1 there = %.2f, linear model %.2f)\n', ...
        p, max(tt(ok)), oo(find(ok,1,'last')), ll(find(ok,1,'last')));
end

fprintf('\n====== PHASE 3 PROBE COMPLETE -> %s ======\n', d);
diary('off');
end

%% =====================================================================
function S = setup_model(nelx, nely, bc)
L = 8; H = 1; E0 = 1e7; nu = 0.3; rho0 = 1; t = 1; rmin_elem = 2.5;
dx = L/nelx; dy = H/nely;
S.nEl = nelx*nely;  S.nDof = 2*(nelx+1)*(nely+1);
S.nelx = nelx; S.nely = nely;
[Ks, Ms] = fe_q4_exact(nu, t, dx, dy);
S.Ke_phys = E0*Ks;  S.Me_phys = rho0*Ms;
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
S.free = setdiff(1:S.nDof, fixed);  S.nFree = numel(S.free);
[S.h, S.Hs] = build_filter(nelx, nely, rmin_elem);
S.opts.tol = 1e-10;  S.opts.maxit = 600;
end

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
[St.N, St.J_idx, ci] = detect_multiplicity(St.omega, n_target, mult_tol);
St.lambda_bar = mean(lam(ci));
fr = compute_generalized_gradients(rho, St.lambda_bar, Phi(:,ci), S.cMat, ...
    S.Ke_phys, S.Me_phys, penal, mass_mode);
St.fsk = zeros(size(fr));
for s = 1:St.N
    for k = 1:St.N
        St.fsk(:,s,k) = apply_sensitivity_filter(fr(:,s,k), rho, S.h, S.Hs, S.nely, S.nelx);
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

function omega = eval_omega(rho, S, penal, mass_mode, n_modes)
[K, M] = assemble_KM_exact(rho, S.Ke_l, S.Me_l, S.iK, S.jK, S.nDof, penal, mass_mode);
[~, D, fl] = eigs(K(S.free,S.free), M(S.free,S.free), n_modes, 'SM', S.opts);
if fl ~= 0
    o.tol=1e-8; o.maxit=1500; o.p=min(S.nFree-1,max(40,4*n_modes));
    [~, D, fl] = eigs(K(S.free,S.free), M(S.free,S.free), n_modes, 'SM', o);
end
if fl == 0, omega = sqrt(max(sort(real(diag(D))),0)); else, omega = nan(n_modes,1); end
end
