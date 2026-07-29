function phase6_bimodal_probe(run_tag)
% PHASE6_BIMODAL_PROBE  What happens at the one near-bimodal iterate?
%
%   phase6_bimodal_probe(run_tag)
%
%   Loads the design snapshots of a completed run, finds the iterate with the
%   smallest relative eigengap g12 = |w2-w1|/w1, and at THAT design:
%
%     A) reports the solver-reported multiplicity N and the independently
%        reconstructed multiplicity at several tolerances;
%     B) builds the Eq.(25) subproblem with N = 1, exactly as the solver did,
%        solves it to full inner convergence and applies the accepted step;
%     C) builds the SAME subproblem with the cluster FORCED to N = 2, i.e. the
%        full generalized-gradient array f_sk, s,k in {1,2}, plus the J-mode
%        constraint moved to mode 3, solves and applies it;
%     D) compares the realised spectra, gaps and mode identities.
%
%   This isolates "the code never detects the cluster" from "the code detects
%   the cluster but the N=2 subproblem does not retain it".  Read-only: the
%   production solver is untouched and the run being probed is not re-run.

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);
addpath(fullfile(this_dir, '..', '..', 'Matlab'));
addpath(fullfile(this_dir, '..', '..', '..', '..', 'tools', 'Matlab'));
if nargin < 1 || isempty(run_tag), run_tag = 'V4_CC_160x20_i2000'; end

rd = fullfile(this_dir, 'results', run_tag);
L = load(fullfile(rd, 'run.mat'));  out = L.out;
cfg = out.cfg;  snap = out.rho_snapshots;

od = fullfile(this_dir, 'results', ['phase6_' run_tag]);
if ~exist(od,'dir'), mkdir(od); end
diary(fullfile(od,'log.txt'));  cleanup = onCleanup(@() diary('off'));

fprintf('\n========= PHASE 6 BIMODALITY PROBE : %s =========\n', run_tag);

S = setup_model(cfg.nelx, cfg.nely, cfg.support_type);
volfrac = cfg.volfrac; rho_min = cfg.rho_min; mass_mode = cfg.mass_mode;
n_modes = cfg.n_modes; n_target = 1; mult_tol = cfg.mult_tol;
penal = 3.0;
move_lim = cfg.move_lim; outer_move = cfg.outer_move; alpha = cfg.alpha;

% ---- locate the minimum-gap accepted design ----
g = out.hist.g12_trial;  g(~isfinite(g)) = Inf;
[gmin, k] = min(g);
fprintf('\n[locate] minimum post-update gap g12 = %.6e at outer iteration %d\n', gmin, k);
rho = snap(:, k);
fprintf('[locate] design taken from rho_snapshots column %d, mean = %.6f\n', k, mean(rho));

St = outer_state(rho, S, penal, mass_mode, n_modes, n_target, mult_tol);
g12 = abs(St.omega(2)-St.omega(1))/St.omega(1);
fprintf('\n--- A) state at that design ---\n');
fprintf('  omega = [%s]\n', sprintf('%.4f  ', St.omega));
fprintf('  g12 = %.6e   g23 = %.6e\n', g12, abs(St.omega(3)-St.omega(2))/St.omega(2));
fprintf('  solver-reported N (mult_tol = %g) = %d,  J = %d\n', mult_tol, St.N, St.J_idx);
for t = [1e-4 1e-3 1.5e-3 2e-3 5e-3 1e-2 2e-2 5e-2]
    fprintf('    independently reconstructed N at tol %-8.1e = %d\n', t, ...
        detect_multiplicity(St.omega, n_target, t));
end
fprintf('  => the generalized-gradient basis actually used spans %d of the %d\n', ...
    St.N, sum(abs(St.omega-St.omega(1))/St.omega(1) <= 1e-2));
fprintf('     modes that lie within 1%% of omega_1: BASIS IS RANK-DEFICIENT w.r.t.\n');
fprintf('     the physically clustered pair whenever those two numbers differ.\n');

rows = {};
% ---- B) N = 1 subproblem, exactly as the solver built it ----
fprintf('\n--- B) N = 1 subproblem (what the solver actually solved) ---\n');
[rB, oB, iB] = do_step(rho, St, 1, S, penal, mass_mode, n_modes, volfrac, ...
    rho_min, move_lim, outer_move, alpha, mult_tol, n_target);
rows(end+1,:) = {'N=1 (as solved)', iB.n_iters, iB.converged, oB.om(1), oB.om(2), ...
                 oB.g12, oB.N, oB.mac(1), oB.mac(2)};

% ---- C) N = 2 subproblem, cluster forced ----
fprintf('\n--- C) N = 2 subproblem (cluster forced, full f_sk array) ---\n');
St2 = outer_state(rho, S, penal, mass_mode, n_modes, n_target, mult_tol, 2);
fprintf('  forced cluster indices = [1 2], lambda_bar = %.6e (was %.6e)\n', ...
    St2.lambda_bar, St.lambda_bar);
fprintf('  f_sk array size = %s;  off-diagonal ||f_12|| / diagonal ||f_11|| = %.4f\n', ...
    mat2str(size(St2.fsk)), norm(St2.fsk(:,1,2))/norm(St2.fsk(:,1,1)));
fprintf('  J-mode moved to mode %d, lambda_J/lambda_bar = %.4f\n', ...
    St2.J_idx, St2.lambda_J/St2.lambda_bar);
[rC, oC, iC] = do_step(rho, St2, 2, S, penal, mass_mode, n_modes, volfrac, ...
    rho_min, move_lim, outer_move, alpha, mult_tol, n_target);
rows(end+1,:) = {'N=2 (forced)', iC.n_iters, iC.converged, oC.om(1), oC.om(2), ...
                 oC.g12, oC.N, oC.mac(1), oC.mac(2)};

% ---- D) comparison ----
fprintf('\n--- D) comparison of the two accepted steps from the SAME design ---\n');
fprintf('%-16s %6s %5s %10s %10s %11s %4s %7s %7s\n', 'subproblem', 'innIt', ...
    'conv', 'omega1', 'omega2', 'g12 after', 'N', 'MAC11', 'MAC22');
for i = 1:size(rows,1)
    fprintf('%-16s %6d %5d %10.4f %10.4f %11.4e %4d %7.4f %7.4f\n', rows{i,:});
end
fprintf('\n  start: omega1 = %.4f  omega2 = %.4f  g12 = %.4e\n', ...
    St.omega(1), St.omega(2), g12);
fprintf('  increment similarity cos(Drho_N1, Drho_N2) = %.4f\n', ...
    (rB'*rC)/(norm(rB)*norm(rC)));
fprintf('  ||Drho_N1||inf = %.4f   ||Drho_N2||inf = %.4f\n', max(abs(rB)), max(abs(rC)));

T = cell2table(rows, 'VariableNames', {'subproblem','inner_iters','converged', ...
    'omega1_after','omega2_after','g12_after','N_after','MAC11','MAC22'});
writetable(T, fullfile(od,'bimodal_step_comparison.csv'));

% ---- E) retention test: iterate the N=2 treatment for a few steps ----
fprintf('\n--- E) retention: 8 further steps with the cluster re-detected each time,\n');
fprintf('       using a DIAGNOSTIC multiplicity tolerance of 1e-2 so that N=2 is\n');
fprintf('       actually engaged (sensitivity diagnostic only, not a primary result) ---\n');
r = rho;  ret = [];
fprintf('%4s %3s %10s %10s %11s %9s\n', 'step', 'N', 'omega1', 'omega2', 'g12', 'd_rms');
for st = 1:8
    Sx = outer_state(r, S, penal, mass_mode, n_modes, n_target, 1e-2);
    [dr, ~, ~] = inner_loop_mma_instr(r, Sx.lambda_bar, Sx.fsk, Sx.lambda_J, ...
        Sx.dlam_J, volfrac, rho_min, 2000, 1e-4, move_lim, outer_move);
    rn = max(rho_min, min(1, r + alpha*dr));
    ox = eval_omega(rn, S, penal, mass_mode, n_modes);
    gg = abs(ox(2)-ox(1))/max(ox(1),eps);
    fprintf('%4d %3d %10.4f %10.4f %11.4e %9.3e\n', st, Sx.N, ox(1), ox(2), gg, ...
        norm(rn-r)/sqrt(S.nEl));
    ret = [ret; st, Sx.N, ox(1), ox(2), gg, norm(rn-r)/sqrt(S.nEl)]; %#ok<AGROW>
    r = rn;
end
writetable(array2table(ret, 'VariableNames', {'step','N','omega1','omega2','g12','d_rms'}), ...
    fullfile(od,'bimodal_retention.csv'));

fprintf('\n========= PHASE 6 PROBE COMPLETE -> %s =========\n', od);
diary('off');
end

%% =====================================================================
function [drho, o, ih] = do_step(rho, St, N, S, penal, mass_mode, n_modes, ...
                                 volfrac, rho_min, move_lim, outer_move, alpha, ...
                                 mult_tol, n_target)
[drho, beta_fin, ih] = inner_loop_mma_instr(rho, St.lambda_bar, St.fsk, ...
    St.lambda_J, St.dlam_J, volfrac, rho_min, 2000, 1e-4, move_lim, outer_move);
fprintf('  inner: %d iterations, converged = %d, sqrt(beta) = %.4f\n', ...
    ih.n_iters, ih.converged, sqrt(max(beta_fin,0)));
rn = max(rho_min, min(1, rho + alpha*drho));
[om, Phi] = eval_modes(rn, S, penal, mass_mode, n_modes);
o.om = om;
o.g12 = abs(om(2)-om(1))/max(om(1),eps);
o.N = detect_multiplicity(om, n_target, mult_tol);
[K,M] = assemble_KM_exact(rn, S.Ke_l, S.Me_l, S.iK, S.jK, S.nDof, penal, mass_mode);
C = St.Phi(:,1:2)' * (M * Phi(:,1:2));
np_ = sqrt(abs(diag(St.Phi(:,1:2)'*(M*St.Phi(:,1:2)))));
nc_ = sqrt(abs(diag(Phi(:,1:2)'*(M*Phi(:,1:2)))));
MACm = (C.^2)./max((np_.^2)*(nc_.^2)', eps);
o.mac = [MACm(1,1), MACm(2,2)];
fprintf('  after step: omega = [%s]  g12 = %.4e  N = %d  MAC = [%.4f %.4f]\n', ...
    sprintf('%.4f  ', om), o.g12, o.N, o.mac(1), o.mac(2));
end

%% =====================================================================
function S = setup_model(nelx, nely, bc)
L=8; H=1; E0=1e7; nu=0.3; rho0=1; t=1; rmin_elem=2.5;
dx=L/nelx; dy=H/nely;
S.nEl=nelx*nely; S.nDof=2*(nelx+1)*(nely+1); S.nelx=nelx; S.nely=nely;
[Ks,Ms]=fe_q4_exact(nu,t,dx,dy);
S.Ke_phys=E0*Ks; S.Me_phys=rho0*Ms;
nodeNrs=reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
cVec=reshape(2*nodeNrs(1:nely,1:nelx)+1, S.nEl, 1);
S.cMat=[cVec, cVec+1, cVec+2*nely+2, cVec+2*nely+3, cVec+2*nely, cVec+2*nely+1, cVec-2, cVec-1];
[Il,Jl]=find(tril(ones(8)));
S.iK=reshape(S.cMat(:,Il)',[],1); S.jK=reshape(S.cMat(:,Jl)',[],1);
S.Ke_l=S.Ke_phys(sub2ind([8,8],Il,Jl)); S.Me_l=S.Me_phys(sub2ind([8,8],Il,Jl));
fixed=build_supports_exact(bc, nodeNrs);
S.free=setdiff(1:S.nDof, fixed); S.nFree=numel(S.free);
[S.h,S.Hs]=build_filter(nelx,nely,rmin_elem);
S.opts.tol=1e-10; S.opts.maxit=600;
end

function St = outer_state(rho, S, penal, mass_mode, n_modes, n_target, mult_tol, force_N)
[om, Phi, lam] = eval_modes(rho, S, penal, mass_mode, n_modes);
St.omega=om; St.lam=lam; St.Phi=Phi;
[St.N, St.J_idx, ci] = detect_multiplicity(om, n_target, mult_tol);
if nargin >= 8 && ~isempty(force_N)
    St.N = force_N; ci = n_target:(n_target+force_N-1);
    St.J_idx = n_target + force_N;
    if St.J_idx > n_modes, St.J_idx = 0; end
end
St.lambda_bar = mean(lam(ci));
fr = compute_generalized_gradients(rho, St.lambda_bar, Phi(:,ci), S.cMat, ...
    S.Ke_phys, S.Me_phys, penal, mass_mode);
St.fsk = zeros(size(fr));
for s=1:St.N, for k=1:St.N
    St.fsk(:,s,k) = apply_sensitivity_filter(fr(:,s,k), rho, S.h, S.Hs, S.nely, S.nelx);
end, end
if St.J_idx > 0
    St.lambda_J = lam(St.J_idx);
    dJ = compute_elem_sensitivity(rho, St.lambda_J, Phi(:,St.J_idx), S.cMat, ...
        S.Ke_phys, S.Me_phys, S.free, S.nDof, penal, mass_mode);
    St.dlam_J = apply_sensitivity_filter(dJ, rho, S.h, S.Hs, S.nely, S.nelx);
else
    St.lambda_J = Inf; St.dlam_J = [];
end
end

function [omega, Phi, lam] = eval_modes(rho, S, penal, mass_mode, nm)
[K,M] = assemble_KM_exact(rho, S.Ke_l, S.Me_l, S.iK, S.jK, S.nDof, penal, mass_mode);
Kf=K(S.free,S.free); Mf=M(S.free,S.free);
[V,D,fl]=eigs(Kf,Mf,nm,'SM',S.opts);
if fl~=0
    o.tol=1e-8; o.maxit=1500; o.p=min(S.nFree-1,max(40,4*nm));
    [V,D,fl]=eigs(Kf,Mf,nm,'SM',o);
end
[lam,ix]=sort(real(diag(D))); V=real(V(:,ix));
for j=1:nm, v=V(:,j); sc=sqrt(abs(v'*(Mf*v))); if sc>1e-14, V(:,j)=v/sc; end, end
omega=sqrt(max(lam,0));
Phi=zeros(S.nDof,nm); for j=1:nm, Phi(S.free,j)=V(:,j); end
end

function omega = eval_omega(rho, S, penal, mass_mode, nm)
[omega,~,~] = eval_modes(rho, S, penal, mass_mode, nm);
end
