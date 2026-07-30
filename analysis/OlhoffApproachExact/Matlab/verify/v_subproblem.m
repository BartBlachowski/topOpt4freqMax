function ok = v_subproblem()
% V_SUBPROBLEM  Phase 4: validation of the increment-subproblem solvers.
%
%   Tests V-I1 .. V-I6 of PLAN_Olhoff2014_exact.md section "Phase 4.4".
%
%   V-I1  N = 1: MMA vs the exact LP.  Objective and direction must agree.
%   V-I2  N = 2, 3 with random symmetric F_e: the cutting-plane optimum must be
%         the true optimum of the LMI, checked independently by bisection on
%         beta with an eigenvalue feasibility test.
%   V-I3  declared-vs-achieved stopping: every solver reports a stop_reason.
%   V-I4  N = 1 reduction: the general-N code path with N = 1 must reproduce the
%         scalar formulation exactly (Eqs. 14, 15).
%   V-I5  basis invariance -- see v_basis_invariance.m.
%   V-I6  REQUIRED: with move = Inf and N = 1 the exact optimum of (19) is a
%         box vertex with essentially every Delta_rho_e at a bound, and taking
%         that step collapses omega_1.  This is the paper's own LP reduction
%         (section 2.5) and the reason the move limit exists.  If this test ever
%         stops failing to preserve the structure, the subproblem is not being
%         solved exactly.

fprintf('\n=== Phase 4: increment subproblem ===\n');
ok = true;
rng(11);

%% ---------------------------------------------------------------- V-I1
fprintf('\n V-I1  N = 1: MMA vs exact LP\n');
worst_dir = 1; worst_gap = 0;
for trial = 1:5
    sp = rand_sp(1, 400, 0.1);
    ex = subproblem_lp(sp);
    mm = subproblem_mma(sp, struct('max_iter', 400));
    gap = (ex.obj - mm.obj)/ex.lam_ref;
    c   = dot(ex.drho, mm.drho)/(norm(ex.drho)*norm(mm.drho));
    worst_dir = min(worst_dir, c);
    worst_gap = max(worst_gap, abs(gap));
end
fprintf('      min cos(drho_LP, drho_MMA) = %.6f, max scaled gap = %.3e   %s\n', ...
    worst_dir, worst_gap, pass(worst_dir > 0.99 && worst_gap < 1e-3));
ok = ok && worst_dir > 0.99 && worst_gap < 1e-3;

%% ---------------------------------------------------------------- V-I2
fprintf('\n V-I2  N = 2,3: cutting-plane optimum vs independent bisection\n');
worst = 0;
for N = [2 3]
    for trial = 1:4
        sp = rand_sp(N, 200, 0.1);
        ex = subproblem_lp(sp);
        bb = bisect_beta(sp);
        worst = max(worst, abs(ex.beta - bb)/ex.lam_ref);
    end
end
fprintf('      max |beta_cut - beta_bisect| / lam_ref = %.3e   %s\n', ...
    worst, pass(worst < 1e-6));
ok = ok && worst < 1e-6;

%% ---------------------------------------------------------------- V-I3
fprintf('\n V-I3  stop_reason always reported\n');
sp  = rand_sp(2, 200, 0.05);
r1  = subproblem_lp(sp);
r2  = subproblem_mma(sp, struct('max_iter', 15));
r3  = subproblem_mma(sp, struct('max_iter', 3000, 'tol', 1e-4));
good = ischar(r1.stop_reason) && ischar(r2.stop_reason) && ischar(r3.stop_reason) ...
       && strcmp(r2.stop_reason,'max_iter');
fprintf('      LP: %-14s  MMA(15): %-10s  MMA(3000,tol 1e-4): %-10s   %s\n', ...
    r1.stop_reason, r2.stop_reason, r3.stop_reason, pass(good));
ok = ok && good;

%% ---------------------------------------------------------------- V-I4
fprintf('\n V-I4  N = 1 reduction of the general-N path\n');
sp = rand_sp(1, 300, 0.1);
ex = subproblem_lp(sp);
% Solve the same thing as a plain scalar LP written out by hand.
nEl = numel(sp.rho);
f   = sp.up.Fe(:,1,1);
lb  = min(max(sp.rho_min - sp.rho, -sp.move), 0);
ub  = max(min(1 - sp.rho, sp.move), 0);
A   = [1, -f'; 1, -sp.up.guard.grad'; 0, ones(1,nEl)/nEl];
b   = [sp.up.L(1); sp.up.guard.lam; sp.volfrac - mean(sp.rho)];
big = max(sp.up.L(1), sp.up.guard.lam) + sum(abs(f).*max(abs(lb),abs(ub))) + 1;
x   = linprog([-1; zeros(nEl,1)], A, b, [], [], [0; lb], [big; ub], ...
              optimoptions('linprog','Display','none'));
rel = abs(x(1) - ex.beta)/max(abs(ex.beta),1e-30);
dd  = norm(x(2:end) - ex.drho)/max(norm(ex.drho),1e-30);
fprintf('      beta rel diff = %.3e, drho rel diff = %.3e   %s\n', ...
    rel, dd, pass(rel < 1e-9 && dd < 1e-7));
ok = ok && rel < 1e-9 && dd < 1e-7;

%% ---------------------------------------------------------------- V-I6
fprintf('\n V-I6  REQUIRED: unrestricted step is a box vertex that collapses omega_1\n');
cfg = struct('support_type','CC','nelx',80,'nely',10,'move',Inf, ...
             'outer_max_iter',1,'verbose',false,'subproblem_solver','lp');
[~, h] = topopt_freq_exact(cfg);
w0 = h.omega(1,1);
w1 = h.final_omega(1);
fb = h.frac_at_bound(1);
good = fb > 0.99 && w1 < 0.05*w0;
fprintf('      omega_1: %.4f -> %.6f (%.2f %% of initial), fraction at a bound = %.4f   %s\n', ...
    w0, w1, 100*w1/w0, fb, pass(good));
fprintf('      (this MUST hold: it is the paper LP reduction of section 2.5)\n');
ok = ok && good;

%% ------------------------------------------ same step with the move limit
fprintf('\n         control: identical setup with move = 0.05\n');
cfg.move = 0.05;
[~, h2] = topopt_freq_exact(cfg);
fprintf('      omega_1: %.4f -> %.4f (%.1f %% of initial), fraction at a bound = %.4f\n', ...
    h2.omega(1,1), h2.final_omega(1), 100*h2.final_omega(1)/h2.omega(1,1), ...
    h2.frac_at_bound(1));

fprintf('\n=== v_subproblem: %s ===\n\n', pass(ok));
end

%% =======================================================================
function sp = rand_sp(N, nEl, mv)
% Random but structurally realistic subproblem: eigenvalue scale ~1e5, gradient
% entries of both signs, an active volume constraint and a J-mode guard.
    rho = 0.2 + 0.6*rand(nEl,1);
    lam = 1e5*(1 + 0.1*rand);
    Fe  = zeros(nEl, N, N);
    for s = 1:N
        for k = s:N
            v = (lam/nEl) * (randn(nEl,1)*3 + (s==k)*2);
            Fe(:,s,k) = v;  Fe(:,k,s) = v;
        end
    end
    sp = struct('mode','nth','rho',rho,'volfrac',mean(rho),'rho_min',1e-3,'move',mv);
    sp.up = struct('L', lam*ones(N,1), 'Fe', Fe, ...
                   'guard', struct('lam', lam*1.6, 'grad', (lam/nEl)*randn(nEl,1)*3));
end

function beta = bisect_beta(sp)
% Independent optimum of  max beta  s.t.  G(drho) - beta I >= 0, guard, volume,
% box -- by bisection on beta, each feasibility test solved as an LP over the
% linearization at the incumbent plus eigenvector cuts until no violation.
% Uses no code from subproblem_lp beyond linprog.
    nEl = numel(sp.rho);
    N   = numel(sp.up.L);
    F2  = reshape(sp.up.Fe, nEl, N*N);
    lb  = min(max(sp.rho_min - sp.rho, -sp.move), 0);
    ub  = max(min(1 - sp.rho, sp.move), 0);
    lo  = min(sp.up.L);  hi = max(sp.up.L) + N*sum(max(abs(sp.up.Fe),[],[2 3]).*max(abs(lb),abs(ub)));
    opt = optimoptions('linprog','Display','none');
    for it = 1:80
        mid = 0.5*(lo+hi);
        if feasible(mid), lo = mid; else, hi = mid; end
    end
    beta = lo;

    function tf = feasible(bt)
        Q = eye(N);
        tf = false;
        for r = 1:30
            A = zeros(size(Q,2)+2, nEl);  b = zeros(size(Q,2)+2, 1);
            for j = 1:size(Q,2)
                q = Q(:,j);
                A(j,:) = -(F2*kron(q,q))';
                b(j)   = sum(sp.up.L(:).*q.^2) - bt;
            end
            A(end-1,:) = -sp.up.guard.grad';
            b(end-1)   = sp.up.guard.lam - bt;
            A(end,:)   = ones(1,nEl)/nEl;
            b(end)     = sp.volfrac - mean(sp.rho);
            [d, ~, ef] = linprog(zeros(nEl,1), A, b, [], [], lb, ub, opt);
            if ef ~= 1, return, end
            G = diag(sp.up.L(:)) + reshape(F2'*d, N, N);  G = (G+G')/2;
            [Vv, Dd] = eig(G);
            [mu, k]  = min(real(diag(Dd)));
            if mu >= bt - 1e-9*max(1,abs(bt)), tf = true; return, end
            Q = [Q, Vv(:,k)/norm(Vv(:,k))]; %#ok<AGROW>
        end
    end
end

function s = pass(tf)
    if tf, s = 'PASS'; else, s = 'FAIL'; end
end
