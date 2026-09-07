function q = subproblem_kkt(sp, cand)
% SUBPROBLEM_KKT  Feasibility and true suboptimality of an increment-subproblem point.
%
%   q = subproblem_kkt(sp, cand)
%
%   Quality certificate for a candidate solution of Olhoff & Du (2014)
%   subproblem (19)/(20).  `cand` is any struct with fields .drho, .beta and
%   (gap mode) .beta1 -- e.g. the output of SUBPROBLEM_MMA or SUBPROBLEM_LP.
%
%   Because the subproblem is CONVEX and SUBPROBLEM_LP solves it exactly (see
%   that file's header), the honest measure of inner-loop quality is not a KKT
%   residual proxy but the TRUE optimality gap against the exact optimum.  This
%   routine reports both that gap and the constraint violations, so a run can
%   never again report an inner loop as "converged" when it merely ran out of
%   iterations.
%
%   OUTPUT (struct q)
%     .feas_cluster   max violation of the cluster LMI, relative to lam_ref
%                     (= beta - mu_min(G_up), and mu_max(G_lo) - beta_1)
%     .feas_guard     max violation of the guard constraints (19b)/(20e)
%     .feas_volume    volume constraint value (<= 0 feasible)
%     .feas_box       max box/move-limit violation
%     .max_violation  max of the four above, relative where applicable
%     .obj            candidate objective, physical units
%     .obj_exact      exact optimum from SUBPROBLEM_LP, physical units
%     .gap_abs        obj_exact - obj  (>= 0 up to feasibility slack)
%     .gap_rel        gap_abs / lam_ref
%     .exact          the full SUBPROBLEM_LP output (for reuse)
%
%   Reference: Olhoff & Du (2014), Eqs. (19), (20).

is_gap = strcmpi(sp.mode, 'gap');

rho  = sp.rho(:);
nEl  = numel(rho);
mlim = sp.move;  if isempty(mlim), mlim = Inf; end

lam_ref = max(abs(sp.up.L));
if ~(lam_ref > 0), lam_ref = 1; end

drho = cand.drho(:);
beta = cand.beta;

N     = numel(sp.up.L);
Feu2D = reshape(sp.up.Fe, nEl, N*N);
Gu    = diag(sp.up.L(:)) + reshape(Feu2D' * drho, N, N);
Gu    = (Gu + Gu')/2;
mu_min = min(real(eig(Gu)));
feas_cluster = (beta - mu_min) / lam_ref;

feas_guard = -Inf;
if isfield(sp.up,'guard') && ~isempty(sp.up.guard)
    feas_guard = max(feas_guard, ...
        (beta - sp.up.guard.lam - sp.up.guard.grad(:)' * drho) / lam_ref);
end

if is_gap
    beta1 = cand.beta1;
    R     = numel(sp.lo.L);
    Fel2D = reshape(sp.lo.Fe, nEl, R*R);
    Gl    = diag(sp.lo.L(:)) + reshape(Fel2D' * drho, R, R);
    Gl    = (Gl + Gl')/2;
    mu_max = max(real(eig(Gl)));
    feas_cluster = max(feas_cluster, (mu_max - beta1) / lam_ref);
    if isfield(sp.lo,'guard') && ~isempty(sp.lo.guard)
        feas_guard = max(feas_guard, ...
            (sp.lo.guard.lam + sp.lo.guard.grad(:)' * drho - beta1) / lam_ref);
    end
    obj = beta - beta1;
else
    obj = beta;
end

lb_d = min(max(sp.rho_min - rho, -mlim*ones(nEl,1)), 0);
ub_d = max(min(1          - rho, +mlim*ones(nEl,1)), 0);

q.feas_cluster  = feas_cluster;
q.feas_guard    = feas_guard;
q.feas_volume   = (sum(rho) + sum(drho))/nEl - sp.volfrac;
q.feas_box      = max([0; lb_d - drho; drho - ub_d]);
q.max_violation = max([q.feas_cluster, q.feas_guard, q.feas_volume, q.feas_box]);

exact       = subproblem_lp(sp);
q.obj       = obj;
q.obj_exact = exact.obj;
q.gap_abs   = exact.obj - obj;
q.gap_rel   = q.gap_abs / lam_ref;
q.exact     = exact;
end
