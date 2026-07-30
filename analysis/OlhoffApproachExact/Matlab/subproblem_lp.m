function out = subproblem_lp(sp, opts)
% SUBPROBLEM_LP  EXACT solution of the Olhoff & Du (2014) increment subproblem.
%
%   out = subproblem_lp(sp)
%   out = subproblem_lp(sp, opts)
%
%   Solves problem (19) (maximization of the n-th eigenfrequency) or problem
%   (20) (maximization of the gap) exactly, by a cutting-plane LP scheme.
%
%   WHY THIS IS EXACT
%   -----------------
%   The generalized gradient vectors f_sk are symmetric (f_sk = f_ks, Eq. (13)),
%   so the N x N matrix
%
%       F(Delta_rho)(s,k) = f_sk' * Delta_rho
%
%   is symmetric and LINEAR in Delta_rho.  Constraint (19d) defines the
%   increments Delta(omega^2) as the eigenvalues of F, and (19c) imposes
%   beta <= omega_j^2 + Delta(omega_j^2) for ALL j in the cluster.  Together
%   these are equivalent to
%
%       beta <= mu_min( G(Delta_rho) ),      G = diag(L) + sum_e Delta_rho_e F_e
%
%   which, because mu_min of a symmetric matrix pencil that is affine in the
%   variables is a concave function, is exactly the linear matrix inequality
%
%       G(Delta_rho)  -  beta * I_N   >=  0     (positive semidefinite)
%
%   So subproblem (19) is a small SEMIDEFINITE PROGRAM: linear objective, one
%   N x N LMI, one linear volume inequality (19e), and box bounds (19f).  It is
%   convex.  For N = 1 the LMI is a single scalar linear constraint and (19) is
%   a plain LP -- which is exactly what the paper states in the last paragraph
%   of section 2.5 ("...both reduce to linear programming problems") and after
%   (20i) ("can be solved using the MMA method ... or a linear programming
%   algorithm").
%
%   Using mu_min(G) = min_{||q||=1} q' G q, the LMI is equivalent to the
%   infinite family of LINEAR constraints
%
%       beta - q'diag(L)q - sum_e Delta_rho_e (q' F_e q) <= 0    for all unit q
%
%   and is solved by cutting planes: solve the LP over a finite cut set, take
%   the eigenvector of the most violated direction, add it as a new cut, repeat.
%   Each cut has exactly the algebraic form of (19c), and its coefficient vector
%   q'F_e q is the same quantity used as the gradient of mu_i in the MMA path.
%
%   For the gap problem (20) the upper cluster contributes mu_min cuts bounding
%   beta_2 from above and the lower R-fold cluster contributes mu_max cuts
%   bounding beta_1 from below.
%
%   SUBPROBLEM SPECIFICATION  (struct sp)
%     .mode      'nth' | 'gap'
%     .rho       nEl x 1   current density
%     .volfrac   scalar    V*/V0 (uniform element volumes assumed)
%     .rho_min   scalar    lower density bound (Eq. (1e), (19f))
%     .move      scalar    move limit m added to (19f); Inf disables  [R1]
%     .up.L      N x 1     reference eigenvalues of the target cluster
%     .up.Fe     nEl x N x N   generalized gradients (generalized_gradients.m)
%     .up.guard  [] or struct('lam', scalar, 'grad', nEl x 1)
%                          the J = n+N constraint (19b)/(20b)
%     .lo.L      R x 1     gap mode only: lower cluster reference eigenvalues
%     .lo.Fe     nEl x R x R   gap mode only
%     .lo.guard  [] or struct(...)   the j = n-R-1 constraint (20e)
%
%   OPTIONS (struct opts, all optional)
%     .max_cuts  max cutting-plane rounds                    (default 40)
%     .cut_tol   relative LMI violation tolerance            (default 1e-9)
%     .verbose   print cut rounds                             (default false)
%
%   OUTPUT (struct out)
%     .drho          nEl x 1  optimal design increment
%     .beta          scalar   optimal beta (or beta_2), PHYSICAL units (rad/s)^2
%     .beta1         scalar   optimal beta_1 (gap mode), physical, else NaN
%     .obj           scalar   optimal objective, physical
%     .n_cuts        scalar   cutting planes added beyond the initial basis
%     .n_lp          scalar   LPs solved
%     .exitflag      scalar   last linprog exit flag (1 = optimal)
%     .stop_reason   char     'lmi_satisfied' | 'max_cuts' | 'lp_failed'
%     .lmi_violation scalar   final relative LMI violation
%     .frac_at_bound scalar   fraction of Delta_rho at a box/move bound
%     .beta_cap_active logical  true if the finite beta bound was binding
%                              (would indicate the safety cap is interfering)
%
%   Reference: Olhoff & Du (2014), Eqs. (12), (13), (19), (20); section 2.5.

if nargin < 2, opts = struct(); end
if ~isfield(opts,'max_cuts') || isempty(opts.max_cuts), opts.max_cuts = 40;   end
if ~isfield(opts,'cut_tol')  || isempty(opts.cut_tol),  opts.cut_tol  = 1e-9; end
if ~isfield(opts,'verbose')  || isempty(opts.verbose),  opts.verbose  = false;end

is_gap = strcmpi(sp.mode, 'gap');

rho  = sp.rho(:);
nEl  = numel(rho);
mlim = sp.move;
if isempty(mlim), mlim = Inf; end

% ---- Scaling: everything eigenvalue-like is divided by lam_ref so the LP is
%      posed on O(1) data.  Delta_rho is already O(1).
lam_ref = max(abs(sp.up.L));
if ~(lam_ref > 0), lam_ref = 1; end

N     = numel(sp.up.L);
Lu    = sp.up.L(:) / lam_ref;
Feu2D = reshape(sp.up.Fe, nEl, N*N) / lam_ref;

if is_gap
    R     = numel(sp.lo.L);
    Ll    = sp.lo.L(:) / lam_ref;
    Fel2D = reshape(sp.lo.Fe, nEl, R*R) / lam_ref;
else
    R = 0;
end

% ---- Variable layout ---------------------------------------------------
%   nth : x = [b2 ; drho]
%   gap : x = [b2 ; b1 ; drho]
nb    = 1 + double(is_gap);
i_b2  = 1;
i_b1  = 2;                       % gap only
i_d   = nb + (1:nEl);
nvar  = nb + nEl;

% ---- Box bounds (19f) intersected with the move limit [R1] -------------
lb_d = max(sp.rho_min - rho, -mlim * ones(nEl,1));
ub_d = min(1          - rho, +mlim * ones(nEl,1));
lb_d = min(lb_d, 0);             % keep Delta_rho = 0 feasible
ub_d = max(ub_d, 0);
span = max(abs(lb_d), abs(ub_d));

% ---- Finite safety caps on the bound variables -------------------------
%      beta is unbounded above in (19a); linprog needs a finite bound.  The cap
%      below is a valid over-estimate of any attainable value, so it must never
%      bind; out.beta_cap_active reports if it did.
b2cap = max(Lu) + N * (max(abs(Feu2D), [], 2)' * span) + 1;
if is_gap
    b1cap = max(Ll) + R * (max(abs(Fel2D), [], 2)' * span) + 1;
    b1cap = max(b1cap, b2cap);
end

lb = [zeros(nb,1); lb_d];
ub = [b2cap;       ub_d];
if is_gap, ub = [b2cap; b1cap; ub_d]; end

% ---- Objective: max b2  (nth)   /   max (b2 - b1)  (gap) ---------------
c        = zeros(nvar,1);
c(i_b2)  = -1;
if is_gap, c(i_b1) = +1; end

% ---- Fixed rows: guards (19b)/(20b)/(20e) and volume (19e)/(20h) -------
Afix = zeros(0, nvar);
bfix = zeros(0, 1);

if isfield(sp.up,'guard') && ~isempty(sp.up.guard)
    row          = zeros(1, nvar);
    row(i_b2)    = 1;
    row(i_d)     = -sp.up.guard.grad(:)' / lam_ref;
    Afix(end+1,:) = row;
    bfix(end+1,1) = sp.up.guard.lam / lam_ref;
end

if is_gap && isfield(sp.lo,'guard') && ~isempty(sp.lo.guard)
    row          = zeros(1, nvar);
    row(i_b1)    = -1;
    row(i_d)     = +sp.lo.guard.grad(:)' / lam_ref;
    Afix(end+1,:) = row;
    bfix(end+1,1) = -sp.lo.guard.lam / lam_ref;
end

% Volume (19e): sum_e (rho_e + drho_e)/nEl - volfrac <= 0
row       = zeros(1, nvar);
row(i_d)  = 1/nEl;
Afix(end+1,:) = row;
bfix(end+1,1) = sp.volfrac - sum(rho)/nEl;

% ---- Initial cut sets = the current eigenvector basis ------------------
%      With Q = I these cuts ARE constraint (19c) written in the basis the
%      eigensolver returned, i.e. the paper's constraint set verbatim.
Qu = eye(N);
if is_gap, Ql = eye(R); else, Ql = []; end

lp_opts = optimoptions('linprog', 'Display', 'none');

x         = [];
n_cuts    = 0;
n_lp      = 0;
exitflag  = 0;
viol      = Inf;
stop      = 'max_cuts';

for round = 1:opts.max_cuts

    % ---- Assemble cut rows -------------------------------------------
    nQu = size(Qu,2);
    Acut = zeros(nQu + size(Ql,2), nvar);
    bcut = zeros(nQu + size(Ql,2), 1);

    for j = 1:nQu
        q                = Qu(:,j);
        g                = Feu2D * kron(q, q);      % q' F_e q, per element
        Acut(j, i_b2)    = 1;
        Acut(j, i_d)     = -g';
        bcut(j)          = sum(Lu .* q.^2);         % q' diag(L) q
    end
    for j = 1:size(Ql,2)
        q                = Ql(:,j);
        g                = Fel2D * kron(q, q);
        r                = nQu + j;
        Acut(r, i_b1)    = -1;
        Acut(r, i_d)     = +g';
        bcut(r)          = -sum(Ll .* q.^2);
    end

    A = [Afix; Acut];
    b = [bfix; bcut];

    [x, ~, exitflag] = linprog(c, A, b, [], [], lb, ub, lp_opts);
    n_lp = n_lp + 1;

    if exitflag ~= 1 || isempty(x)
        stop = 'lp_failed';
        break
    end

    drho = x(i_d);

    % ---- Most violated direction of the upper LMI --------------------
    Gu       = diag(Lu) + reshape(Feu2D' * drho, N, N);
    Gu       = (Gu + Gu')/2;
    [Vu, Du] = eig(Gu);
    [muu, k] = min(real(diag(Du)));
    qu       = Vu(:,k);  qu = qu / norm(qu);
    viol_u   = x(i_b2) - muu;

    viol_l = -Inf; ql = [];
    if is_gap
        Gl       = diag(Ll) + reshape(Fel2D' * drho, R, R);
        Gl       = (Gl + Gl')/2;
        [Vl, Dl] = eig(Gl);
        [mul, k] = max(real(diag(Dl)));
        ql       = Vl(:,k);  ql = ql / norm(ql);
        viol_l   = mul - x(i_b1);
    end

    viol = max(viol_u, viol_l);

    if opts.verbose
        fprintf('   cut round %2d: viol = %.3e  (nQu = %d)\n', round, viol, nQu);
    end

    if viol <= opts.cut_tol
        stop = 'lmi_satisfied';
        break
    end

    if viol_u > opts.cut_tol, Qu = [Qu, qu]; n_cuts = n_cuts + 1; end %#ok<AGROW>
    if viol_l > opts.cut_tol, Ql = [Ql, ql]; n_cuts = n_cuts + 1; end %#ok<AGROW>
end

% ---- Pack results -----------------------------------------------------
if isempty(x) || exitflag ~= 1
    out.drho  = zeros(nEl,1);
    out.beta  = max(sp.up.L);
    out.beta1 = NaN;
    out.obj   = NaN;
else
    out.drho  = x(i_d);
    out.beta  = x(i_b2) * lam_ref;
    if is_gap
        out.beta1 = x(i_b1) * lam_ref;
        out.obj   = (x(i_b2) - x(i_b1)) * lam_ref;
    else
        out.beta1 = NaN;
        out.obj   = out.beta;
    end
end

at_bound          = (out.drho <= lb_d + 1e-12) | (out.drho >= ub_d - 1e-12);
out.n_cuts        = n_cuts;
out.n_lp          = n_lp;
out.exitflag      = exitflag;
out.stop_reason   = stop;
out.lmi_violation = viol;
out.frac_at_bound = mean(at_bound);
out.beta_cap_active = ~isempty(x) && (x(i_b2) >= b2cap - 1e-9*max(1,b2cap));
out.lam_ref       = lam_ref;
end
