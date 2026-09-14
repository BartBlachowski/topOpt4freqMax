function out = fp_equivalence()
%FP_EQUIVALENCE  Parts 3-5: cluster-constraint redundancy, SOC derivation
%   checks, pointwise SOCP <-> deltaLambda equivalence, gradient and Hessian
%   agreement, coneprog sign-convention toy test, smoothness margin.
S = fp_setup();                                   %#ok<NASGU>
ev = fullfile(S.study,'evaluations');
L = load(fullfile(ev,'frozen_ctx.mat'),'ctx');
ctx = L.ctx; P = fp_problem(ctx);
NE = P.NE; lamref = P.lamref;
T = fp_testpoints(P, ev);
deltas = [-1e-3, 0, 1e-3];

o = struct();
o.N = P.N; o.dOff = P.dOff.'; o.dOff_present = ~isempty(ctx.dOff);
o.lam = P.lam.'; o.lam_sorted_ascending = issorted(P.lam);
o.dOff_identity = struct('lam2_minus_dOff2_minus_lam1', (P.lam(2)-P.d2)-P.lam(1), 'bar_4eps_lam1', 4*eps*P.lam(1));
o.smoothMargin = P.smoothMargin;

rows = struct('name',{},'delta',{},'bs',{},'fval_prod',{},'cone_resid',{},'diff_row1',{}, ...
    'sign_agree',{},'class_prod',{},'class_cone',{},'ev_sorted',{},'dlam',{},'ev',{},'dlam_sorted',{},'row1_le_row2',{},'row1_minus_row2_lam',{}, ...
    'lin_row3_diff',{},'lin_row4_diff',{},'obj_diff',{},'grad_row1_relinf',{},'ddlam1_vs_closed_relinf',{}, ...
    'e1_prod_vs_closed',{},'sep_e2_minus_e1',{},'maxabs_drho_over_move',{},'vol_row',{});
for i = 1:numel(T)
    drho = T(i).drho;
    [fv0, ~, dlam0] = P.evalProd([drho; 0]);     %#ok<ASGLU> value at bs=0 to get e1
    e1prod = P.lam(1) + dlam0(1);                 % lam1 + dlam1 = lam1 + e1
    for d = deltas
        bs = e1prod/lamref*(1+d);
        x = [drho; bs];
        [fval, dfdx, dlam, ddlam, evals] = P.evalProd(x);
        cr = P.coneResid(x);
        r = struct();
        r.name = T(i).name; r.delta = d; r.bs = bs;
        r.fval_prod = fval.'; r.cone_resid = cr; r.diff_row1 = fval(1) - cr;
        % three-way class {feasible, active, infeasible}; the delta = 0 points
        % are constructed EXACTLY active (production value is 0.0 by
        % construction), so a pure sign test there compares roundoff.  Band
        % 1e-12 -- a disclosed refinement of the preregistered sign test.
        cls = @(v) (v > 1e-12) - (v < -1e-12);
        r.class_prod = cls(fval(1)); r.class_cone = cls(cr);
        r.sign_agree = r.class_prod == r.class_cone;
        r.ev_sorted = issorted(evals);
        r.dlam = dlam.'; r.ev = evals.'; r.dlam_sorted = issorted(dlam);
        pred = P.lam(:) + dlam(:);                % lam_j + dlam_j
        r.row1_le_row2 = pred(1) <= pred(2) + 1e-9*lamref;
        r.row1_minus_row2_lam = pred(1) - pred(2);
        r.lin_row3_diff = fval(3) - (P.Alin(1,:)*x - P.blin(1));
        r.lin_row4_diff = fval(4) - (P.Alin(2,:)*x - P.blin(2));
        r.obj_diff = (-bs) - P.f.'*x;
        gc = P.coneGrad(x); gp = dfdx(1,:).';
        r.grad_row1_relinf = max(abs(gp - gc))/max(abs(gp));
        ge = P.ge1closed(drho);
        r.ddlam1_vs_closed_relinf = max(abs(ddlam(:,1) - ge))/max(abs(ge));
        r.e1_prod_vs_closed = (P.lam(1)+dlam(1)) - (P.lam(1)+P.e1closed(drho));
        r.sep_e2_minus_e1 = evals(2) - evals(1);
        r.maxabs_drho_over_move = max(abs(drho))/P.move;
        r.vol_row = fval(4);
        rows(end+1) = r; %#ok<AGROW>
    end
end
o.points = rows;
o.n_points = numel(T); o.point_names = {T.name};

% ---- Hessian-multiply check (finite difference of the production gradient)
rng(7);
xh = [T(2).drho; 1]; vdir = randn(NE+1,1); vdir(end) = 0; vdir = vdir/norm(vdir);
h = 1e-7;
[~, gp1] = P.evalProd(xh + h*vdir); [~, gm1] = P.evalProd(xh - h*vdir);
fdH = (gp1(1,:) - gm1(1,:)).'/(2*h);              % d/dv of grad row1
lamS = struct('ineqnonlin',[1;0;0;0]);
Hv = P.hessmult(xh, lamS, vdir);
o.hessian_check = struct('relerr_row1', norm(fdH - Hv)/max(norm(fdH),eps), 'h',h);
lamS2 = struct('ineqnonlin',[0;1;0;0]);
fdH2 = (gp1(2,:) - gm1(2,:)).'/(2*h); Hv2 = P.hessmult(xh, lamS2, vdir);
o.hessian_check.relerr_row2 = norm(fdH2 - Hv2)/max(norm(fdH2),eps);

% ---- coneprog sign-convention toy test ---------------------------------
% min x2  s.t.  ||x1 - 0.5|| <= x2 - (-2)   =>  x2* = -2 under the documented
% convention ||A x - b|| <= d'x - gamma; x2* = +2 under the alternative.
soc = secondordercone([1 0], 0.5, [0;1], -2);
opts = optimoptions('coneprog','Display','off','OptimalityTolerance',1e-10);
[xt, ft, ef] = coneprog([0;1], soc, [], [], [], [], [-10;-10], [10;10], opts);
o.toy = struct('x',xt.','fval',ft,'exitflag',ef,'expected_documented',-2,'expected_alternative',2, ...
    'documented_convention_confirmed', abs(ft - (-2)) <= 1e-6);

% ---- verdicts ------------------------------------------------------------
d1 = [rows.diff_row1]; sa = [rows.sign_agree]; g1 = [rows.grad_row1_relinf];
g2 = [rows.ddlam1_vs_closed_relinf]; l3 = [rows.lin_row3_diff]; l4 = [rows.lin_row4_diff]; ob = [rows.obj_diff];
r12 = [rows.row1_le_row2]; ds = [rows.dlam_sorted]; es = [rows.ev_sorted];
o.summary = struct('max_abs_diff_row1', max(abs(d1)), 'all_sign_agree', all(sa), ...
    'max_grad_row1_relinf', max(g1), 'max_ddlam1_vs_closed_relinf', max(g2), ...
    'max_lin_row3_diff', max(abs(l3)), 'max_lin_row4_diff', max(abs(l4)), 'max_obj_diff', max(abs(ob)), ...
    'all_row1_le_row2', all(r12), 'all_ev_sorted', all(es), ...
    'all_dlam_sorted_NONCRITERIAL', all(ds), ...
    'note_dlam', ['deltaLambda sorts the EIGENVALUES e_j of diag(dOff)+A and returns dlam_j = e_j - dOff_j; ' ...
                  'with dOff(2) = 7428.6 the increments are measured from different baselines and need not be ordered. ' ...
                  'The constraint-relevant ordering is lam_j + dlam_j = lam_1 + e_j, which IS ordered (all_row1_le_row2).'], ...
    'n_class_active_points', nnz([rows.class_prod] == 0), ...
    'min_sep_e2_minus_e1', min([rows.sep_e2_minus_e1]), ...
    'max_abs_e1_prod_vs_closed', max(abs([rows.e1_prod_vs_closed])));
clusterPass = o.lam_sorted_ascending && all(es) && all(r12) && ...
    abs(o.dOff_identity.lam2_minus_dOff2_minus_lam1) <= o.dOff_identity.bar_4eps_lam1;
if clusterPass, o.cluster_verdict = 'CLUSTER_CONSTRAINT_REDUCTION_PASS';
else,           o.cluster_verdict = 'CLUSTER_CONSTRAINT_REDUCTION_FAIL'; end
socpPass = max(abs(d1)) <= 1e-10 && all(sa) && max(g1) <= 1e-9 && max(g2) <= 1e-9 && ...
    max(abs(l3)) <= 1e-12 && max(abs(l4)) <= 1e-12 && max(abs(ob)) <= 1e-12 && ...
    o.toy.documented_convention_confirmed && o.hessian_check.relerr_row1 <= 1e-4;
if socpPass, o.socp_verdict = 'FROZEN_PROBLEM25_SOCP_EQUIVALENCE_PASS';
else,        o.socp_verdict = 'FROZEN_PROBLEM25_SOCP_EQUIVALENCE_FAIL'; end
if clusterPass && socpPass
    o.convexity_verdict = 'FROZEN_PROBLEM25_CONVEXITY_CERTIFIED';  % a SOC is convex whether or not its apex is reachable
else
    o.convexity_verdict = 'FROZEN_PROBLEM25_CONVEXITY_NOT_CERTIFIED';
end

fid = fopen(fullfile(ev,'socp_equivalence.json'),'w');
fprintf(fid,'%s',jsonencode(o,'PrettyPrint',true)); fclose(fid);
fprintf('[fp_equivalence] %s\n[fp_equivalence] %s\n[fp_equivalence] %s\n', o.cluster_verdict, o.socp_verdict, o.convexity_verdict);
disp(o.summary); disp(o.smoothMargin); disp(o.toy); disp(o.hessian_check);
fprintf('  points: %s\n', strjoin(o.point_names, ', '));
out = o;
end
