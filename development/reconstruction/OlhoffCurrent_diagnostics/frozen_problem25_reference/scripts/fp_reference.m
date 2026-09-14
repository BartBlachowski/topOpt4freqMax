function out = fp_reference()
%FP_REFERENCE  Parts 7, 9, 10, 13: the certified conic reference solution.
%
%   Re-runs the coneprog configuration selected by the sweep (x-form, 'schur',
%   1e-10 -- smallest independently certified duality gap), maximizes the
%   weak-duality bound independently of the solver's duals, evaluates KKT of
%   the EXACT production problem with (i) the solver's multipliers and
%   (ii) the certificate multipliers, and describes the solution.
%   FROZEN-SUBPROBLEM REFERENCE -- drho is never applied to a density.
S = fp_setup();                                   %#ok<NASGU>
ev = fullfile(S.study,'evaluations');
L = load(fullfile(ev,'frozen_ctx.mat'),'ctx','xP19','xM500'); P = fp_problem(L.ctx);
NE = P.NE; nvar = P.nvar;

cfgSel = struct('form','x','LinearSolver','schur','OptimalityTolerance',1e-10,'ConstraintTolerance',1e-10,'MaxIterations',500);
soc = secondordercone(P.Ac, P.bc, P.dc, P.gammac);
opts = optimoptions('coneprog','Display','iter','OptimalityTolerance',cfgSel.OptimalityTolerance, ...
    'ConstraintTolerance',cfgSel.ConstraintTolerance,'MaxIterations',cfgSel.MaxIterations,'LinearSolver',cfgSel.LinearSolver);
t0 = tic;
[x, fv, ef, op, lam] = coneprog(P.f, soc, P.Alin, P.blin, [], [], P.xmin, P.xmax, opts);
wall = toc(t0);
xRef = x;
o = struct('label','FROZEN-SUBPROBLEM REFERENCE (coneprog, certified) -- drho discarded, never applied');
o.config = cfgSel; o.exitflag = ef; o.output = op; o.wall_s = wall; o.fval_solver = fv;
o.bs = x(end); o.beta = x(end)*P.lamref; o.omega_pred = sqrt(o.beta); o.bs_minus_1 = x(end)-1;
o.lamref = P.lamref; o.omega1_current = sqrt(P.lamref);
o.box_violation_raw = max([max(P.xmin - x), max(x - P.xmax), 0]);

% ---- solver duals ---------------------------------------------------------
mu = lam.soc; if iscell(mu), mu = mu{1}; end
xsiS = lam.lower(:); etaS = lam.upper(:); nuS = lam.ineqlin(:);
o.solver_duals = struct('mu_soc',mu(:).','nu_ineqlin',nuS.','xsi_max',max(xsiS),'eta_max',max(etaS));
muRowsS = [max(mu(1),0); 0; max(nuS,0)];
KS = fp_kkt(P, x, muRowsS, xsiS, etaS, 'solver multipliers (coneprog, dual unconverged)');
o.kkt_solver_duals = rmfield(KS,'masks');

% ---- certificate duals (independent of solver status) --------------------
D = fp_dualbound(P, x, max(mu(1),0), nuS);
o.certificate = D;
% KKT uses the ALIGNED certificate (exact cone direction at x); the free
% maximizer may tilt w by ~1e-6 for a bound better by <1e-12, which would
% leave a spurious stationarity residual.  Both bounds are reported.
p = D.aligned.p(:); muC = D.aligned.mu; nuC = D.aligned.nu(:);
o.certificate_aligned = D.aligned;
q = P.f + P.Ac.'*p - muC*P.dc + P.Alin.'*nuC;
xsiC = max(q,0); etaC = max(-q,0);
muRowsC = [muC; 0; nuC];
KC = fp_kkt(P, x, muRowsC, xsiC, etaC, 'certificate multipliers (maximized dual bound)');
o.kkt_certificate_duals = rmfield(KC,'masks');
% complementarity decomposition of the gap (every term is >= 0 for feasible x)
s = P.Ac*x - P.bc; ns = norm(s);
termBox = sum(q.*x - min(q.*P.xmin, q.*P.xmax));
termCone = muC*ns - p.'*s;
termLin = nuC.'*(P.blin - P.Alin*x);
o.gap_decomposition = struct('box_complementarity',termBox,'cone_complementarity',termCone, ...
    'linear_complementarity',termLin,'sum',termBox+termCone+termLin,'gap',D.gap);
o.cone_dual_feasible = struct('norm_p',norm(p),'mu',muC,'norm_p_le_mu',norm(p) <= muC + 1e-10, ...
    'cone_gap',muC*ns - p.'*s);
% free coordinates: |q_e| tiny means the coordinate is on a degenerate face
qs = abs(q(1:NE)); qmax = max(qs);
o.free_coordinates = struct('n_abs_q_below_1e8_rel',nnz(qs <= 1e-8*qmax),'n_abs_q_below_1e6_rel',nnz(qs <= 1e-6*qmax), ...
    'n_abs_q_below_1e4_rel',nnz(qs <= 1e-4*qmax),'qmax',qmax,'q_bs',q(nvar));
% KKT verdict: certificate duals decide (preregistration sec. 5: multipliers
% used are actual multipliers or a mathematically correct KKT treatment; the
% certificate duals are verified, not assumed).
o.kkt_verdict = KC.verdict;
o.kkt_verdict_solver_duals = KS.verdict;
globalOK = strcmp(KC.verdict,'REFERENCE_PROBLEM25_KKT_PASS') && min(D.gap, D.aligned.gap) <= 1e-8;
if globalOK, o.global_verdict = 'GLOBAL_PROBLEM25_REFERENCE_CERTIFIED';
else,        o.global_verdict = 'LOCAL_PROBLEM25_REFERENCE_ONLY'; end

% ---- solution description ------------------------------------------------
[fval, ~, dlam, ~, evals] = P.evalProd(x); %#ok<ASGLU>
drho = x(1:NE);
o.production_fval = fval.'; o.evals = evals.';
o.solution = struct('bs',x(end),'beta',x(end)*P.lamref,'omega_pred',sqrt(x(end)*P.lamref), ...
    'max_abs_drho',max(abs(drho)),'max_abs_drho_over_move',max(abs(drho))/P.move, ...
    'norm2_drho',norm(drho),'mean_abs_drho',mean(abs(drho)),'rms_drho',sqrt(mean(drho.^2)), ...
    'volume_change_abs',sum(drho),'volume_change_rel',sum(drho)/P.Vtot,'volume_row',fval(4),'volume_slack',-fval(4), ...
    'nextmode_row',fval(3),'nextmode_slack',-fval(3),'cluster1_row',fval(1),'cluster2_row',fval(2),'cluster2_slack',-fval(2), ...
    'separation_e2_minus_e1',evals(2)-evals(1),'separation_over_d2',(evals(2)-evals(1))/P.d2, ...
    'image_abc',P.abc(drho).','fJJ_drho',P.fJJ.'*drho,'sum_drho',sum(drho), ...
    'n_positive',nnz(drho>0),'n_negative',nnz(drho<0),'n_zero',nnz(drho==0));
o.active = KC.active; o.active_sweep = KC.active_sweep; o.active_sweep_cols = KC.active_sweep_cols;
o.move_bound_dominated = KC.active.frac_move_bound >= 0.5;
% by density class of rho385
rho = P.rho; cls = {'void (rho<0.1)', rho < 0.1; 'gray', rho >= 0.1 & rho <= 0.9; 'solid (rho>0.9)', rho > 0.9};
m = KC.masks;
for k = 1:3
    idx = cls{k,2};
    o.by_class.(matlab.lang.makeValidName(cls{k,1})) = struct('n',nnz(idx), ...
        'n_minus_move',nnz(idx & m.atLo & P.loMoveLimited),'n_floor',nnz(idx & m.atLo & ~P.loMoveLimited), ...
        'n_plus_move',nnz(idx & m.atHi & P.hiMoveLimited),'n_ceiling',nnz(idx & m.atHi & ~P.hiMoveLimited), ...
        'n_interior',nnz(idx & ~m.atLo & ~m.atHi),'mean_drho',mean(drho(idx)),'sum_drho',sum(drho(idx)));
end
% gains relative to the MMA states available now
o.compare_quick = struct('bs_P19',L.xP19(end),'bs_M500',L.xM500(end), ...
    'G_P19',(x(end)-L.xP19(end))/(x(end)-1),'G_M500',(x(end)-L.xM500(end))/(x(end)-1));

fid = fopen(fullfile(ev,'reference_solution.json'),'w');
fprintf(fid,'%s',jsonencode(o,'PrettyPrint',true)); fclose(fid);
masks = KC.masks; muRef = muRowsC; xsiRef = xsiC; etaRef = etaC; qRef = q; lamSolver = lam;
save(fullfile(ev,'conic_reference.mat'),'xRef','muRef','xsiRef','etaRef','masks','qRef','lamSolver','-v7.3');
fprintf('\n[fp_reference] exitflag=%d it=%d wall=%.1fs  bs=%.12f beta=%.6f  bs-1=%.6e\n', ef, op.iterations, wall, o.bs, o.beta, o.bs_minus_1);
fprintf('  production fval = %s   box viol raw = %.2e\n', mat2str(fval.',6), o.box_violation_raw);
fprintf('  certificate: primal=%.13f dual=%.13f gap=%.3e (rel to gain %.3e)  mu=%.8f nu=%s ||p||=%.8f\n', ...
    D.primal, D.dual_bound, D.gap, D.gap_relative_to_gain, muC, mat2str(nuC.',8), norm(p));
fprintf('  aligned certificate: dual=%.13f gap=%.3e mu=%.10f nu=%s\n', D.aligned.dual_bound, D.aligned.gap, muC, mat2str(nuC.',10));
fprintf('  gap decomposition: box %.3e cone %.3e lin %.3e\n', termBox, termCone, termLin);
fprintf('  KKT(solver duals)      : %s statRMS=%.3e\n', KS.verdict, KS.stationarity.norm_rms);
fprintf('  KKT(certificate duals) : %s statRMS=%.3e statMax=%.3e comp=%.3e boxcomp=%.3e maxfval=%.3e\n', KC.verdict, ...
    KC.stationarity.norm_rms, KC.stationarity.norm_max, KC.complementarity.max_abs_mu_f, KC.complementarity.max_box_comp_normalized, KC.primal.max_fval);
fprintf('  %s\n', o.global_verdict);
fprintf('  active: -move %d floor %d +move %d ceiling %d interior %d (frac move %.4f, any bound %.4f)\n', ...
    KC.active.n_minus_move, KC.active.n_floor, KC.active.n_plus_move, KC.active.n_ceiling, KC.active.n_interior, KC.active.frac_move_bound, KC.active.frac_any_bound);
fprintf('  free coords |q|<=1e-8 qmax: %d ; <=1e-6: %d ; <=1e-4: %d\n', o.free_coordinates.n_abs_q_below_1e8_rel, o.free_coordinates.n_abs_q_below_1e6_rel, o.free_coordinates.n_abs_q_below_1e4_rel);
fprintf('  separation e2-e1 = %.3f (%.4f d2); next-mode slack %.4f; volume row %.3e\n', o.solution.separation_e2_minus_e1, o.solution.separation_over_d2, o.solution.nextmode_slack, fval(4));
fprintf('  G_P19 = %.5f  G_M500 = %.5f\n', o.compare_quick.G_P19, o.compare_quick.G_M500);
disp(o.by_class);
out = o;
end
