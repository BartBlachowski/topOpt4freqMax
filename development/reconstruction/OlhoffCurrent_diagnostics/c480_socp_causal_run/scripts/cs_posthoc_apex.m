function H = cs_posthoc_apex()
%CS_POSTHOC_APEX  POST-HOC DIAGNOSTIC -- EXCLUDED FROM EVERY VERDICT.
%
%   The treatment terminated fail-closed at outer 15 (SOCP_CERTIFICATE_FAILURE).
%   This function re-poses THAT frozen, rejected sub-problem from the saved
%   treatment state and asks one question: was the rejected primal point in fact
%   optimal (certificate-construction limitation at the cone apex), or not
%   (solver failure)?
%
%   No increment is applied to any design.  The trajectory is not continued.
%   Nothing here can alter the preregistered termination or verdicts.
S = cs_setup();
guard = olhoffcurrent_paths(); %#ok<NASGU>
maxNumCompThreads(1);
TR = jsondecode(fileread(fullfile(S.study,'evaluations','termination_record.json')));
st = load(fullfile(S.evDir,'C480x60_socp_state.mat'),'state'); cfg = st.state.cfg; rho = st.state.rho;
frozen = rho;
g = @(p) olh.config.getPath(cfg, p);
flat = olh.config.toLegacy(cfg); mdl = model2D(flat);
rminEl = g('filter.radiusPhysical')/(g('domain.b')/g('domain.mesh.nely'));
flt = prepFilter(g('domain.mesh.nelx'), g('domain.mesh.nely'), rminEl);
n = g('eigen.targetMode'); Jcalc = n + g('eigen.maxCluster');
pNow = g('material.stiffness.p'); massNowCfg = g('material.mass');
% ---- identical expressions to olhoffSolve step 1-2 at outer 15 -----------
[K,M] = assemble2D(mdl, rho, pNow, massNowCfg);
[w, Phi, lam] = eigSolve(K, M, Jcalc, g('eigen.solver'));
[N, ~] = olh.multi.detect(cfg, w, n, Jcalc, []);
J = n + N; idx = n:(n+N-1);
F = genGrad(mdl, rho, pNow, massNowCfg, Phi, lam(n), idx);
for j = 1:N
    Gj = genGrad(mdl, rho, pNow, massNowCfg, Phi, lam(idx(j)), idx(j)); F(:,j,j) = Gj(:,1,1);
end
dOff = lam(idx) - lam(idx(1));
FJ = genGrad(mdl, rho, pNow, massNowCfg, Phi, lam(J), J); fJJ = FJ(:,1,1);
for s = 1:N, for k = s:N
    v = applyFilter(flt, rho, F(:,s,k)); F(:,s,k) = v; F(:,k,s) = v;
end, end
fJJ = applyFilter(flt, rho, fJJ);
ctx = struct('F',F,'fJJ',fJJ,'lam',lam(idx),'lamJ',lam(J),'rho',rho,'rhomin',g('design.minimum'), ...
    'volfrac',g('design.volumeFraction'),'move',TR.move,'maxInner',g('optimizer.inner.maxIterations'), ...
    'tolInner',g('optimizer.inner.tolerance'),'minInner',g('optimizer.inner.minIterations'), ...
    'offDiag',g('multiplicity.offDiagonal'),'dOff',dOff);
H = struct('label','POST-HOC DIAGNOSTIC -- excluded from all verdicts; no design update');
H.reposed_identity = struct('lam_bitwise', isequal(ctx.lam(:).', TR.lam(:).'), 'lamJ_bitwise', ctx.lamJ == TR.lamJ, ...
    'dOff_bitwise', isequal(dOff(:).', TR.dOff(:).'), 'move', TR.move);
P = fp_problem(ctx); NE = P.NE;
soc = secondordercone(P.Ac, P.bc, P.dc, P.gammac);
opts = optimoptions('coneprog','Display','off','OptimalityTolerance',1e-10,'ConstraintTolerance',1e-10, ...
    'MaxIterations',500,'LinearSolver','schur');
t = tic; [x, ~, ef, op, la] = coneprog(P.f, soc, P.Alin, P.blin, [], [], P.xmin, P.xmax, opts); H.primal_solve_s = toc(t);
x = min(P.xmax, max(P.xmin, x));
H.primal = struct('exitflag',ef,'iterations',op.iterations,'bs',x(end), ...
    'bs_equals_rejected_record', abs(x(end) - TR.attempts(1).bs) <= 1e-14, ...
    'coneNorm', norm(P.Ac*x - P.bc), 'maxRow', max(P.evalProd(x)), 'lambda_soc', la.soc, 'lambda_ineqlin', la.ineqlin(:).');
width = P.xmax(1:NE) - P.xmin(1:NE);
atLo = x(1:NE) <= P.xmin(1:NE) + 1e-6*width; atHi = x(1:NE) >= P.xmax(1:NE) - 1e-6*width;
H.primal.nInterior = nnz(~atLo & ~atHi); H.primal.frac_any_bound = mean(atLo | atHi);
gray = rho > .1 & rho < .9;
H.primal.gray_n = nnz(gray); H.primal.gray_full_move_frac = mean((atLo(gray) | atHi(gray)));

% ---- (A) exact Lagrangian dual as a conic program -------------------------
% vars y = [p(2); mu; nu(2); tvec(nvar)] ; maximize sum(tvec) - p'bc + mu*gammac - nu'blin
% q = f + Ac'p - mu*dc + Alin'nu ;  tvec <= q.*xmin ,  tvec <= q.*xmax ;  ||p|| <= mu ; nu >= 0
nv = P.nvar; ny = 5 + nv;
Bq = [P.Ac.', -P.dc, P.Alin.'];                      % nv x 5, q = f + Bq*[p;mu;nu]
Dmin = spdiags(P.xmin,0,nv,nv); Dmax = spdiags(P.xmax,0,nv,nv);
Aineq = [ -Dmin*Bq, speye(nv) ; -Dmax*Bq, speye(nv) ];
bineq = [ P.f.*P.xmin ; P.f.*P.xmax ];
cobj = -[ -P.bc ; P.gammac ; -P.blin ; ones(nv,1) ];
lb = [-inf; -inf; 0; 0; 0; -inf(nv,1)];
ub = inf(ny,1);
Acone = sparse([1 2],[1 2],[1 1],2,ny); dcone = sparse(3,1,1,ny,1);
socD = secondordercone(Acone, zeros(2,1), dcone, 0);  % ||p|| <= mu
t = tic;
[yd, fvD, efD, opD] = coneprog(cobj, socD, Aineq, bineq, [], [], lb, ub, ...
    optimoptions('coneprog','Display','off','OptimalityTolerance',1e-12,'ConstraintTolerance',1e-12,'MaxIterations',500));
H.dual_solve_s = toc(t);
p = yd(1:2); mu = max(yd(3), norm(p)); nu = max(yd(4:5), 0);
H.exact_dual = local_eval(P, x, p, mu, nu, 'exactLagrangianDual', efD, opD.iterations);
% ---- (B) generalized complementary slackness with free p ----------------
inter = [~atLo & ~atHi; true];
Mls = Bq(inter,:); rhs = -P.f(inter);
z = lsqlin(Mls, rhs, [], [], [], [], [-inf;-inf;0;0;0], [], [], optimoptions('lsqlin','Display','off'));
H.cs_free_p = local_eval(P, x, z(1:2), max(z(3), norm(z(1:2))), max(z(4:5),0), 'complementarySlacknessFreeP', NaN, NaN);
H.cs_free_p.n_interior_rows = nnz(inter);
% ---- (C) the preregistered certificate, re-evaluated for reference -------
[C, ~] = cs_socp_certify(P, x, la);
H.preregistered_certificate = struct('certified', C.certified, 'bestGap', C.bestGap, 'reason', C.reason, 'apex', C.apex);
assert(isequal(rho, frozen), 'frozen state changed');
cs_json(fullfile(S.study,'evaluations','posthoc_apex_diagnostic.json'), H);
disp(H.reposed_identity); disp(H.primal); disp(H.exact_dual); disp(H.cs_free_p); disp(H.preregistered_certificate);
end

function E = local_eval(P, x, p, mu, nu, name, ef, its)
NE = P.NE;
q = P.f + P.Ac.'*p - mu*P.dc + P.Alin.'*nu;
D = sum(min(q.*P.xmin, q.*P.xmax)) - p.'*P.bc + mu*P.gammac - nu.'*P.blin;
fval = P.evalProd(x);
sRow0 = sqrt(mean((P.F11/P.lamref).^2));
xi = max(q(1:NE),0); et = max(-q(1:NE),0);
E = struct('name',name,'exitflag',ef,'iterations',its,'p',p(:).','mu',mu,'nu',nu(:).', ...
    'dual_feasible', mu >= norm(p) && all(nu >= 0), 'dual_bound', D, 'gap', P.f.'*x - D, ...
    'boxComp', max([xi.*(x(1:NE)-P.xmin(1:NE)); et.*(P.xmax(1:NE)-x(1:NE))])/(sRow0*P.move), ...
    'rowComp', max(abs([mu;0;nu].*fval)), 'bsStat', abs(q(end)), ...
    'cone_complementarity', mu*norm(P.Ac*x - P.bc) - p.'*(P.Ac*x - P.bc), 'p_norm_over_mu', norm(p)/max(mu,realmin));
E.would_pass_C1_C8_bars = fval(1) <= 1e-8 && max(fval) <= 1e-8 && E.gap >= -1e-8 && E.gap <= 1e-8 && E.dual_feasible && ...
    E.rowComp <= 1e-6 && E.boxComp <= 1e-4 && E.bsStat <= 1e-6;
end
