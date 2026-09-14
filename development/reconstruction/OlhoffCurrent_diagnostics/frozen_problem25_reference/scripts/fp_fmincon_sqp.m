function out = fp_fmincon_sqp()
%FP_FMINCON_SQP  Part 8, cross-check C: fmincon 'sqp' from S1 (P19), capped at
%   MaxIterations = 50 and 30 min wall.  n = 28 801 makes SQP's dense
%   quasi-Newton Hessian and dense QP subproblem the viability question.
S = fp_setup();                                   %#ok<NASGU>
ev = fullfile(S.study,'evaluations');
L = load(fullfile(ev,'frozen_ctx.mat'),'ctx','xP19'); P = fp_problem(L.ctx);
fun = @(x) deal(-x(end), P.f);
nonl = @(x) local_nonl(P, x);
t0 = tic; capS = 1800;
outfcn = @(x,ov,state) toc(t0) > capS;
opts = optimoptions('fmincon','Algorithm','sqp','Display','iter', ...
    'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true, ...
    'OptimalityTolerance',1e-10,'ConstraintTolerance',1e-10,'StepTolerance',1e-14, ...
    'MaxIterations',50,'MaxFunctionEvaluations',1e4,'OutputFcn',outfcn);
o = struct('start','S1_P19','algorithm','sqp','cap_s',capS,'MaxIterations',50);
try
    [x, fv, ef, op, lam] = fmincon(fun, L.xP19, P.Alin, P.blin, [], [], P.xmin, P.xmax, nonl, opts);
    o.exitflag = ef; o.iterations = op.iterations; o.funcCount = op.funcCount; o.message = op.message;
    o.bs = x(end); o.firstorderopt = op.firstorderopt; o.constrviolation = op.constrviolation;
    muRows = [lam.ineqnonlin(:); lam.ineqlin(:)];
    K = fp_kkt(P, x, muRows, lam.lower(:), lam.upper(:), 'fmincon sqp S1');
    o.kkt = rmfield(K,'masks'); o.viable = true;
    xSQP = x; save(fullfile(ev,'fmincon_sqp.mat'),'xSQP','-v7.3');
catch ME
    o.exitflag = -99; o.message = ME.message; o.viable = false; o.bs = NaN;
end
o.wall_s = toc(t0);
fid = fopen(fullfile(ev,'fmincon_sqp.json'),'w');
fprintf(fid,'%s',jsonencode(o,'PrettyPrint',true)); fclose(fid);
fprintf('[fp_fmincon_sqp] ef=%d wall=%.0fs viable=%d bs=%.10f msg=%s\n', o.exitflag, o.wall_s, o.viable, o.bs, o.message);
out = o;
end
function [c, ceq, gc, gceq] = local_nonl(P, x)
[fval, dfdx] = P.evalProd(x);
c = fval(1:2); ceq = []; gc = dfdx(1:2,:).'; gceq = [];
end
