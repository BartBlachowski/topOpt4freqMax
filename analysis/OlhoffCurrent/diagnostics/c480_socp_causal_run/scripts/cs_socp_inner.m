function [drho, st, rec] = cs_socp_inner(ctx, flags, treat)
%CS_SOCP_INNER  Exact certified solution of Du-Olhoff sub-problem (25) for the
%   N = 2 consistent-offset class (AUDIT_PREREGISTRATION.md sections 3-4).
%
%   Fail-closed: returns rec.accepted = false with a termination code and a zero
%   increment when the state is outside the proved class, the equivalence
%   checks fail, or no attempt is certified.  The caller then applies NO update.
%   There is no MMA fallback, no approximation and no skipped iteration.
tAll = tic;
NE = numel(ctx.rho);
drho = zeros(NE,1);
st = struct('nInner',0,'degenHits',0,'conv',false,'dxHist',[],'relHist',[],'beta',NaN);
rec = struct('outer',flags.outer,'accepted',false,'termination','','reason','', ...
    'move',ctx.move,'lam',ctx.lam(:).','lamJ',ctx.lamJ,'N',flags.N);
if isfield(ctx,'dOff'), rec.dOff = ctx.dOff(:).'; else, rec.dOff = []; end

% ---- E1-E8 --------------------------------------------------------------
te = tic;
[ok, E] = cs_socp_eligible(ctx, flags);
rec.eligibility = E; rec.tEligibility = toc(te);
if ~ok
    rec.termination = 'SOCP_UNSUPPORTED_CASE_HIT'; rec.reason = E.reason;
    rec.tTotal = toc(tAll); return
end

% ---- assembly -----------------------------------------------------------
ta = tic;
P = fp_problem(ctx);
soc = secondordercone(P.Ac, P.bc, P.dc, P.gammac);
rec.tAssembly = toc(ta);
rec.lamref = P.lamref;

[ok0, Q0] = cs_socp_equiv(P, [zeros(NE,1); 1], P.d2/P.lamref <= 1e-9);   % 2||s|| at x=[0;1] is d2/lamref
rec.equivZero = Q0;
if ~ok0
    rec.termination = 'SOCP_EQUIVALENCE_FAILURE'; rec.reason = 'E9-E12 failed at x=[0;1]';
    rec.tTotal = toc(tAll); return
end

% ---- attempt cascade (PREREGISTRATION_AMENDMENT_1: schur first) ---------
solvers = {'schur','augmented'};
att = struct('solver',{},'exitflag',{},'iterations',{},'message',{},'tSolve',{}, ...
             'tCertificate',{},'certified',{},'reason',{},'cert',{});
xAcc = []; C = [];
for a = 1:numel(solvers)
    opts = optimoptions('coneprog','Display','off','OptimalityTolerance',1e-10, ...
        'ConstraintTolerance',1e-10,'MaxIterations',500,'LinearSolver',solvers{a});
    ts = tic;
    try
        [x, ~, ef, op, la] = coneprog(P.f, soc, P.Alin, P.blin, [], [], P.xmin, P.xmax, opts);
        msg = op.message; its = op.iterations;
    catch ME
        x = []; ef = -99; la = []; msg = ME.message; its = NaN;
    end
    tSolve = toc(ts);
    tc = tic;
    [Ca, xa] = cs_socp_certify(P, x, la);
    tCert = toc(tc);
    att(end+1) = struct('solver',solvers{a},'exitflag',ef,'iterations',its, ...
        'message',strtrim(regexprep(msg,'\s+',' ')),'tSolve',tSolve,'tCertificate',tCert, ...
        'certified',Ca.certified,'reason',Ca.reason,'cert',Ca); %#ok<AGROW>
    st.nInner = st.nInner + max(its, 0)*isfinite(its);
    if Ca.certified
        xAcc = xa; C = Ca; rec.acceptedAttempt = a; rec.acceptedSolver = solvers{a};
        break
    end
end
rec.attempts = att;
rec.tSolve = sum([att.tSolve]); rec.tCertificate = sum([att.tCertificate]);
if isempty(xAcc)
    rec.termination = 'SOCP_CERTIFICATE_FAILURE';
    rec.reason = strjoin(arrayfun(@(t) sprintf('%s: %s', t.solver, t.reason), att, 'UniformOutput', false), ' | ');
    rec.tTotal = toc(tAll); return
end

% ---- E9-E12 at the accepted point --------------------------------------
[ok1, Q1] = cs_socp_equiv(P, xAcc, C.apex);
rec.equivAccepted = Q1;
if ~ok1
    rec.termination = 'SOCP_EQUIVALENCE_FAILURE'; rec.reason = 'E9-E12 failed at the accepted point';
    rec.tTotal = toc(tAll); return
end

% ---- accept ---------------------------------------------------------------
drho = xAcc(1:NE);
bs = xAcc(end);
st.conv = true; st.beta = bs*P.lamref;
rec.accepted = true;
rec.bs = bs; rec.beta = st.beta; rec.apex = C.apex;
rec.predGain = st.beta - ctx.lam(1);
[dlam, ~] = deltaLambda(ctx.F, drho, ctx.dOff);
rec.dlamPred = dlam(:).'; rec.fJJdrho = ctx.fJJ.'*drho;
rec.predLam = (ctx.lam(:) + dlam(:)).'; rec.predLamJ = ctx.lamJ + rec.fJJdrho;
rec.maxAbsDrho = max(abs(drho)); rec.norm2Drho = norm(drho); rec.sumDrho = sum(drho);
rec.primalResidual = C.maxRow; rec.fval = C.fval;
rec.gap = C.accepted.gap; rec.dualBound = C.accepted.dualBound;
rec.bestGap = C.bestGap; rec.bestDualBound = C.bestDualBound;
rec.candidate = C.candidate; rec.rowComp = C.accepted.rowComp; rec.boxComp = C.accepted.boxComp;
rec.bsStat = C.accepted.bsStat; rec.statRms = C.accepted.statRms; rec.statMax = C.accepted.statMax;
rec.mu = C.mu; rec.nu = C.nu(:).'; rec.sRow0 = C.sRow0; rec.coneNorm = C.coneNorm;
rec.separationPred = 2*C.coneNorm*P.lamref;
rec.rawBoxViolation = C.rawBoxViolation;
rec.candidateGaps = [C.candidates.gap];
rec.candidateNames = {C.candidates.name};

% bound structure (tolerance 1e-6 * box width, as fp_kkt / fi_metric)
width = P.xmax(1:NE) - P.xmin(1:NE);
atLo = drho <= P.xmin(1:NE) + 1e-6*width;
atHi = drho >= P.xmax(1:NE) - 1e-6*width;
rec.nLowerDensity = nnz(atLo & ~P.loMoveLimited);
rec.nLowerMove    = nnz(atLo &  P.loMoveLimited);
rec.nUpperMove    = nnz(atHi &  P.hiMoveLimited);
rec.nUpperDensity = nnz(atHi & ~P.hiMoveLimited);
rec.nInterior     = nnz(~atLo & ~atHi);
gray = ctx.rho > 0.1 & ctx.rho < 0.9;
rec.nGray = nnz(gray); rec.nGrayFullMove = nnz(gray & (atLo | atHi) & (P.loMoveLimited | P.hiMoveLimited));

% ---- cross-solver diagnostic (never gates acceptance) -----------------
rec.cross = [];
ce = 0; if isfield(treat,'crossEvery'), ce = treat.crossEvery; end
if ce > 0 && (flags.outer == 1 || mod(flags.outer, ce) == 0)
    other = solvers{3 - rec.acceptedAttempt};
    opts = optimoptions('coneprog','Display','off','OptimalityTolerance',1e-10, ...
        'ConstraintTolerance',1e-10,'MaxIterations',500,'LinearSolver',other);
    tx = tic;
    try
        [x2, ~, ef2, op2, la2] = coneprog(P.f, soc, P.Alin, P.blin, [], [], P.xmin, P.xmax, opts);
        tSolve2 = toc(tx);
        tc2 = tic; [C2, x2] = cs_socp_certify(P, x2, la2); tCert2 = toc(tc2);
        d2v = x2(1:NE);
        large = abs(drho) >= 0.9*ctx.move;
        b1 = int8(atHi) - int8(atLo);
        b2 = int8(d2v >= P.xmax(1:NE) - 1e-6*width) - int8(d2v <= P.xmin(1:NE) + 1e-6*width);
        ob = b1 ~= 0;
        rec.cross = struct('solver',other,'exitflag',ef2,'iterations',op2.iterations, ...
            'd2',norm(d2v-drho)/max(norm(drho),eps),'dinf',max(abs(d2v-drho))/ctx.move, ...
            'dbs',abs(x2(end)-bs),'signAgreement',mean(sign(d2v(large))==sign(drho(large))), ...
            'boundAgreement',mean(b2(ob)==b1(ob)),'nDiffGt0p1Move',nnz(abs(d2v-drho) > 0.1*ctx.move), ...
            'certified',C2.certified,'bestGap',C2.bestGap,'tSolve',tSolve2,'tCertificate',tCert2, ...
            'tTotal',toc(tx));
    catch ME
        rec.cross = struct('solver',other,'error',ME.message,'tSolve',toc(tx));
    end
end
rec.tTotal = toc(tAll);
if isfield(treat,'progress') && treat.progress
    cd2 = NaN; cdi = NaN;
    if ~isempty(rec.cross) && isfield(rec.cross,'d2'), cd2 = rec.cross.d2; cdi = rec.cross.dinf; end
    fprintf('[socp] it %4d mv %.3g w1 %.6f gain %.3e att %d(%s) ipm %d gap %.1e box %.1e int %d tS %.1f tC %.1f | x d2 %.2e dinf %.2f\n', ...
        flags.outer, ctx.move, sqrt(ctx.lam(1)), rec.predGain, rec.acceptedAttempt, ...
        rec.acceptedSolver, st.nInner, rec.gap, rec.boxComp, rec.nInterior, rec.tSolve, rec.tCertificate, cd2, cdi);
end
end
