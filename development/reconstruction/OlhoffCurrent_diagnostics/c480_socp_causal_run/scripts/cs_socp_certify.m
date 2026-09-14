function [C, x] = cs_socp_certify(P, x, la)
%CS_SOCP_CERTIFY  Preregistered independent certificate (section 4.3) for a
%   returned point x of the conic problem P.  la = coneprog's lambda struct
%   (may be empty).  Returns the CLIPPED x that the certificate is about.
%
%   Every dual candidate is dual-feasible by construction, so each gives a valid
%   weak-duality lower bound regardless of how it was obtained.  x is certified
%   iff at least one candidate meets ALL bars C1-C8 simultaneously.
C = struct('certified',false,'reason','','candidate','','apex',false,'bestGap',NaN,'maxRow',NaN);
NE = P.NE;
C.nonfinite = isempty(x) || any(~isfinite(x));
if C.nonfinite
    C.reason = 'no finite primal point'; C.rawBoxViolation = NaN; return
end
x = x(:);
C.rawBoxViolation = max([P.xmin - x; x - P.xmax; 0]);
if C.rawBoxViolation > 1e-9
    C.reason = sprintf('raw box violation %.3e > 1e-9', C.rawBoxViolation); return
end
x = min(P.xmax, max(P.xmin, x));
C.clipBoxViolation = max([P.xmin - x; x - P.xmax; 0]);

[fval, dfdx] = P.evalProd(x);
C.fval = fval.'; C.maxRow = max(fval);
s = P.Ac*x - P.bc; ns = norm(s);
C.coneNorm = ns; C.apex = 2*ns <= 1e-9;
C.bs = x(end); C.primal = P.f.'*x;
lo = P.xmin; hi = P.xmax; width = hi - lo;
sRow0 = sqrt(mean((P.F11/P.lamref).^2)); C.sRow0 = sRow0;

cands = struct('name',{},'p',{},'mu',{},'nu',{});
mu0 = 0; nu0 = [0;0];
if ~isempty(la) && isstruct(la)
    ms = la.soc; if iscell(ms), ms = ms{1}; end
    if ~isempty(ms) && isfinite(ms(1)), mu0 = max(ms(1),0); end
    if isfield(la,'ineqlin') && numel(la.ineqlin) == 2 && all(isfinite(la.ineqlin))
        nu0 = max(la.ineqlin(:),0);
    end
end
if ~C.apex
    w = s/ns;
    % (i) solver duals, cone direction w
    cands(end+1) = struct('name','solverDuals','p',mu0*w,'mu',mu0,'nu',nu0);
    % (ii) complementary slackness: q(interior) = 0 and q(bs) = 0, (mu,nu) >= 0
    gmu = P.Ac.'*w - P.dc;
    inter = [x(1:NE) > lo(1:NE) + 1e-6*width(1:NE) & x(1:NE) < hi(1:NE) - 1e-6*width(1:NE); true];
    Mx = [gmu(inter), P.Alin(:,inter).'];
    z = lsqnonneg(Mx, -P.f(inter));
    cands(end+1) = struct('name','complementarySlackness','p',z(1)*w,'mu',z(1),'nu',z(2:3));
end
% (iii)/(iv) frozen fp_dualbound (byte-identical copy)
D = fp_dualbound(P, x, mu0, nu0);
cands(end+1) = struct('name','dualboundGeneral','p',D.p(:),'mu',D.mu,'nu',D.nu(:));
cands(end+1) = struct('name','dualboundAligned','p',D.aligned.p(:),'mu',D.aligned.mu,'nu',D.aligned.nu(:));

R = struct('name',{},'dualBound',{},'gap',{},'dualFeasible',{},'rowComp',{},'boxComp',{}, ...
           'bsStat',{},'statRms',{},'statMax',{},'pass',{},'fails',{},'mu',{},'nu',{});
C.certified = false;
for i = 1:numel(cands)
    c = cands(i); p = c.p(:); nu = c.nu(:);
    mu = max(c.mu, norm(p));                    % project onto mu >= ||p|| (roundoff of ||w||)
    dualFeasible = mu >= norm(p) && all(nu >= 0) && isfinite(mu) && all(isfinite(p));
    q = P.f + P.Ac.'*p - mu*P.dc + P.Alin.'*nu;
    Dval = sum(min(q.*P.xmin, q.*P.xmax)) - p.'*P.bc + mu*P.gammac - nu.'*P.blin;
    gap = C.primal - Dval;
    muRows = [mu; 0; nu];
    if C.apex
        qK = q;                                 % generalized conic subgradient
    else
        qK = P.f + dfdx.'*muRows;               % production row gradients
    end
    xi = [max(qK(1:NE),0); 0]; et = [max(-qK(1:NE),0); 0];
    rowComp = max(abs(muRows.*fval));
    boxComp = max([xi(1:NE).*(x(1:NE)-lo(1:NE)); et(1:NE).*(hi(1:NE)-x(1:NE))])/(sRow0*P.move);
    bsStat = abs(qK(end));
    gL = qK(1:NE) - xi(1:NE) + et(1:NE);
    statRms = sqrt(mean(gL.^2))/sRow0; statMax = max(abs(gL))/sRow0;
    bars = struct('C1', C.maxRow <= 1e-8, 'C2', C.rawBoxViolation <= 1e-9 && C.clipBoxViolation == 0, ...
        'C3', gap >= -1e-8 && gap <= 1e-8, 'C4', dualFeasible, 'C5', rowComp <= 1e-6, ...
        'C6', boxComp <= 1e-4, 'C7', bsStat <= 1e-6, 'C8', statRms <= 1e-6 && statMax <= 1e-5);
    fn = fieldnames(bars); fails = fn(~cellfun(@(f) bars.(f), fn));
    pass = isempty(fails);
    R(end+1) = struct('name',c.name,'dualBound',Dval,'gap',gap,'dualFeasible',dualFeasible, ...
        'rowComp',rowComp,'boxComp',boxComp,'bsStat',bsStat,'statRms',statRms,'statMax',statMax, ...
        'pass',pass,'fails',strjoin(fails.',','),'mu',mu,'nu',nu.'); %#ok<AGROW>
    if pass && ~C.certified
        C.certified = true; C.candidate = c.name;
        C.accepted = R(end); C.mu = mu; C.nu = nu; C.p = p;
    end
end
C.candidates = R;
[C.bestDualBound, ib] = max([R.dualBound]);
C.bestGap = C.primal - C.bestDualBound; C.bestGapCandidate = R(ib).name;
if ~C.certified
    C.reason = sprintf('no candidate passes: %s', strjoin(arrayfun(@(r) sprintf('%s[%s]', r.name, r.fails), R, 'UniformOutput', false), ' '));
end
if ~C.apex
    % fp_kkt (byte-identical copy) with the accepted or best multipliers, recorded only
    if C.certified, mR = [C.mu; 0; C.nu]; else, mR = [R(ib).mu; 0; R(ib).nu(:)]; end
    qK = P.f + dfdx.'*mR;
    K = fp_kkt(P, x, mR, [max(qK(1:NE),0);0], [max(-qK(1:NE),0);0], 'cs certificate multipliers');
    C.fp_kkt = rmfield(K, 'masks');
end
end
