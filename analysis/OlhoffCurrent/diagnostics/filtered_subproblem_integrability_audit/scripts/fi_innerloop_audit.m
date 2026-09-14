function [drho, st, rec] = fi_innerloop_audit(ctx, tolOverride, lean)
%FI_INNERLOOP_AUDIT  AUDIT-ONLY mirror of +impl/algo/innerLoop that RETAINS the
%   MMA dual state innerLoop discards.
%
%   PROVENANCE.  Every numeric expression below is copied character-for-
%   character from +impl/algo/innerLoop.m.  The ONLY differences are:
%     (1) the mmasub outputs lam, xsi, eta, mu, zet, s -- which production
%         discards with ~ -- are captured into `rec`;
%     (2) an optional tolInner override for the preregistered certification
%         re-solve (AUDIT_PREREGISTRATION sec. 4).
%   No constraint, bound, scaling, asymptote or MMA constant is altered, and
%   nothing here writes to any density.  The returned drho is NEVER applied.

if nargin < 2 || isempty(tolOverride), tolInner = ctx.tolInner;
else,                                  tolInner = tolOverride; end
% lean = true keeps ONLY the last iterate's full state plus the scalar
% histories.  Retaining every iterate costs ~2.4 MB each, which is fine for the
% 19-iterate production replay and ruinous for a 5000-iterate certification.
if nargin < 3 || isempty(lean), lean = false; end

NE = numel(ctx.rho);
N  = numel(ctx.lam);
lamref = ctx.lam(1);
Vtot   = ctx.volfrac*NE;

nvar = NE + 1;
lo = max(ctx.rhomin - ctx.rho, -ctx.move);
hi = min(1          - ctx.rho,  ctx.move);
xmin = [lo; 0];
xmax = [hi; 5];

x = [zeros(NE,1); 1];
xold1 = x; xold2 = x;
low = xmin; upp = xmax;

if ctx.offDiag, m = N + 2;
else,           m = N + 2 + N*(N-1); end
a0 = 1; aMMA = zeros(m,1); cMMA = 1000*ones(m,1); dMMA = zeros(m,1);

if isfield(ctx,'dOff'), dOff = ctx.dOff; else, dOff = []; end

st = struct('nInner',0,'degenHits',0,'conv',false,'dxHist',[],'relHist',[], ...
            'relStepHist',[],'maxAbsDrhoHist',[]);
rec = struct('iter',{},'x',{},'fval',{},'f0val',{},'lam',{},'xsi',{},'eta',{}, ...
             'mu',{},'zet',{},'s',{},'ymma',{},'zmma',{},'low',{},'upp',{}, ...
             'dfdx',{},'df0dx',{},'xmin',{},'xmax',{},'dx',{},'relStep',{});

for it = 1:ctx.maxInner
    drho = x(1:NE);
    bs   = x(end);

    if ctx.offDiag
        [dlam, ddlam, ~, degen] = deltaLambda(ctx.F, drho, dOff);
    else
        ddlam = zeros(NE,N); dlam = zeros(N,1); degen = false;
        for j = 1:N
            ddlam(:,j) = ctx.F(:,j,j);
            dlam(j)    = ctx.F(:,j,j).'*drho;
        end
    end
    st.degenHits = st.degenHits + degen;

    fval = zeros(m,1);
    dfdx = zeros(m,nvar);
    for j = 1:N
        fval(j)      = bs - (ctx.lam(j) + dlam(j))/lamref;
        dfdx(j,1:NE) = -ddlam(:,j).'/lamref;
        dfdx(j,nvar) = 1;
    end
    fval(N+1)      = bs - (ctx.lamJ + ctx.fJJ.'*drho)/lamref;
    dfdx(N+1,1:NE) = -ctx.fJJ.'/lamref;
    dfdx(N+1,nvar) = 1;
    fval(N+2)      = (sum(ctx.rho + drho) - Vtot)/Vtot;
    dfdx(N+2,1:NE) = 1/Vtot;

    f0val  = -bs;
    df0dx  = zeros(nvar,1); df0dx(nvar) = -1;

    [xmma,ymma,zmma,lamD,xsi,eta,muD,zet,sD,low,upp] = mmasub(m,nvar,it,x,xmin,xmax, ...
        xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,aMMA,cMMA,dMMA);

    dx = max(abs(xmma(1:NE)-x(1:NE)));
    st.dxHist(end+1) = dx;                                          %#ok<AGROW>
    st.relHist(end+1) = dx / max(max(abs(xmma(1:NE))), 1e-12);      %#ok<AGROW>

    if lean, r = 1; else, r = numel(rec)+1; end
    rec(r).iter = it;          rec(r).x = x;
    rec(r).fval = fval;        rec(r).f0val = f0val;
    rec(r).lam = lamD;         rec(r).xsi = xsi;      rec(r).eta = eta;
    rec(r).mu  = muD;          rec(r).zet = zet;      rec(r).s   = sD;
    rec(r).ymma = ymma;        rec(r).zmma = zmma;
    rec(r).low = low;          rec(r).upp = upp;
    rec(r).dfdx = dfdx;        rec(r).df0dx = df0dx;
    rec(r).xmin = xmin;        rec(r).xmax = xmax;
    rec(r).dx = dx;
    rec(r).relStep = dx / max(max(abs(xmma(1:NE))), 1e-12);

    xold2 = xold1; xold1 = x; x = xmma;
    st.nInner = it;
    st.relStepHist(it,1) = dx / max(max(abs(xmma(1:NE))), 1e-12);
    st.maxAbsDrhoHist(it,1) = max(abs(xmma(1:NE)));
    relStep = dx / max(max(abs(xmma(1:NE))), 1e-12);
    if it >= ctx.minInner && relStep < tolInner
        st.conv = true;
        break
    end
end

drho = x(1:NE);
st.beta = x(end)*lamref;
st.xFinal = x;
st.tolUsed = tolInner;
end
