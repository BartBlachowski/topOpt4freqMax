function [drho, st, state] = innerLoopRho(ctx, state)
%INNERLOOPRHO  Step 3 of Fig. 1 in DESIGN coordinates, MMA state persistent.
%
%   Solves the SAME sub-optimization problem (25) as innerLoop.m -- independent
%   variables beta and drho, dependent DELTA(omega_j^2) from (25d) -- but with
%   the trivial affine reparametrization x = rho + drho, so that the MMA
%   variables are the design densities themselves, bounded by (25f) alone
%   (plus an OPTIONAL hard move box; pass ctx.move = inf for the pure (25f)
%   form, which is all the paper states).
%
%   The essential difference from innerLoop.m: the MMA solver state -- the
%   asymptotes low/upp, the history xold1/xold2, and the global iteration
%   counter -- PERSISTS across outer iterations, because each inner loop
%   starts from the design where the previous one ended.  Svanberg's asymptote
%   adaptation then acts as a per-element step control across the whole run:
%   elements that oscillate between outer iterations get their asymptotes
%   contracted (asydecr), elements moving monotonically get them expanded
%   (asyincr).  No explicit move limit is needed -- consistent with the paper
%   stating none -- and near the optimum the increments contract, which is
%   what lets the printed test ||drho|| < eps (Fig. 1) actually fire.
%
%   RECONSTRUCTION, not the authors' text: the paper specifies problem (25)
%   and MMA, but neither the coordinates nor the solver-state handling.
%   innerLoop.m (increment coordinates, state reset per outer iteration, hard
%   move box) is the other labelled realization of the same equations.
%
%   [drho, st, state] = innerLoopRho(ctx, state)
%     state : [] on the first outer iteration, then the struct returned by
%             the previous call.  Fields: itG, low, upp, xold1, xold2.
%
%   ctx.asyOuter (optional, default false) selects WHICH history adapts the
%   asymptotes:
%     false  the inner sub-iterate sequence, global counter (the behaviour
%            described above; the outer reversal rho_k+1 - rho_k is invisible
%            to Svanberg's rule because every inner loop starts at x = xold1).
%     true   the OUTER design sequence: at the first sub-iterate of outer
%            iteration k the asymptotes are formed by Svanberg's rule from
%            rho_k, rho_k-1, rho_k-2 and the asymptotes used at rho_k-1; they
%            are then HELD for the remaining sub-iterates of the frozen
%            sub-problem (xold1 = xold2 = x makes the update factor 1, only
%            the published clamps about the current iterate still act).  This
%            is the standard use of Svanberg (1987) across an outer loop and
%            is the mechanism by which ||drho|| can decay to zero near a
%            stationary point.  RECONSTRUCTION (class C): the paper names MMA
%            and the nested loop, and does not say how the two are joined.
%            state fields added: kOuter, rhoPrev1, rhoPrev2.

NE = numel(ctx.rho);
N  = numel(ctx.lam);
lamref = ctx.lam(1);
Vtot   = ctx.volfrac*NE;
nvar   = NE + 1;                       % [rho_new ; beta_scaled]

lo = max(ctx.rhomin, ctx.rho - ctx.move);    % ctx.move may be inf -> (25f) box
hi = min(1,          ctx.rho + ctx.move);
xmin = [lo; 0];
xmax = [hi; 5];

x = [ctx.rho; 1];                      % start at drho = 0, beta = lamref

asyOuter = isfield(ctx,'asyOuter') && ~isempty(ctx.asyOuter) && ctx.asyOuter;
if isempty(state) || ~isfield(state, 'itG')
    state = struct('itG', 0, 'low', xmin, 'upp', xmax, 'xold1', x, 'xold2', x, ...
                   'kOuter', 0, 'rhoPrev1', [], 'rhoPrev2', []);
end
if ~isfield(state,'kOuter'), state.kOuter = 0; state.rhoPrev1 = []; state.rhoPrev2 = []; end
low   = state.low;    upp   = state.upp;
if asyOuter
    kOut = state.kOuter + 1;               % this outer iteration, Svanberg's `iter`
    if kOut >= 3
        xold1 = [state.rhoPrev1; 1];  xold2 = [state.rhoPrev2; 1];
    else
        xold1 = x;  xold2 = x;
    end
    asyLow = low;  asyUpp = upp;
else
    xold1 = state.xold1;  xold2 = state.xold2;
end

if ctx.offDiag
    m = N + 2;                         % (25c) x N, (25b), (25e)
else
    m = N + 2 + N*(N-1);               % + eq. (22) as paired inequalities
end
a0 = 1; aMMA = zeros(m,1); cMMA = 1000*ones(m,1); dMMA = zeros(m,1);

st = struct('nInner',0,'degenHits',0,'conv',false,'dxHist',[],'relHist',[]);

for it = 1:ctx.maxInner
    drho = x(1:NE) - ctx.rho;
    bs   = x(end);

    % ---- dependent variables: (25d), erratum form ----------------------
    if ctx.offDiag
        [dlam, ddlam, ~, degen] = deltaLambda(ctx.F, drho);
    else
        ddlam = zeros(NE,N); dlam = zeros(N,1); degen = false;
        for j = 1:N
            ddlam(:,j) = ctx.F(:,j,j);
            dlam(j)    = ctx.F(:,j,j).'*drho;
        end
    end
    st.degenHits = st.degenHits + degen;

    % ---- constraints (identical to innerLoop.m, in terms of drho) -------
    fval = zeros(m,1);
    dfdx = zeros(m,nvar);
    for j = 1:N                        % (25c)
        fval(j)          = bs - (ctx.lam(j) + dlam(j))/lamref;
        dfdx(j,1:NE)     = -ddlam(:,j).'/lamref;
        dfdx(j,nvar)     = 1;
    end
    fval(N+1)        = bs - (ctx.lamJ + ctx.fJJ.'*drho)/lamref;    % (25b)
    dfdx(N+1,1:NE)   = -ctx.fJJ.'/lamref;
    dfdx(N+1,nvar)   = 1;
    fval(N+2)        = (sum(ctx.rho + drho) - Vtot)/Vtot;          % (25e)
    dfdx(N+2,1:NE)   = 1/Vtot;
    if ~ctx.offDiag                    % eq. (22) route
        r = N+2;
        for s = 1:N
            for k = s+1:N
                g  = ctx.F(:,s,k).'*drho/lamref;
                gs = ctx.F(:,s,k).'/lamref;
                r = r+1; fval(r) =  g; dfdx(r,1:NE) =  gs;
                r = r+1; fval(r) = -g; dfdx(r,1:NE) = -gs;
            end
        end
    end

    f0val  = -bs;
    df0dx  = zeros(nvar,1); df0dx(nvar) = -1;

    if asyOuter
        if it == 1
            itArg = kOut;                  % outer history forms the asymptotes
        else
            itArg = 3;  xold1 = x;  xold2 = x;   % factor 1: asymptotes HELD
        end
    else
        itArg = state.itG + it;            % GLOBAL counter: asymptotes persist
    end
    [xmma,~,~,~,~,~,~,~,~,low,upp] = mmasub(m,nvar,itArg,x,xmin,xmax, ...
        xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,aMMA,cMMA,dMMA);
    if asyOuter && it == 1
        asyLow = low;  asyUpp = upp;       % the asymptotes formed AT rho_k
    end

    dx = max(abs(xmma(1:NE)-x(1:NE)));
    st.dxHist(end+1) = dx;                                          %#ok<AGROW>
    if ~asyOuter, xold2 = xold1; xold1 = x; end
    x = xmma;
    st.nInner = it;
    relStep = dx / max(max(abs(x(1:NE)-ctx.rho)), 1e-12);
    st.relHist(end+1) = relStep;                                    %#ok<AGROW>
    if it >= ctx.minInner && relStep < ctx.tolInner
        st.conv = true;
        break
    end
end

drho    = x(1:NE) - ctx.rho;
st.beta = x(end)*lamref;

state.itG   = state.itG + st.nInner;
if asyOuter
    state.kOuter   = kOut;
    state.rhoPrev2 = state.rhoPrev1;
    state.rhoPrev1 = ctx.rho;              % the design this outer iteration started from
    state.low = asyLow;  state.upp = asyUpp;
    state.xold1 = x;     state.xold2 = x;
else
    state.low   = low;    state.upp   = upp;
    state.xold1 = xold1;  state.xold2 = xold2;
end
end
