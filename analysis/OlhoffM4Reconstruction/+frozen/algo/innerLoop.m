function [drho, st] = innerLoop(ctx)
%INNERLOOP  Step 3 of Fig. 1: solve sub-optimization problem (25) for the
%   design increments drho, with all outer-loop iterates held fixed.
%
%   Independent variables : beta and drho_e            (paper sec. 3.5.2)
%   Dependent variables   : DELTA(omega_j^2) from (25d)
%
%   ctx fields:
%     F        NE x N x N  generalized gradients for the n..n+N-1 set (filtered)
%     fJJ      NE x 1      gradient for the simple mode J = n+N        (25b)
%     lam      N x 1       current lambda_j, j = n..n+N-1
%     lamJ     scalar      current lambda_J
%     rho, rhomin, volfrac, Ve, move
%     maxInner, tolInner, minInner
%       The paper states only "Increments drho_e converged?" (Fig. 1) and never
%       gives a criterion.  An ABSOLUTE test on the MMA sub-iterate step is
%       unsafe: that step scales with the move limit, so a fixed tolerance
%       silently turns the inner loop into a no-op once the move limit is small
%       (measured: at move=0.01 it exits after one sub-iterate having travelled
%       4.6e-5).  A test relative to the move limit fails for the same reason,
%       because the step size GROWS before it decays.  The criterion used here
%       is relative to the accumulated increment,
%           max|dx_step| / max|drho|  <  tolInner ,
%       which is invariant to both the move limit and the problem scale, with
%       minInner sub-iterates always taken.  RECONSTRUCTION, not the authors'.
%     offDiag  true  -> full nonlinear coupling (25d)          [BASELINE]
%              false -> impose f_sk'drho = 0, s~=k  (Krog & Olhoff 1999 LP route)
%     dOff     OPTIONAL N x 1 eigenvalue offsets lambda_j - lambda_n, passed
%              straight to deltaLambda.  Absent/empty -> (25d) exactly as
%              printed (the frozen path).  See deltaLambda's header and
%              audit_multiplicity_reconstruction/WP3_candidates.md.
%
%   st records inner-loop statistics for the efficiency study.

NE = numel(ctx.rho);
N  = numel(ctx.lam);
lamref = ctx.lam(1);
Vtot   = ctx.volfrac*NE;             % all elements have equal volume

nvar = NE + 1;                       % [drho ; beta_scaled]
lo = max(ctx.rhomin - ctx.rho, -ctx.move);
hi = min(1          - ctx.rho,  ctx.move);
xmin = [lo; 0];
xmax = [hi; 5];

x = [zeros(NE,1); 1];                % drho = 0, beta = lambda_n
xold1 = x; xold2 = x;
low = xmin; upp = xmax;

if ctx.offDiag
    m = N + 2;                       % (25c) x N, (25b), (25e)
else
    m = N + 2 + N*(N-1);             % the equalities f_sk'drho = 0 as two-sided
end
a0 = 1; aMMA = zeros(m,1); cMMA = 1000*ones(m,1); dMMA = zeros(m,1);

if isfield(ctx,'dOff'), dOff = ctx.dOff; else, dOff = []; end

st = struct('nInner',0,'degenHits',0,'conv',false,'dxHist',[],'relHist',[]);

for it = 1:ctx.maxInner
    drho = x(1:NE);
    bs   = x(end);

    % ---- dependent variables: (25d) ------------------------------------
    if ctx.offDiag
        [dlam, ddlam, ~, degen] = deltaLambda(ctx.F, drho, dOff);
    else
        % off-diagonal terms forced to vanish -> (23): dlam_j = f_jj' drho
        ddlam = zeros(NE,N); dlam = zeros(N,1); degen = false;
        for j = 1:N
            ddlam(:,j) = ctx.F(:,j,j);
            dlam(j)    = ctx.F(:,j,j).'*drho;
        end
    end
    st.degenHits = st.degenHits + degen;

    % ---- constraints ----------------------------------------------------
    fval = zeros(m,1);
    dfdx = zeros(m,nvar);

    % (25c)  beta - [omega_j^2 + Delta(omega_j^2)] <= 0
    for j = 1:N
        fval(j)          = bs - (ctx.lam(j) + dlam(j))/lamref;
        dfdx(j,1:NE)     = -ddlam(:,j).'/lamref;
        dfdx(j,nvar)     = 1;
    end
    % (25b)  beta - [omega_J^2 + f_JJ' drho] <= 0
    fval(N+1)        = bs - (ctx.lamJ + ctx.fJJ.'*drho)/lamref;
    dfdx(N+1,1:NE)   = -ctx.fJJ.'/lamref;
    dfdx(N+1,nvar)   = 1;
    % (25e)  volume
    fval(N+2)        = (sum(ctx.rho + drho) - Vtot)/Vtot;
    dfdx(N+2,1:NE)   = 1/Vtot;

    % Enforced vanishing off-diagonals -- eq. (22), the Krog & Olhoff (1999) LP
    % route.  Imposed as a PAIR of inequalities  +g <= 0, -g <= 0  rather than
    % via abs(), which would reintroduce a non-smooth point at g = 0.
    if ~ctx.offDiag
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

    [xmma,~,~,~,~,~,~,~,~,low,upp] = mmasub(m,nvar,it,x,xmin,xmax, ...
        xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,aMMA,cMMA,dMMA);

    dx = max(abs(xmma(1:NE)-x(1:NE)));
    st.dxHist(end+1) = dx;                                          %#ok<AGROW>
    st.relHist(end+1) = dx / max(max(abs(xmma(1:NE))), 1e-12);      %#ok<AGROW>
    xold2 = xold1; xold1 = x; x = xmma;
    st.nInner = it;
    relStep = dx / max(max(abs(xmma(1:NE))), 1e-12);
    if it >= ctx.minInner && relStep < ctx.tolInner
        st.conv = true;
        break
    end
end

drho = x(1:NE);
st.beta = x(end)*lamref;
end
