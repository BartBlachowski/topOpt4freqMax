function res = du_olhoff(prob, opts)
%DU_OLHOFF  Bound formulation of Du & Olhoff (2007) for maximizing the fundamental
%   eigenfrequency of a 2D plane-stress continuum (the reference method of the paper).
%
%   res = du_olhoff(prob)          prob from problem_ss_beam / problem_clamped_beam /
%   res = du_olhoff(prob, opts)    problem_building (configs/)
%
%   Reference: J. Du, N. Olhoff, "Topological design of freely vibrating continuum
%   structures for maximum values of simple and multiple eigenfrequencies and
%   frequency gaps", Struct. Multidisc. Optim. 34 (2007) 91-110 [DO07 below].
%
%   What is implemented (Section 3.1 and Algorithm 1 of our paper):
%     - nested loop of [DO07] Fig. 1: one eigenanalysis per OUTER iteration, the
%       design increments from the bound sub-problem [DO07] eq. (25) solved by MMA
%       (Svanberg 1987, published 2007 version) in an INNER loop;
%     - target mode n = 1 with a FIXED two-mode subspace (N = 2, J = n + N = 3):
%       generalized gradients f_sk of [DO07] eq. (19), including the off-diagonal
%       terms, enter the sub-eigenvalue problem [DO07] eq. (24) with the actual
%       separation lambda_j - lambda_1 on its diagonal (exact at a double eigenvalue);
%     - SIMP stiffness, linearized below x = 0.1 as proposed by Pedersen (2000) and
%       named by [DO07] Sec. 2.2 as the alternative to their mass cut-off eq. (4);
%       linear mass [DO07] eq. (2) with q = 1;
%     - sensitivity filter of Sigmund (1997) on every f_sk, physical radius R;
%     - per-element adaptive move limit 0.002 <= delta_e <= 0.1 (x1.2 / x0.7);
%     - stop: the common stagnation_x criterion of our Section 2.5.
%   Extension used only for the building of Section 4.4 (not in [DO07]): passive
%   solid elements (x = 1, excluded from the sub-problem, counted in the volume) and
%   a clamped base.
%
%   This file is a compact, line-by-line port of the production path of the research
%   code (analysis/Olhoff, tag applsci-revision-1); tests/test_du_olhoff_port.m checks
%   that both give the same designs.
%
%   opts (all optional)
%     filterRadius  physical filter radius R [m]          (default 0.06)
%     maxOuter      safety cap on outer iterations        (default 1000)
%     stopTol       eps_x = eps_M of stagnation_x         (default 1e-2)
%     stopWindow    W of stagnation_x                     (default 10)
%     stopRule      'stagnation_x' (default; all comparisons of the paper) | 'native':
%                   ||drho||_2 < 0.05 sqrt(Ne/3200), the design-increment test of
%                   [DO07] Sec. 3.5.1 with a mesh-scaled tolerance (Section 3.1 check)
%     verbose       print one line per outer iteration    (default false)
%
%   res fields
%     x        final densities in the common element numbering (see fe_mesh)
%     omega    omega_1..omega_5 of the final design in the solver's own material model
%     nOuter, nInner, status ('CONVERGED' | 'CAP_HIT')
%     time     struct: total, eig, grad, inner, outerExclInner [s]
%     hist     per outer iteration: omega(1:5,k), nInner(k), move(k), vol(k)

if nargin < 2, opts = struct(); end
R       = getopt(opts, 'filterRadius', 0.06);
maxOuter= getopt(opts, 'maxOuter', 1000);
tolStop = getopt(opts, 'stopTol', 1e-2);
W       = getopt(opts, 'stopWindow', 10);
verbose = getopt(opts, 'verbose', false);
stopRule= getopt(opts, 'stopRule', 'stagnation_x');
tStart = tic;

% ---- fixed scientific settings of the paper (Section 3.1) ------------------
n      = 1;          % target eigenfrequency
Nsub   = 2;          % fixed subspace size of the bound sub-problem
Jcalc  = n + 4;      % eigenpairs computed per outer iteration (J = 3 are used)
p      = 3;          % SIMP exponent
xLin   = 0.1;        % Pedersen: stiffness linear below this density
rhomin = 1e-3;       % lower box bound of [DO07] eq. (25f)
move0  = 0.1;  moveMin = 0.002;  grow = 1.2;  shrink = 0.7;
tolInner = 0.05;  minInner = 5;  maxInner = 500;

mdl = model2D(prob);
NE  = mdl.nele;
pas = toNative(prob.passive(:), prob.nelx, prob.nely);   % passive solid, native numbering
act = setdiff((1:NE)', pas);
volfrac = prob.volfrac;
rho = 0.5*ones(NE,1);              % uniform start x = 0.5 for every V_f, as in [DO07] Fig. 3
if ~isempty(pas)                                         % building: frame solid, rest uniform
    volfracAct = (volfrac*NE - numel(pas))/numel(act);
    rho(act) = volfracAct;  rho(pas) = 1;
else
    volfracAct = volfrac;
end
flt = prepFilter(prob.nelx, prob.nely, R/(prob.H/prob.nely));

hist = struct('omega', [], 'nInner', [], 'move', [], 'vol', []);
tEig = 0; tGrad = 0; tInner = 0;
dvec = []; rhoPrev1 = []; rhoPrev2 = [];
relCh = []; grey = []; rhoAnalysedPrev = [];
status = 'CAP_HIT';

for outer = 1:maxOuter
    % ---- Step 1 of [DO07] Fig. 1: eigenanalysis, M-orthonormal modes ----------
    te = tic;
    [K, M] = assemble2D(mdl, rho, p, xLin);
    [w, Phi, lam] = eigSolve(K, M, Jcalc);
    tEig = tEig + toc(te);
    % stagnation_x bookkeeping on the analysed design (Section 2.5 of the paper)
    grey(outer) = 4*mean(rho.*(1-rho));                      %#ok<AGROW>
    relCh(outer) = NaN;                                      %#ok<AGROW>
    if ~isempty(rhoAnalysedPrev)
        relCh(outer) = norm(rho - rhoAnalysedPrev)/max(norm(rhoAnalysedPrev), realmin);
    end
    rhoAnalysedPrev = rho;

    % ---- Step 2: generalized gradients [DO07] eq. (19) ------------------------
    tg = tic;
    J   = n + Nsub;                       % simple mode bounding the cluster, (25b)
    idx = n:(n+Nsub-1);
    F = genGrad(mdl, rho, p, xLin, Phi, lam(n), idx);     % lambda~ = lambda_n (Sec. 3.5.1)
    for j = 1:Nsub                        % diagonal blocks with each mode's own lambda_j
        Gj = genGrad(mdl, rho, p, xLin, Phi, lam(idx(j)), idx(j));
        F(:,j,j) = Gj(:,1,1);
    end
    dOff = lam(idx) - lam(idx(1));        % separations kept in the sub-eigenvalue problem (24)
    FJ  = genGrad(mdl, rho, p, xLin, Phi, lam(J), J);
    fJJ = FJ(:,1,1);
    if ~isempty(pas), F(pas,:,:) = 0; fJJ(pas) = 0; end
    for s = 1:Nsub                        % Sigmund (1997) filter on every f_sk
        for k = s:Nsub
            v = applyFilter(flt, rho, F(:,s,k));
            F(:,s,k) = v;  F(:,k,s) = v;
        end
    end
    fJJ = applyFilter(flt, rho, fJJ);
    tGrad = tGrad + toc(tg);

    % ---- adaptive move limit (per element; not specified by [DO07]) ----------
    if isempty(dvec), dvec = move0*ones(NE,1); end
    if ~isempty(rhoPrev2)
        zzz = (rho - rhoPrev1).*(rhoPrev1 - rhoPrev2);
        f = ones(NE,1);  f(zzz > 0) = grow;  f(zzz < 0) = shrink;
        dvec = min(move0, max(moveMin, f.*dvec));
    end
    rhoPrev2 = rhoPrev1;  rhoPrev1 = rho;

    % ---- Step 3: inner loop, bound sub-problem [DO07] eq. (25) by MMA ---------
    ti = tic;
    ctx = struct('F', F(act,:,:), 'fJJ', fJJ(act), 'lam', lam(idx), 'lamJ', lam(J), ...
        'rho', rho(act), 'rhomin', rhomin, 'volfrac', volfracAct, 'move', dvec(act), ...
        'maxInner', maxInner, 'tolInner', tolInner, 'minInner', minInner, 'dOff', dOff);
    [dAct, st] = innerLoop(ctx);
    drho = zeros(NE,1);  drho(act) = dAct;
    tInner = tInner + toc(ti);

    % ---- Step 4: update ------------------------------------------------------
    rho = min(1, max(rhomin, rho + drho));
    hist.omega(:,outer) = w(1:Jcalc);
    hist.nInner(outer)  = st.nInner;
    hist.move(outer)    = max(dvec);
    hist.vol(outer)     = mean(rho);
    if verbose
        fprintf('%4d  omega1..3 = %8.2f %8.2f %8.2f  inner %3d  vol %.4f\n', ...
            outer, w(1), w(2), w(3), st.nInner, mean(rho));
    end
    if strcmp(stopRule, 'native'), conv = norm(drho) < 0.05*sqrt(NE/3200);
    else,                          conv = stagnationX(relCh, grey, outer, W, tolStop, tolStop); end
    if conv
        status = 'CONVERGED';
        break
    end
end

% ---- final eigenanalysis of the returned design ----------------------------
[K, M] = assemble2D(mdl, rho, p, xLin);
w = eigSolve(K, M, Jcalc);
res.x      = toCommon(rho, prob.nelx, prob.nely);
res.xNative= rho;
res.omega  = w;
res.nOuter = numel(hist.nInner);
res.nInner = sum(hist.nInner);
res.status = status;
res.hist   = hist;
tTot = toc(tStart);
res.time = struct('total', tTot, 'eig', tEig, 'grad', tGrad, 'inner', tInner, ...
    'outerExclInner', tTot - tInner);
end

% =============================================================================
%  Finite-element model (top88 numbering: element e = (elx-1)*nely + ely, ely
%  counted from the TOP edge; nodes column-wise from the top-left corner)
% =============================================================================
function mdl = model2D(prob)
nelx = prob.nelx;  nely = prob.nely;
dx = prob.L/nelx;  dy = prob.H/nely;
nele = nelx*nely;  nnode = (nelx+1)*(nely+1);  ndof = 2*nnode;
nodenrs = reshape(1:nnode, 1+nely, 1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1, nele, 1);
edofMat = repmat(edofVec,1,8) + repmat([0 1 2*nely+[2 3 0 1] -2 -1], nele, 1);
[K0, M0] = elemMats(dx, dy, prob.E, prob.nu, prob.rho, 1);
dofsOf = @(nodes, comp) 2*nodes(:) - 2 + comp;
clampFace = @(col) reshape([dofsOf(nodenrs(:,col),1); dofsOf(nodenrs(:,col),2)],[],1);
switch prob.support
    case 'simplySupported'     % pins at mid-height of both end faces, ux fixed at both
        rowSS = nely/2 + 1;    % ([DO07] Fig. 2a; the published 68.7 rad/s fits mid-height)
        fixed = [dofsOf(nodenrs(rowSS,1),2); dofsOf(nodenrs(rowSS,end),2); ...
                 dofsOf(nodenrs(rowSS,1),1); dofsOf(nodenrs(rowSS,end),1)];
    case 'clamped'             % both end faces clamped ([DO07] Fig. 2c)
        fixed = [clampFace(1); clampFace(nelx+1)];
    case 'clampedBottom'       % building: every node of the bottom edge clamped
        fixed = reshape([dofsOf(nodenrs(end,:),1); dofsOf(nodenrs(end,:),2)],[],1);
    otherwise
        error('du_olhoff:support', 'unknown support %s', prob.support);
end
fixed = unique(fixed(:));
free  = setdiff((1:ndof)', fixed);
% lower-triangle assembly data: each element matrix enters once, then mirrored
ltMask = tril(true(8));
[ltRow, ltCol] = find(ltMask);
iL = reshape(edofMat(:,ltRow)',[],1);
jL = reshape(edofMat(:,ltCol)',[],1);
sw = iL < jL;  [iL(sw), jL(sw)] = deal(jL(sw), iL(sw));
K0s = (K0+K0')/2;  M0s = (M0+M0')/2;
mdl = struct('nele',nele,'ndof',ndof,'edofMat',edofMat,'K0',K0,'M0',M0,'free',free, ...
             'iL',iL,'jL',jL,'K0L',K0s(ltMask),'M0L',M0s(ltMask));
end

function [K0, M0] = elemMats(dx, dy, E, nu, rhom, t)
% Bilinear Q4 plane-stress stiffness and consistent mass, 2x2 Gauss quadrature.
D = E/(1-nu^2) * [1 nu 0; nu 1 0; 0 0 (1-nu)/2];
g = 1/sqrt(3);  gp = [-g -g; g -g; g g; -g g];
detJ = dx*dy/4;  dxi_dx = 2/dx;  deta_dy = 2/dy;
xn = [-1 1 1 -1];  yn = [-1 -1 1 1];
K0 = zeros(8);  Mc = zeros(8);
for q = 1:4
    xi = gp(q,1); eta = gp(q,2);
    N     = 0.25*(1+xn*xi).*(1+yn*eta);
    dNdx  = 0.25*xn.*(1+yn*eta)*dxi_dx;
    dNdy  = 0.25*(1+xn*xi).*yn*deta_dy;
    B = zeros(3,8);
    B(1,1:2:7) = dNdx;  B(2,2:2:8) = dNdy;
    B(3,1:2:7) = dNdy;  B(3,2:2:8) = dNdx;
    K0 = K0 + B'*D*B*detJ*1*t;
    Nm = zeros(2,8);  Nm(1,1:2:7) = N;  Nm(2,2:2:8) = N;
    Mc = Mc + Nm'*Nm*rhom*detJ*1*t;
end
M0 = 0.5*(Mc+Mc');
end

function [gK, dgK] = stiffness(rho, p, xLin)
% SIMP rho^p, linear below xLin with the same value at xLin (Pedersen 2000).
gK = rho.^p;  dgK = p*rho.^(p-1);
lo = rho < xLin;  c = xLin^(p-1);
gK(lo) = c*rho(lo);  dgK(lo) = c;
end

function [K, M] = assemble2D(mdl, rho, p, xLin)
% Global stiffness and mass on the free dofs; mass linear in rho ([DO07] eq. (2)).
rho = rho(:);
gK = stiffness(rho, p, xLin);
sK = mdl.K0L * gK';  sM = mdl.M0L * rho';
K = sparse(mdl.iL, mdl.jL, sK(:), mdl.ndof, mdl.ndof);
M = sparse(mdl.iL, mdl.jL, sM(:), mdl.ndof, mdl.ndof);
K = K + tril(K,-1)';  M = M + tril(M,-1)';
K = K(mdl.free, mdl.free);  M = M(mdl.free, mdl.free);
end

function [omega, Phi, lambda] = eigSolve(K, M, J)
% J lowest eigenpairs (shift-invert Arnoldi), deterministic start vector, M-normalized.
nd = size(K,1);
v0 = sin((1:nd)'*0.7071067811865476) + 0.5;
[V, Dm, flag] = eigs(K, M, J, 'smallestabs', struct('v0', v0, 'tol', 1e-12, ...
    'maxit', 5000, 'p', min(nd, max(20, 4*J))));
if flag ~= 0, error('du_olhoff:eigs', 'eigs did not converge'); end
[lambda, ix] = sort(real(diag(Dm)), 'ascend');
Phi = V(:, ix);
for j = 1:size(Phi,2)
    Phi(:,j) = Phi(:,j)/sqrt(Phi(:,j)'*M*Phi(:,j));
end
omega = sqrt(max(lambda,0));
end

function F = genGrad(mdl, rho, p, xLin, Phi, lamTilde, idx)
% Generalized gradients [DO07] eq. (19):
%   f_sk,e = phi_s' (dK/drho_e - lamTilde dM/drho_e) phi_k ,  F(:,s,k) = f_sk.
nI = numel(idx);  NE = mdl.nele;
Ufull = zeros(mdl.ndof, nI);  Ufull(mdl.free,:) = Phi(:, idx);
[~, sK] = stiffness(rho(:), p, xLin);
sM = lamTilde * ones(NE,1);                 % d(mass)/drho = 1 for linear mass
F = zeros(NE, nI, nI);
Ue = cell(nI,1);
for s = 1:nI, Ue{s} = reshape(Ufull(mdl.edofMat(:), s), NE, 8); end
for s = 1:nI
    UsK = Ue{s}*mdl.K0;  UsM = Ue{s}*mdl.M0;
    for k = s:nI
        f = sK.*sum(UsK .* Ue{k}, 2) - sM.*sum(UsM .* Ue{k}, 2);
        F(:,s,k) = f;  F(:,k,s) = f;
    end
end
end

function flt = prepFilter(nelx, nely, rmin)
% Weights of the Sigmund (1997) filter, rmin in element units.
iH = ones(nelx*nely*(2*(ceil(rmin)-1)+1)^2,1);  jH = ones(size(iH));  sH = zeros(size(iH));
k = 0;
for i1 = 1:nelx
  for j1 = 1:nely
    e1 = (i1-1)*nely+j1;
    for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
      for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
        k = k+1;  iH(k) = e1;  jH(k) = (i2-1)*nely+j2;
        sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
      end
    end
  end
end
H = sparse(iH,jH,sH);
flt = struct('H',H,'Hs',sum(H,2));
end

function df = applyFilter(flt, rho, df)
rho = rho(:);
df = (flt.H * (rho .* df)) ./ (flt.Hs .* max(1e-3, rho));
end

function [drho, st] = innerLoop(ctx)
% Bound sub-problem [DO07] eq. (25) in the increments drho and the scaled bound
% beta/lambda_1, solved by MMA:
%   max beta  s.t.  beta <= lambda_j + dlambda_j(drho),  j in the cluster   (25c)
%                   beta <= lambda_J + f_JJ'drho                           (25b)
%                   sum(rho + drho) <= V_f N_e                             (25e)
%                   rhomin <= rho + drho <= 1,  |drho_e| <= delta_e        (25f)
% dlambda_j are the eigenvalues of the sub-eigenvalue problem (24) (deltaLambda).
% Inner stop: largest sub-iteration step below tolInner times the largest
% accumulated increment, after at least minInner sub-iterations (our choice;
% [DO07] does not state a criterion).
NE = numel(ctx.rho);  N = numel(ctx.lam);  lamref = ctx.lam(1);
Vtot = ctx.volfrac*NE;  nvar = NE + 1;
lo = max(ctx.rhomin - ctx.rho, -ctx.move);
hi = min(1 - ctx.rho, ctx.move);
xmin = [lo; 0];  xmax = [hi; 5];
x = [zeros(NE,1); 1];                         % drho = 0, beta = lambda_1
xold1 = x;  xold2 = x;  low = xmin;  upp = xmax;
m = N + 2;
a0 = 1; aMMA = zeros(m,1); cMMA = 1000*ones(m,1); dMMA = zeros(m,1);
st = struct('nInner', 0);
for it = 1:ctx.maxInner
    drho = x(1:NE);  bs = x(end);
    [dlam, ddlam] = deltaLambda(ctx.F, drho, ctx.dOff);
    fval = zeros(m,1);  dfdx = zeros(m,nvar);
    for j = 1:N                                             % (25c)
        fval(j) = bs - (ctx.lam(j) + dlam(j))/lamref;
        dfdx(j,1:NE) = -ddlam(:,j).'/lamref;  dfdx(j,nvar) = 1;
    end
    fval(N+1) = bs - (ctx.lamJ + ctx.fJJ.'*drho)/lamref;    % (25b)
    dfdx(N+1,1:NE) = -ctx.fJJ.'/lamref;  dfdx(N+1,nvar) = 1;
    fval(N+2) = (sum(ctx.rho + drho) - Vtot)/Vtot;          % (25e)
    dfdx(N+2,1:NE) = 1/Vtot;
    f0val = -bs;  df0dx = zeros(nvar,1);  df0dx(nvar) = -1;
    [xmma,~,~,~,~,~,~,~,~,low,upp] = mmasub(m,nvar,it,x,xmin,xmax, ...
        xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,aMMA,cMMA,dMMA);
    dx = max(abs(xmma(1:NE)-x(1:NE)));
    xold2 = xold1;  xold1 = x;  x = xmma;
    st.nInner = it;
    if it >= ctx.minInner && dx/max(max(abs(xmma(1:NE))), 1e-12) < ctx.tolInner
        break
    end
end
drho = x(1:NE);
end

function [dlam, ddlam] = deltaLambda(F, drho, dOff)
% Increments of the clustered eigenvalues from the sub-eigenvalue problem [DO07]
% eq. (24): eigenvalues of A_sk = f_sk'drho + delta_sk (lambda_s - lambda_1),
% minus the offsets, and their gradients sum_sk v_s v_k f_sk.
[NE, N, ~] = size(F);
A = zeros(N);
for s = 1:N
    for k = s:N
        a = F(:,s,k).'*drho;  A(s,k) = a;  A(k,s) = a;
    end
end
A = (A+A')/2 + diag(dOff(:));
[V, D] = eig(A);
[dlam, ord] = sort(real(diag(D)),'ascend');
V = V(:,ord);
dlam = dlam - dOff(:);
for j = 1:N, V(:,j) = V(:,j)/norm(V(:,j)); end
ddlam = zeros(NE, N);
for j = 1:N
    vj = V(:,j);  g = zeros(NE,1);
    for s = 1:N
        for k = 1:N
            w = vj(s)*vj(k);
            if w ~= 0, g = g + w*F(:,s,k); end
        end
    end
    ddlam(:,j) = g;
end
end

% =============================================================================
function met = stagnationX(d, m, k, W, relTol, mndTol)
% stagnation_x of the paper (Section 2.5) on the analysed designs k-W..k.
met = false;
if k - W < 2, return; end
met = max(d(k-W+1:k)) < relTol && (max(m(k-W:k)) - min(m(k-W:k))) < mndTol;
end

function idx = toNative(e, nelx, nely)
% common numbering (ely from the bottom) -> top88 numbering (ely from the top)
m = false(nelx*nely,1);  m(e) = true;
m = flipud(reshape(m, nely, nelx));
idx = find(m(:));
end

function x = toCommon(xNative, nelx, nely)
x = reshape(flipud(reshape(xNative, nely, nelx)), [], 1);
end

function v = getopt(s, f, d)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
