function res = yuksel_freq(prob, opts)
%YUKSEL_FREQ  Two-stage static method of Yuksel & Yilmaz (2025), Algorithm 2 of the paper.
%
%   res = yuksel_freq(prob)          simply supported beam only (configs/problem_ss_beam)
%   res = yuksel_freq(prob, opts)
%
%   Reference: O. Yuksel, C. Yilmaz, "Efficient topology optimization for maximizing
%   fundamental frequency with design-dependent inertial loads", Engineering
%   Computations 42 (2025) 3194-3214.
%
%   Stage 1: compliance minimization under a unit vertical point load at mid-span
%            (mid-height node); its displacement u is the first estimate of the
%            fundamental mode shape.
%   Stage 2: compliance minimization under the inertial load f = M(x) u_hat, with
%            u_hat the normalized displacement of the previous iteration, so the
%            load follows the design (no eigenanalysis inside the loop).
%   Both stages: SIMP stiffness Emin + x^3 (E0 - Emin), sensitivity filter of radius
%   2.5 elements (symmetric boundary), OC with move limit 0.1, each stage stopped by
%   stagnation_x (the window restarts at Stage 2).  Mass linear above x = 0.1 and
%   proportional to x^6 below it, as in the authors' code.
%
%   This file is a compact port of the production path of the research code
%   (analysis/Yuksel/Matlab/top99neo_inertial_freq.m, tag applsci-revision-1); see
%   tests/test_ports_against_research_code.m.
%
%   opts (all optional): rminElements (2.5), move (0.1), stopTol (1e-2),
%     stopWindow (10), maxIter per stage (5000), stiffness ('simp' | 'pedersen'),
%     xLin (0.1, Pedersen threshold), massCut (0.1; 0 gives a linear mass),
%     Emin, rhoMin (default prob.Emin, prob.rhoMin), filterRadius (physical, overrides
%     rminElements; matched settings of Table 4).
%
%   res: x (common numbering, see fe_mesh), iterations [n1 n2], stop, time
%   (.stage1, .stage2, .total).

if nargin < 2, opts = struct(); end
assert(strcmp(prob.support, 'simplySupported'), 'yuksel_freq: only the simply supported beam is implemented');
nelx = prob.nelx;  nely = prob.nely;  nEl = nelx*nely;
rmin = getopt(opts, 'rminElements', 2.5);
if isfield(opts, 'filterRadius') && ~isempty(opts.filterRadius), rmin = opts.filterRadius/(prob.L/nelx); end
move  = getopt(opts, 'move', 0.1);
tol   = getopt(opts, 'stopTol', 1e-2);
W     = getopt(opts, 'stopWindow', 10);
maxit = getopt(opts, 'maxIter', 5000);
E0 = prob.E;  Emin = getopt(opts, 'Emin', prob.Emin);  nu = prob.nu;
rho0 = prob.rho;  rhoMin = getopt(opts, 'rhoMin', prob.rhoMin);
dMass = 6;  xMassCut = getopt(opts, 'massCut', 0.1);
penal = 3;
x0lin = 0;  if strcmp(getopt(opts, 'stiffness', 'simp'), 'pedersen'), x0lin = getopt(opts, 'xLin', 0.1); end
volfrac = prob.volfrac;
tTotal = tic;

% ---- discretization (top99neo of Ferrari & Sigmund; element e counted from the top) --
nodeNrs = int32(reshape(1:(1+nelx)*(1+nely), 1+nely, 1+nelx));
cVec = reshape(2*nodeNrs(1:end-1, 1:end-1) + 1, nEl, 1);
cMat = cVec + int32([0, 1, 2*nely + [2, 3, 0, 1], -2, -1]);
nDof = (1+nely)*(1+nelx)*2;
[sI, sII] = deal([]);
for j = 1:8
    sI  = cat(2, sI, j:8);
    sII = cat(2, sII, repmat(j, 1, 8-j+1));
end
[iK, jK] = deal(cMat(:, sI)', cMat(:, sII)');
Iar = sort([iK(:), jK(:)], 2, 'descend');  clear iK jK
c1 = [12;3;-6;-3;-6;-3;0;3;12;3;0;-3;-6;-3;-6;12;-3;0;-3;-6;3;12;3;...
    -6;3;-6;12;3;-6;-3;12;3;0;12;-3;12];
c2 = [-4;3;-2;9;2;-3;4;-9;-4;-9;4;-3;2;9;-2;-4;-3;4;9;2;3;-4;-9;-2;...
    3;2;-4;3;-2;9;-4;-9;4;-4;-3;-4];
Ke = 1/(1-nu^2)/24*(c1 + nu.*c2);               % lower triangle of the unit Q4 stiffness
Ke0(tril(ones(8)) == 1) = Ke';
Ke0 = reshape(Ke0, 8, 8);
Ke0 = Ke0 + Ke0' - diag(diag(Ke0));
elemArea = (prob.L/nelx)*(prob.H/nely);
Me0 = kron((elemArea/36)*[4 2 1 2; 2 4 2 1; 1 2 4 2; 2 1 2 4], eye(2));

% pins at mid-height of both ends; Stage-1 load: vertical dof at mid-span, mid-height
midRow = round(nely/2) + 1;
leftMid = nodeNrs(midRow, 1);  rightMid = nodeNrs(midRow, end);
fixed = unique(double([2*leftMid-1, 2*leftMid, 2*rightMid-1, 2*rightMid]));
lcDof = 2*nodeNrs(midRow, round((nelx+1)/2));
F_point = sparse(double(lcDof), 1, -1, nDof, 1);
free = setdiff(1:nDof, fixed);
act = (1:nEl)';

% sensitivity filter with symmetric boundary (imfilter)
[dy, dx] = meshgrid(-ceil(rmin)+1:ceil(rmin)-1, -ceil(rmin)+1:ceil(rmin)-1);
h  = max(0, rmin - sqrt(dx.^2 + dy.^2));
Hs = imfilter(ones(nely, nelx), h, 'symmetric');
dV = ones(nEl,1)/nEl/volfrac;
x  = volfrac*ones(nEl,1);
dsK = zeros(nEl,1);

C = struct('nelx', nelx, 'nely', nely, 'nEl', nEl, 'nDof', nDof, 'cMat', cMat, 'Iar', Iar, ...
    'Ke', Ke, 'Ke0', Ke0, 'Me0', Me0, 'E0', E0, 'Emin', Emin, 'rho0', rho0, 'rhoMin', rhoMin, ...
    'dMass', dMass, 'xMassCut', xMassCut, 'penal', penal, 'x0lin', x0lin, 'h', h, 'Hs', Hs, ...
    'dV', dV, 'move', move, 'free', free, 'fixed', fixed, 'act', act, 'tol', tol, 'W', W, 'maxit', maxit);

% ---- Stage 1: point-load compliance -------------------------------------------------
t1 = tic;
[x, U, n1, stop1] = stageLoop(C, x, zeros(nDof,1), dsK, F_point, false);
t1 = toc(t1);
% ---- Stage 2: inertial-load compliance, load updated from the previous displacement --
t2 = tic;
[x, ~, n2, stop2] = stageLoop(C, x, U, dsK, [], true);
t2 = toc(t2);

res.x = reshape(flipud(reshape(x, nely, nelx)), [], 1);   % to the common numbering
res.iterations = [n1 n2];
res.stop = {stop1, stop2};
res.time = struct('stage1', t1, 'stage2', t2, 'total', toc(tTotal));
end

% =====================================================================================
function [xPhys, U, loop, stop] = stageLoop(C, x, U, dsK, F, inertial)
% Returns the last ANALYSED design xPhys (as the authors' code): Stage 2 starts from
% it and it is the final design of the method.
xPhys = x;  loop = 0;  stop = 'max_iterations';
d = [];  grey = [];  xPrev = [];
while loop < C.maxit
    loop = loop + 1;
    xPhys(C.act) = x(C.act);
    sK = C.Emin + gStiff(xPhys, C.penal, C.x0lin)*(C.E0 - C.Emin);
    dsK(C.act) = dStiff(xPhys(C.act), C.penal, C.E0, C.Emin, C.x0lin);
    sK = reshape(C.Ke(:)*sK', length(C.Ke)*C.nEl, 1);
    K = sparse(double(C.Iar(:,1)), double(C.Iar(:,2)), sK, C.nDof, C.nDof);   % lower triangle
    if inertial
        rhoe = C.rhoMin + (C.rho0 - C.rhoMin)*xPhys;
        low = xPhys <= C.xMassCut;
        rhoe(low) = C.rhoMin + (C.rho0 - C.rhoMin)*(xPhys(low).^C.dMass);
        ltMask = tril(true(8));
        sM = reshape(C.Me0(ltMask)*rhoe', nnz(ltMask)*C.nEl, 1);
        M = sparse(double(C.Iar(:,1)), double(C.Iar(:,2)), sM, C.nDof, C.nDof);
        M = M + M' - spdiags(diag(M), 0, C.nDof, C.nDof);
        uhat = U;
        nrm = norm(uhat(C.free));  if nrm == 0, nrm = 1; end
        F = M*(uhat/nrm);              % inertial load from the normalized previous displacement
        F(C.fixed) = 0;
    end
    U(:) = 0;
    U(C.free) = decomposition(K(C.free, C.free), 'chol', 'lower') \ F(C.free);
    dc = dsK .* sum((U(C.cMat)*C.Ke0) .* U(C.cMat), 2);
    xMat = reshape(max(1e-3, x), C.nely, C.nelx);
    dc  = imfilter(reshape(x.*dc,   C.nely, C.nelx), C.h, 'symmetric') ./ C.Hs ./ xMat;
    dV0 = imfilter(reshape(x.*C.dV, C.nely, C.nelx), C.h, 'symmetric') ./ C.Hs ./ xMat;
    x = ocUpdate(x, C.act, dc(:), dV0(:), C.move, mean(xPhys));
    % stagnation_x on the analysed design xPhys; the window restarts in each stage
    grey(loop) = mean(4*xPhys.*(1-xPhys));                               %#ok<AGROW>
    d(loop) = NaN;                                                       %#ok<AGROW>
    if ~isempty(xPrev), d(loop) = norm(xPhys - xPrev)/max(norm(xPrev), realmin); end
    xPrev = xPhys;
    if stagnation_x(d, grey, loop, C.W, C.tol, C.tol), stop = 'stagnation_x'; break; end
end
end

function x = ocUpdate(x, act, dc, dV0, move, targetMean)
% OC update with a bisection on the volume multiplier (authors' implementation).
xT = x(act);  xU = min(1, xT + move);  xL = max(0, xT - move);
ocArg = -dc(act)./max(dV0(act), 1e-30);
ocArg(~isfinite(ocArg)) = 1e-30;  ocArg = max(ocArg, 1e-30);
ocP = xT.*sqrt(ocArg);
l1 = 0;  l2 = max(mean(ocP)/max(mean(x), eps), 1);
for k = 1:60
    x(act) = max(max(min(min(ocP/l2, xU), 1), xL), 0);
    if mean(x) <= targetMean + 1e-12, break; end
    l2 = 2*l2;
    if ~isfinite(l2), l2 = realmax('double'); break; end
end
for k = 1:120
    if (l2 - l1)/max(l2 + l1, eps) <= 1e-4, break; end
    lmid = 0.5*(l1 + l2);
    if ~isfinite(lmid) || lmid <= 0, break; end
    x(act) = max(max(min(min(ocP/lmid, xU), 1), xL), 0);
    if mean(x) > targetMean, l1 = lmid; else, l2 = lmid; end
end
end

function g = gStiff(x, penal, x0)
g = x.^penal;
if x0 > 0, lo = x < x0;  g(lo) = x(lo)*x0^(penal-1); end
end

function dd = dStiff(x, penal, E0, Emin, x0)
dd = -penal*(E0 - Emin)*x.^(penal-1);
if x0 > 0, dd(x < x0) = -(E0 - Emin)*x0^(penal-1); end
end

function v = getopt(s, f, d)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
