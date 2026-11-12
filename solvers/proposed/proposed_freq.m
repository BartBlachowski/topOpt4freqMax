function res = proposed_freq(prob, opts)
%PROPOSED_FREQ  Proposed quasi-static method (Algorithm 3 and Section 2.2 of the paper).
%
%   res = proposed_freq(prob)        prob from configs/problem_*.m
%   res = proposed_freq(prob, opts)
%
%   Step I : one eigenanalysis of the SOLID design domain gives the reference
%            eigenpairs (omega_j^(0), phi_j^(0)), M_0-normalized and then frozen.
%   Step II: static compliance minimization under the design-dependent inertial loads
%                f_l(x) = (omega_j(l)^(0))^2 M(x) phi_j(l)^(0),   K(x) u_l = f_l(x),
%                c(x)   = sum_l w_l u_l' K(x) u_l                 (Eqs. (5) and (14)),
%            SIMP stiffness, linear mass, sensitivity filter of radius 2 elements,
%            OC update with move limit 0.2, volume sum(x)/N = V_f, stagnation_x stop.
%            The sensitivity is the frozen-load term g_e = -w_l E'(x_e) u_e' k_e u_e;
%            the load term of Eq. (13) is omitted on purpose (Section 3.2).
%
%   opts (all optional)
%     alpha        weight of mode 1; the objective is alpha c_1 + (1-alpha) c_2
%                  (default 1: single load case of the first mode)
%     reference    struct with fields omega0 (1 or 2 values) and phi0 (ndof x 1|2)
%                  replacing the solid-domain eigenpairs (reference-mode study)
%     sensitivity  'frozen' (the method, default) | 'full' (adds the load term,
%                  Table 1 ablation)
%     optimizer    'OC' (default) | 'MMA' (Table 1 ablation)
%     volume       'equality' (default) | 'inequality'  (MMA only)
%     move, rminElements, stopTol, stopWindow, maxIter (0.2, 2, 1e-2, 10, 2000)
%     filterRadius physical filter radius (overrides rminElements)
%     stiffness    'simp' (default) | 'pedersen' (linear below xLin = 0.1)
%     Emin, rhoMin material floors (default prob.Emin, prob.rhoMin)
%     (the last three only for the matched settings of Table 4)
%     keepDesigns  store every analysed design in res.designs (default false)
%     verbose
%
%   res: x (final design), c, iterations, stop, omega0, phi0, mdl, time
%   (.reference = Step I, .loop = Step II, .total), hist (c, d, grey per iteration).
%   The frequencies of the final design are evaluated separately with
%   structural_modes (common evaluator), as everywhere in the paper.

if nargin < 2, opts = struct(); end
alpha   = getopt(opts, 'alpha', 1);
sens    = getopt(opts, 'sensitivity', 'frozen');
optim   = upper(getopt(opts, 'optimizer', 'OC'));
volMode = getopt(opts, 'volume', 'equality');
move    = getopt(opts, 'move', 0.2);
rElem   = getopt(opts, 'rminElements', 2);
tol     = getopt(opts, 'stopTol', 1e-2);
W       = getopt(opts, 'stopWindow', 10);
maxIt   = getopt(opts, 'maxIter', 2000);
keepX   = getopt(opts, 'keepDesigns', false);
verbose = getopt(opts, 'verbose', false);
tTotal = tic;

mdl = fe_mesh(prob);
E0 = prob.E;  Emin = getopt(opts, 'Emin', prob.Emin);  rho0 = prob.rho;  rhoMin = getopt(opts, 'rhoMin', prob.rhoMin);  p = 3;
xLin = 0;  if strcmp(getopt(opts, 'stiffness', 'simp'), 'pedersen'), xLin = 0.1; end
w = [alpha; 1 - alpha];

% ---- Step I: reference eigenpairs of the solid design domain ----------------
tRef = tic;
f = mdl.free;
if isfield(opts, 'reference') && ~isempty(opts.reference)
    omega0 = opts.reference.omega0(:);  phi0 = opts.reference.phi0;
    if numel(omega0) == 1, omega0 = [omega0; 0]; phi0 = [phi0, zeros(mdl.ndof,1)]; end
else
    [K, M] = assemble_KM(mdl, ones(mdl.nEl,1), E0, Emin, rho0, rhoMin, p, xLin);
    v0 = sin((1:numel(f))'*0.7071067811865476) + 0.5;
    [V, D] = eigs(K(f,f), M(f,f), 2, 'smallestabs', struct('tol', 1e-12, 'v0', v0));
    [lam, ix] = sort(real(diag(D)));  V = V(:, ix);
    omega0 = sqrt(lam);  phi0 = zeros(mdl.ndof, 2);
    for k = 1:2, phi0(f,k) = V(:,k)/sqrt(V(:,k)'*M(f,f)*V(:,k)); end
end
b = phi0 .* (omega0'.^2);                 % f_l = M(x) b_l
tRef = toc(tRef);

% ---- sensitivity filter (Sigmund 1997), radius rElem element widths -----------
rPhys = getopt(opts, 'filterRadius', rElem*mdl.hx);
Hf = filterMatrix(mdl, rPhys);  Hs = full(sum(Hf, 2));

% ---- Step II -----------------------------------------------------------------
tLoop = tic;
pas = mdl.pas;  act = mdl.act;
x = prob.volfrac*ones(mdl.nEl,1);
if ~isempty(pas)                          % passive frame solid, rest uniform at V_f
    x(pas) = 1;  x(act) = min(1, max(0, (prob.volfrac*mdl.nEl - numel(pas))/numel(act)));
end
d = NaN(maxIt,1);  grey = NaN(maxIt,1);  cH = NaN(maxIt,1);  designs = {};
xPrev = [];  lOC = 0;  stop = 'max_iterations';
if strcmp(optim, 'MMA')
    nA = numel(act);  xold1 = x(act);  xold2 = x(act);  low = zeros(nA,1);  upp = ones(nA,1);  c0 = NaN;
end
for it = 1:maxIt
    [c, gK, gF] = objectiveAndGradients(mdl, x, w, b, E0, Emin, rho0, rhoMin, p, xLin);
    if keepX, designs{end+1} = x; end                                    %#ok<AGROW>
    cH(it) = c;  grey(it) = mean(4*x.*(1-x));
    if ~isempty(xPrev), d(it) = norm(x - xPrev)/max(norm(xPrev), realmin); end
    xPrev = x;
    dc = gK;  if strcmp(sens, 'full'), dc = gK + gF; end
    dv = ones(mdl.nEl,1);  dv(pas) = 0;  dc(pas) = 0;
    dc = (Hf*(x.*dc))./Hs./max(0.001, x);                               % filtered sensitivity
    dc(pas) = 0;
    if strcmp(optim, 'OC')
        [x, lOC] = oc(x, dc, dv, lOC, move);
    else                                                                % MMA (Table 1 only)
        if isnan(c0), c0 = c; end
        vS = prob.volfrac*mdl.nEl;  fval = sum(x)/vS - 1;  dfv = dv(act)'/vS;
        [xa, ~, ~, ~, ~, ~, ~, ~, ~, low, upp] = mmasub(1, nA, it, x(act), zeros(nA,1), ones(nA,1), ...
            xold1, xold2, c/c0, dc(act)/c0, fval, dfv, low, upp, 1, 0, 1e3, 1);
        if strcmp(volMode, 'equality')    % exact projection onto the volume (uniform shift)
            lo = max(0, x(act) - move);  hi = min(1, x(act) + move);
            Vt = prob.volfrac*mdl.nEl - sum(x) + sum(x(act));
            l1 = -1;  l2 = 1;
            for kb = 1:100
                lm = 0.5*(l1 + l2);
                if sum(min(hi, max(lo, xa + lm))) > Vt, l2 = lm; else, l1 = lm; end
            end
            xa = min(hi, max(lo, xa + 0.5*(l1 + l2)));
        end
        xa = max(max(0, x(act) - move), min(min(1, x(act) + move), xa));
        xold2 = xold1;  xold1 = x(act);  x(act) = xa;
    end
    x(pas) = 1;
    if verbose, fprintf('%4d  c = %.6e  vol = %.4f  change = %.3e\n', it, c, mean(x), d(it)); end
    if stagnation_x(d, grey, it, W, tol, tol), stop = 'stagnation_x'; break; end
end
res.x = x;  res.c = objectiveAndGradients(mdl, x, w, b, E0, Emin, rho0, rhoMin, p, xLin);
res.iterations = it;  res.stop = stop;
res.omega0 = omega0;  res.phi0 = phi0;  res.mdl = mdl;
res.hist = struct('c', cH(1:it), 'd', d(1:it), 'grey', grey(1:it));
if keepX, designs{end+1} = x; res.designs = designs; end
res.time = struct('reference', tRef, 'loop', toc(tLoop), 'total', toc(tTotal));
end

% ============================================================================
function [c, gK, gF] = objectiveAndGradients(mdl, x, w, b, E0, Emin, rho0, rhoMin, p, xLin)
% c = sum_l w_l f_l'u_l; gK = frozen-load sensitivity g_e; gF = omitted load term.
[K, M] = assemble_KM(mdl, x, E0, Emin, rho0, rhoMin, p, xLin);
f = mdl.free;
F = M*b;
U = zeros(mdl.ndof, 2);
U(f,:) = K(f,f) \ F(f,:);
c = 0;  gK = zeros(mdl.nEl,1);  gF = zeros(mdl.nEl,1);
if xLin > 0
    dE = ((x >= xLin).*(p*x.^(p-1)) + (x < xLin)*xLin^(p-1))*(E0 - Emin);
else
    dE = p*x.^(p-1)*(E0 - Emin);
end
dR = ones(size(x))*(rho0 - rhoMin);
for l = 1:2
    if w(l) == 0, continue; end
    c = c + w(l)*(U(f,l)'*F(f,l));
    ue = reshape(U(mdl.edofMat, l), mdl.nEl, 8);
    be = reshape(b(mdl.edofMat, l), mdl.nEl, 8);
    gK = gK - w(l)*dE.*sum((ue*mdl.KE).*ue, 2);
    gF = gF + w(l)*2*dR.*sum((ue*mdl.ME).*be, 2);
end
gK(mdl.pas) = 0;  gF(mdl.pas) = 0;
end

function [xnew, gt] = oc(x, dc, dv, g, move)
% Optimality-criteria update with bisection on the volume Lagrange multiplier.
l1 = 0;  l2 = 1e9;
dv_safe = max(dv, 1e-12);  dc_safe = min(dc, -1e-12);
for iter = 1:200
    lmid = 0.5*(l1 + l2);
    B = max(-dc_safe./dv_safe/max(lmid, 1e-30), 1e-30);
    xnew = max(0, max(x - move, min(1, min(x + move, x.*sqrt(B)))));
    gt = g + sum(dv_safe.*(xnew - x));
    if gt > 0, l1 = lmid; else, l2 = lmid; end
    if (l2 - l1)/max(l1 + l2, 1e-30) < 1e-3, break; end
end
end

function Hf = filterMatrix(mdl, rmin)
nelx = mdl.nelx;  nely = mdl.nely;  hx = mdl.hx;  hy = mdl.hy;
rx = max(1, ceil(rmin/hx));  ry = max(1, ceil(rmin/hy));
iH = zeros(mdl.nEl*(2*rx-1)*(2*ry-1), 1);  jH = iH;  sH = iH;  cc = 0;
for i = 0:nelx-1
    for j = 0:nely-1
        row = i*nely + j + 1;
        for k = max(i-(rx-1), 0):min(i+rx, nelx)-1
            for l = max(j-(ry-1), 0):min(j+ry, nely)-1
                cc = cc + 1;  iH(cc) = row;  jH(cc) = k*nely + l + 1;
                sH(cc) = max(0, rmin - sqrt(((i-k)*hx)^2 + ((j-l)*hy)^2));
            end
        end
    end
end
Hf = sparse(iH(1:cc), jH(1:cc), sH(1:cc), mdl.nEl, mdl.nEl);
end

function v = getopt(s, f, d)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
