function T = c11_spectral_check(outDir)
%C11_SPECTRAL_CHECK  AppliSci revision C11 (reduced): the spectral form of the
%   quasi-static compliance on the final designs.
%
%   For a design x, reference shape v = phi_1^(0) (solid domain), s = omega0^2,
%   f = s M(x) v, c = f' K^-1 f:
%     lambda_hat = s^2 (v' M v) / c                (weighted harmonic mean)
%     w1         = MAC_M(v, phi_1) of the lowest structural mode
%   and the check  w1*lambda_hat <= lambda_1 <= lambda_hat  (lambda_1 = lowest raw
%   eigenvalue), plus the share of c captured by the 100 lowest modes.
%   Designs: alpha = 1 runs of the alpha sweep (clamped, building; results/<case>/
%   alpha_1p00/result.mat) and the simply supported 400x50 design of C9 (V_f = 0.5).
%   (If result.mat of the sweep is absent, the alpha = 1 design is recomputed with
%   the standalone loop of C4 variant (a), which reproduces it.)
%   All in the optimization model of each example (SS floor 1e-9, others 1e-6).

here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
if nargin < 1 || isempty(outDir), outDir = fullfile(here, 'results', 'c11'); end
if ~exist(outDir, 'dir'), mkdir(outDir); end
diary(fullfile(outDir, 'c11_log.txt')); cleanupDiary = onCleanup(@() diary('off'));
cases = {'ss', fullfile(here, 'results', 'c8c9', 'c8c9_ss_400x50_vf50.mat'); ...
         'clamped', fullfile(here, 'results', 'clamped', 'alpha_1p00', 'result.mat'); ...
         'building', fullfile(here, 'results', 'building', 'alpha_1p00', 'result.mat')};
rows = {};
for ic = 1:size(cases, 1)
    name = cases{ic, 1};
    switch name
        case 'ss', mdl = buildModel(repoRoot, name, 400, 50, 0.5); x = designOf(mdl, '');   % recomputed (= C9 V_f = 0.5 proposed run)
        case 'clamped', mdl = buildModel(repoRoot, name, 400, 50, []); x = designOf(mdl, cases{ic, 2});
        case 'building', mdl = buildModel(repoRoot, name, 80, 240, []); x = designOf(mdl, cases{ic, 2});
    end
    f = mdl.free;
    [K, M] = assemble(mdl, x);
    v = mdl.phi0(:, 1); s = mdl.omega0(1)^2;
    F = s * (M * v); u = zeros(mdl.ndof, 1); u(f) = K(f,f) \ F(f);
    c = F(f)' * u(f);
    mv = v(f)' * M(f,f) * v(f);
    lamHat = s^2 * mv / c;
    nm = 100;
    [V, D] = eigs(K(f,f), M(f,f), nm, 'smallestabs', struct('tol', 1e-10, 'v0', startVec(numel(f))));
    [lam, ix] = sort(real(diag(D))); V = V(:, ix);
    for k = 1:nm, V(:, k) = V(:, k) / sqrt(V(:, k)' * M(f,f) * V(:, k)); end
    a = V' * (M(f,f) * v(f));
    share = s^2 * sum(a.^2 ./ lam) / c;          % part of c from the nm lowest modes
    wRaw1 = a(1)^2 / mv;
    [omS, ~] = structuralOmega(mdl, x);
    [~, js] = min(abs(sqrt(lam) - omS(1)));      % index of the lowest structural mode
    wS = a(js)^2 / mv;
    fprintf(['%-8s omega0 = %.3f  lambda_hat^(1/2) = %.3f  omega1 raw = %.3f (w = %.4f)  omega1 structural = %.3f ' ...
        '(mode %d, w1 = MAC_M = %.4f)  bound %.3f <= %.3f <= %.3f  share of c in %d modes = %.4f  m_v = %.4f\n'], ...
        name, sqrt(s), sqrt(lamHat), sqrt(lam(1)), wRaw1, omS(1), js, wS, sqrt(wRaw1*lamHat), sqrt(lam(1)), ...
        sqrt(lamHat), nm, share, mv);
    rows(end+1, :) = {name, sqrt(s), sqrt(lamHat), sqrt(lam(1)), wRaw1, omS(1), js, wS, share, mv}; %#ok<AGROW>
end
T = cell2table(rows, 'VariableNames', {'case', 'omega0', 'omega_hat', 'omega1_raw', 'w_raw1', ...
    'omega1_structural', 'structural_index', 'MAC_M_structural', 'share_c_100modes', 'modal_mass'});
writetable(T, fullfile(outDir, 'c11_summary.csv'));
disp(T);
end

function x = designOf(mdl, file)
if exist(file, 'file'), S = load(file, 'x'); x = S.x(:); return; end
r = optimize(mdl, [1; 0], false); x = r.x;
fprintf('%s: alpha = 1 design recomputed (%d iterations, %s)\n', mdl.name, r.iter, r.stop);
end

function mdl = buildModel(repoRoot, name, nelxIn, nelyIn, vfIn)
switch name
    case 'ss'
        cfg = jsondecode(fileread(fullfile(repoRoot, 'examples', 'HingedBeam', 'BeamTopOptFreq.json')));
        nelx = 400; nely = 50; EminRatio = 1e-9; rhoMin = 1e-9;
    case 'clamped'
        cfg = jsondecode(fileread(fullfile(repoRoot, 'examples', 'ClampedBeam', 'BeamTopOptFreq.json')));
        nelx = 400; nely = 50; EminRatio = cfg.void_material.E_min_ratio; rhoMin = cfg.void_material.rho_min;
    case 'building'
        cfg = jsondecode(fileread(fullfile(repoRoot, 'examples', 'Building', 'BuildingTopOptFreq.json')));
        nelx = 80; nely = 240; EminRatio = cfg.void_material.E_min_ratio; rhoMin = cfg.void_material.rho_min;
end
nelx = nelxIn; nely = nelyIn;
L = cfg.domain.size.length; H = cfg.domain.size.height;
mdl.name = name; mdl.cfg = cfg;
mdl.nelx = nelx; mdl.nely = nely; mdl.L = L; mdl.H = H;
mdl.nEl = nelx * nely;
hx = L / nelx; hy = H / nely;
mdl.E0 = cfg.material.E; mdl.nu = cfg.material.nu; mdl.rho0 = cfg.material.rho;
mdl.Emin = EminRatio * mdl.E0; mdl.rho_min = rhoMin;
mdl.penal = cfg.optimization.penalization; mdl.pmass = 1;
mdl.volfrac = cfg.optimization.volume_fraction;
if ~isempty(vfIn), mdl.volfrac = vfIn; end
mdl.rmin = 2 * hx;
mdl.KE = lk(hx, hy, mdl.nu);
mdl.ME = lm(hx, hy);
edofMat = zeros(mdl.nEl, 8);
for elx = 0:nelx-1
    for ely = 0:nely-1
        el = ely + elx*nely + 1;
        n1 = (nely+1)*elx + ely; n2 = (nely+1)*(elx+1) + ely;
        n3 = n2 + 1; n4 = n1 + 1;
        edofMat(el, :) = [2*n1+1, 2*n1+2, 2*n2+1, 2*n2+2, 2*n3+1, 2*n3+2, 2*n4+1, 2*n4+2];
    end
end
mdl.edofMat = edofMat;
mdl.iK = reshape(kron(edofMat, ones(1,8))', [], 1);
mdl.jK = reshape(kron(edofMat, ones(8,1))', [], 1);
mdl.ndof = 2 * (nelx+1) * (nely+1);
nodeIds = (1:(nelx+1)*(nely+1))';
nx = floor((nodeIds-1) / (nely+1)) * hx;
ny = mod(nodeIds-1, nely+1) * hy;
switch name
    case 'ss'
        % closest_point supports at (0, H/2) and (L, H/2): one node each, ux and uy
        [~, n1] = min(hypot(nx - 0, ny - H/2)); [~, n2] = min(hypot(nx - L, ny - H/2));
        fixedNodes = [n1; n2];
    case 'clamped', fixedNodes = nodeIds(abs(nx) < 1e-9 | abs(nx - L) < 1e-9);
    case 'building', fixedNodes = nodeIds(abs(ny) < 1e-9);
end
fixed = sort([2*fixedNodes-1; 2*fixedNodes]);
mdl.free = setdiff((1:mdl.ndof)', fixed);
if strcmp(name, 'building')
    [pasS, pasV] = parsePassiveRegions(cfg, nelx, nely, L, H);
else
    pasS = []; pasV = [];   % ss and clamped beams: no passive regions in the paper
end
mdl.pasS = pasS(:); mdl.pasV = pasV(:);
mdl.act = setdiff((1:mdl.nEl)', [mdl.pasS; mdl.pasV]);
rminx = max(1, ceil(mdl.rmin/hx)); rminy = max(1, ceil(mdl.rmin/hy));
iH = zeros(mdl.nEl*(2*rminx-1)*(2*rminy-1), 1); jH = iH; sH = iH; cc = 0;
for i = 0:nelx-1
    for j = 0:nely-1
        row = i*nely + j + 1;
        for k = max(i-(rminx-1), 0):min(i+rminx, nelx)-1
            for l = max(j-(rminy-1), 0):min(j+rminy, nely)-1
                cc = cc + 1;
                iH(cc) = row; jH(cc) = k*nely + l + 1;
                sH(cc) = max(0, mdl.rmin - sqrt(((i-k)*hx)^2 + ((j-l)*hy)^2));
            end
        end
    end
end
mdl.Hf = sparse(iH(1:cc), jH(1:cc), sH(1:cc), mdl.nEl, mdl.nEl);
mdl.Hs = full(sum(mdl.Hf, 2));
xS = ones(mdl.nEl, 1); xS(mdl.pasV) = 0;
[K, M] = assemble(mdl, xS);
f = mdl.free;
[V, D] = eigs(K(f,f), M(f,f), 2, 'smallestabs', struct('tol', 1e-12, 'v0', startVec(numel(f))));
[lam, ix] = sort(real(diag(D))); V = V(:, ix);
mdl.omega0 = sqrt(lam);
mdl.phi0 = zeros(mdl.ndof, 2);
for k = 1:2
    mdl.phi0(f, k) = V(:, k) / sqrt(V(:, k)' * M(f,f) * V(:, k));
end
mdl.b = mdl.phi0 .* (mdl.omega0'.^2);
end

function [K, M] = assemble(mdl, x)
Ee = mdl.Emin + x.^mdl.penal * (mdl.E0 - mdl.Emin);
re = mdl.rho_min + x.^mdl.pmass * (mdl.rho0 - mdl.rho_min);
K = sparse(mdl.iK, mdl.jK, reshape(mdl.KE(:) * Ee', [], 1), mdl.ndof, mdl.ndof);
M = sparse(mdl.iK, mdl.jK, reshape(mdl.ME(:) * re', [], 1), mdl.ndof, mdl.ndof);
K = (K + K') / 2; M = (M + M') / 2;
end

function [c, gK, gF, U] = objectiveAndGradients(mdl, x, w)
[K, M] = assemble(mdl, x);
f = mdl.free;
F = M * mdl.b;
U = zeros(mdl.ndof, 2);
U(f, :) = K(f,f) \ F(f, :);
c = 0; gK = zeros(mdl.nEl, 1); gF = zeros(mdl.nEl, 1);
dE = mdl.penal * x.^(mdl.penal-1) * (mdl.E0 - mdl.Emin);
dR = mdl.pmass * x.^(mdl.pmass-1) * (mdl.rho0 - mdl.rho_min);
for l = 1:2
    if w(l) == 0, continue; end
    c = c + w(l) * (U(f, l)' * F(f, l));
    ue = U(mdl.edofMat, l); ue = reshape(ue, mdl.nEl, 8);
    be = mdl.b(mdl.edofMat, l); be = reshape(be, mdl.nEl, 8);
    gK = gK - w(l) * dE .* sum((ue * mdl.KE) .* ue, 2);
    gF = gF + w(l) * 2 * dR .* sum((ue * mdl.ME) .* be, 2);
end
gK([mdl.pasS; mdl.pasV]) = 0; gF([mdl.pasS; mdl.pasV]) = 0;
end

function g = fdGradient(mdl, x, w, h, smp)
g = zeros(numel(smp), 1);
for i = 1:numel(smp)
    e = smp(i);
    xp = x; xp(e) = xp(e) + h;
    xm = x; xm(e) = xm(e) - h;
    g(i) = (objectiveAndGradients(mdl, xp, w) - objectiveAndGradients(mdl, xm, w)) / (2*h);
end
end

function m = gradientMetrics(mdl, x, gK, gF, gFD, gKall, gFall)
full = gK + gF;
m.errFull = norm(full - gFD) / norm(gFD);
m.errK = norm(gK - gFD) / norm(gFD);
% Size of the omitted term over ALL active elements (analytic).
m.ratio = norm(gFall(mdl.act)) / norm(gKall(mdl.act));
m.cosK = (gK' * gFD) / (norm(gK) * norm(gFD));
m.signFlip = mean(sign(gK) ~= sign(gFD));
m.posFD = mean(gFD > 0);
m.posK = mean(gK > 0);
% After the sensitivity filter (ft = 0, as in topopt_freq): what OC sees.
filt = @(d) (mdl.Hf * (x .* d)) ./ mdl.Hs ./ max(0.001, x);
pas = [mdl.pasS; mdl.pasV];
fK = filt(gKall); fK(pas) = 0;
fFull = filt(gKall + gFall); fFull(pas) = 0;
fK = fK(mdl.act); fFull = fFull(mdl.act);
m.ratioFilt = norm(fFull - fK) / norm(fK);
m.cosKFilt = (fK' * fFull) / (norm(fK) * norm(fFull));
m.signFlipFilt = mean(sign(fK) ~= sign(fFull));
end

function res = optimize(mdl, w, useLoadSens)
% Standalone copy of the Proposed loop of topopt_freq (OC, ft = 0).
x = initialDesign(mdl);
g = 0; maxIt = 2000; W = 10; tol = 1e-2;
d = NaN(maxIt, 1); gr = NaN(maxIt, 1); cH = NaN(maxIt, 1);
xPrev = []; pas = [mdl.pasS; mdl.pasV];
res.stop = 'max_iterations';
for it = 1:maxIt
    [c, gK, gF] = objectiveAndGradients(mdl, x, w);
    cH(it) = c;
    gr(it) = mean(4*x.*(1-x));
    if ~isempty(xPrev), d(it) = norm(x - xPrev) / max(norm(xPrev), realmin); end
    xPrev = x;
    dc = gK; if useLoadSens, dc = gK + gF; end
    dv = ones(mdl.nEl, 1); dv(pas) = 0; dc(pas) = 0;
    dc = (mdl.Hf * (x .* dc)) ./ mdl.Hs ./ max(0.001, x);
    dc(pas) = 0; dv(pas) = 0;
    % Same order as topopt_freq: update, then test the analysed history.
    [x, g] = oc(x, dc, dv, g, 0.2);
    x(mdl.pasS) = 1; x(mdl.pasV) = 0;
    [met, ~] = stop_stagnation_x(d, gr, it, W, tol, tol);
    if met, res.stop = 'stagnation_x_tolerance'; break; end
end
res.x = x; res.iter = it; res.c = objectiveAndGradients(mdl, x, w);
res.hist = struct('c', cH(1:it), 'd', d(1:it), 'gray', gr(1:it));
end

function res = optimizeMMA(mdl, w, useLoadSens)
% MMA variant of the Proposed loop (ft = 0 sensitivity filter, move 0.2,
% stagnation_x W = 10, 1e-2).  Objective scaled by its initial value.
x = initialDesign(mdl);
act = mdl.act; n = numel(act); pas = [mdl.pasS; mdl.pasV];
xold1 = x(act); xold2 = x(act); low = zeros(n, 1); upp = ones(n, 1);
maxIt = 2000; W = 10; tol = 1e-2; move = 0.2;
d = NaN(maxIt, 1); gr = NaN(maxIt, 1); cH = NaN(maxIt, 1);
xPrev = []; c0 = NaN;
res.stop = 'max_iterations';
for it = 1:maxIt
    [c, gK, gF] = objectiveAndGradients(mdl, x, w);
    if isnan(c0), c0 = c; end
    cH(it) = c;
    gr(it) = mean(4*x.*(1-x));
    if ~isempty(xPrev), d(it) = norm(x - xPrev) / max(norm(xPrev), realmin); end
    xPrev = x;
    dc = gK; if useLoadSens, dc = gK + gF; end
    dv = ones(mdl.nEl, 1); dv(pas) = 0; dc(pas) = 0;
    dc = (mdl.Hf * (x .* dc)) ./ mdl.Hs ./ max(0.001, x);
    dc(pas) = 0;
    vScale = mdl.volfrac * mdl.nEl;
    fval = sum(x) / vScale - 1;
    [xa, ~, ~, ~, ~, ~, ~, ~, ~, low, upp] = mmasub(1, n, it, x(act), zeros(n, 1), ones(n, 1), ...
        xold1, xold2, c / c0, dc(act) / c0, fval, dv(act)' / vScale, low, upp, ...
        1, 0, 1e3, 1);
    xa = max(max(0, x(act) - move), min(min(1, x(act) + move), xa));
    xold2 = xold1; xold1 = x(act);
    x(act) = xa; x(mdl.pasS) = 1; x(mdl.pasV) = 0;
    [met, ~] = stop_stagnation_x(d, gr, it, W, tol, tol);
    if met, res.stop = 'stagnation_x_tolerance'; break; end
end
res.x = x; res.iter = it; res.c = objectiveAndGradients(mdl, x, w);
res.hist = struct('c', cH(1:it), 'd', d(1:it), 'gray', gr(1:it));
end

function x = initialDesign(mdl)
x = mdl.volfrac * ones(mdl.nEl, 1);
x(mdl.pasS) = 1; x(mdl.pasV) = 0;
if ~isempty(mdl.pasS) || ~isempty(mdl.pasV)
    x(mdl.act) = min(1, max(0, (mdl.volfrac*mdl.nEl - numel(mdl.pasS)) / numel(mdl.act)));
end
end

function [xnew, gt] = oc(x, dc, dv, g, move)
% Copy of topopt_freq/oc.
l1 = 0; l2 = 1e9;
dv_safe = max(dv, 1e-12);
dc_safe = min(dc, -1e-12);
for iter = 1:200
    lmid = 0.5 * (l1 + l2);
    B = max(-dc_safe ./ dv_safe / max(lmid, 1e-30), 1e-30);
    xnew = max(0, max(x - move, min(1, min(x + move, x .* sqrt(B)))));
    gt = g + sum(dv_safe .* (xnew - x));
    if gt > 0, l1 = lmid; else, l2 = lmid; end
    if (l2 - l1) / max(l1 + l2, 1e-30) < 1e-3, break; end
end
end

function [om, nVoid] = structuralOmega(mdl, x)
[K, M] = assemble(mdl, x);
f = mdl.free;
Ee = mdl.Emin + x.^mdl.penal * (mdl.E0 - mdl.Emin);
re = mdl.rho_min + x.^mdl.pmass * (mdl.rho0 - mdl.rho_min);
[om, ~, info] = lowest_structural_modes(K(f,f), M(f,f), f, mdl.ndof, mdl.edofMat, ...
    mdl.KE, mdl.ME, Ee, re, x, 2, 6);
nVoid = info.nVoid;
end

function saveDesignPng(mdl, xs, titles, file)
fig = figure('Visible', 'off', 'Color', 'w');
n = numel(xs);
for k = 1:n
    subplot(1, n, k);
    imagesc(1 - reshape(xs{k}, mdl.nely, mdl.nelx)); axis image off; colormap(gray); caxis([0 1]);
    set(gca, 'YDir', 'normal');
    title(titles{k}, 'FontSize', 8);
end
exportgraphics(fig, file, 'Resolution', 150);
close(fig);
end

function v = startVec(n)
v = sin((1:n)' * 0.7071067811865476) + 0.5;
end

function KE = lk(hx, hy, nu)
D = (1 / (1 - nu^2)) * [1, nu, 0; nu, 1, 0; 0, 0, 0.5*(1-nu)];
invJ = [2/hx, 0; 0, 2/hy]; detJ = 0.25 * hx * hy;
gp = 1 / sqrt(3);
KE = zeros(8, 8);
for xi = [-gp, gp]
    for eta = [-gp, gp]
        dN = invJ * (0.25 * [-(1-eta), (1-eta), (1+eta), -(1+eta); -(1-xi), -(1+xi), (1+xi), (1-xi)]);
        B = zeros(3, 8);
        B(1, 1:2:end) = dN(1, :); B(2, 2:2:end) = dN(2, :);
        B(3, 1:2:end) = dN(2, :); B(3, 2:2:end) = dN(1, :);
        KE = KE + (B' * D * B) * detJ;
    end
end
KE = (KE + KE') / 2;
end

function ME = lm(hx, hy)
ME = kron((hx*hy/36) * [4 2 1 2; 2 4 2 1; 1 2 4 2; 2 1 2 4], eye(2));
end

