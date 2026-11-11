function R = c1_fd_gradient_check(outDir, parts)
%C1_FD_GRADIENT_CHECK  AppliSci revision C1: weight of the omitted dF/dx term.
%
%   R = c1_fd_gradient_check()          writes results under results/c1/
%   R = c1_fd_gradient_check(outDir)
%
%   The Proposed objective (revision formulation, load_case_weighting =
%   "compliance") is
%
%       c(x) = sum_l w_l f_l(x)' u_l(x),   K(x) u_l = f_l(x),
%       f_l(x) = omega0_l^2 M(x) phi0_l,   w = [alpha, 1-alpha],
%
%   with (omega0_l, phi0_l) the FROZEN solid-domain eigenpairs.  Its exact
%   derivative is (K, M symmetric)
%
%       dc/dx_e = sum_l w_l [ -u_l' dK/dx_e u_l  +  2 u_l' dM/dx_e (omega0_l^2 phi0_l) ]
%               =          g_K                   +  g_F
%
%   topopt_freq uses g_K only (Eq. 6 of the paper; semi_harmonic load
%   sensitivity omitted).  This script
%
%   Part A  checks g_K + g_F against central finite differences of c on 400
%           randomly chosen design variables (so the full derivative above is the derivative
%           of the objective the code evaluates) and reports how large g_F is
%           relative to g_K, raw and after the sensitivity filter, on three
%           designs: uniform x = volfrac, a random design, and the design
%           topopt_freq converges to on that mesh.
%   Part B  reruns the optimization with a standalone copy of the Proposed
%           loop (OC, move 0.2, sensitivity filter, stagnation_x W=10, 1e-2)
%           once with g_K (as the paper) and once with g_K + g_F, and compares
%           compliance, structural omega1/omega2 and iteration counts.  The
%           g_K run is cross-checked against topopt_freq itself.
%
%   Part C  as Part B but with MMA (move 0.2, objective scaled by its initial
%           value, volume constraint sum(x)/(volfrac*n) - 1 <= 0), because OC
%           clips positive sensitivities (dc_safe = min(dc, -1e-12)) and so
%           cannot follow g_K + g_F, which is positive in many elements.
%           Both variants (g_K and g_K + g_F) run with the same MMA.
%
%   parts: any subset of 'ABC' (default 'ABC').  Part C alone writes only
%   c1_partC_mma.csv.
%
%   Small meshes: clamped beam 160x20 (8 x 1 m), building 40x120 (5 x 15 m).
%   alpha in {1, 0.5, 0}.  Material and void parameters as in the paper's
%   JSON files (E0 = 1e7, nu = 0.3, rho0 = 1, E_min = 1e-6 E0, rho_min = 1e-6,
%   p = 3, linear mass).

here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
if nargin < 1 || isempty(outDir)
    outDir = fullfile(here, 'results', 'c1');
end
if nargin < 2 || isempty(parts), parts = 'ABC'; end
doAB = any(parts == 'A') || any(parts == 'B');
doC = any(parts == 'C');
if ~exist(outDir, 'dir'), mkdir(outDir); end
logFile = fullfile(outDir, sprintf('c1_log_%s.txt', parts));
if exist(logFile, 'file'), delete(logFile); end
diary(logFile); cleanupDiary = onCleanup(@() diary('off'));

cases = {'clamped', 'building'};
alphas = [1.0 0.5 0.0];
nFD = 400;   % elements checked by finite differences per design
rowsA = {};
rowsB = {};
rowsC = {};
R = struct();
for ic = 1:numel(cases)
    name = cases{ic};
    mdl = buildModel(repoRoot, name);
    fprintf('\n===== %s %dx%d: omega0 = %.4f, %.4f rad/s (solid domain) =====\n', ...
        name, mdl.nelx, mdl.nely, mdl.omega0(1), mdl.omega0(2));
    for ia = 1:numel(alphas)
        a = alphas(ia);
        w = [a; 1 - a];

        if doC
            % --- Part C: MMA with and without g_F -------------------------
            mK = optimizeMMA(mdl, w, false);
            mF = optimizeMMA(mdl, w, true);
            [omK, ~] = structuralOmega(mdl, mK.x);
            [omF, ~] = structuralOmega(mdl, mF.x);
            fprintf(['[C] %-8s alpha=%.2f  MMA g_K: it=%d c=%.6e om=%.3f/%.3f vol=%.4f (%s) | ' ...
                     'MMA g_K+g_F: it=%d c=%.6e om=%.3f/%.3f vol=%.4f (%s) | mean|dx|=%.3f\n'], ...
                name, a, mK.iter, mK.c, omK(1), omK(2), mean(mK.x), mK.stop, ...
                mF.iter, mF.c, omF(1), omF(2), mean(mF.x), mF.stop, mean(abs(mK.x - mF.x)));
            rowsC(end+1, :) = {name, a, mK.iter, mK.c, omK(1), omK(2), mean(mK.x), mK.stop, ...
                mF.iter, mF.c, omF(1), omF(2), mean(mF.x), mF.stop, mean(abs(mK.x - mF.x))}; %#ok<AGROW>
            tag = sprintf('%s_alpha_%s', name, strrep(sprintf('%.2f', a), '.', 'p'));
            R.([tag '_mma']) = struct('xK', mK.x, 'xF', mF.x, 'histK', mK.hist, 'histF', mF.hist);
            saveDesignPng(mdl, {mK.x, mF.x}, {'MMA, g_K', 'MMA, g_K+g_F'}, ...
                fullfile(outDir, [tag '_mma.png']));
        end
        if ~doAB, continue; end

        % --- reference run of topopt_freq on the same mesh --------------
        [xRef, omRef, itRef, objRef] = runReference(repoRoot, mdl, a, outDir);
        [omRefS, ~] = structuralOmega(mdl, xRef);

        % --- Part A: FD check -------------------------------------------
        rng(1);
        xRand = min(1, max(0.05, mdl.volfrac + 0.35 * (2*rand(mdl.nEl, 1) - 1)));
        xRand(mdl.pasS) = 1; xRand(mdl.pasV) = 0;
        xUni = initialDesign(mdl);
        designs = {'uniform', xUni; 'random', xRand; 'optimized', xRef};
        for id = 1:size(designs, 1)
            x = designs{id, 2};
            [c, gK, gF] = objectiveAndGradients(mdl, x, w);
            % FD on a fixed random sample of active elements (cost: 2 solves each).
            rng(2); smp = mdl.act(randperm(numel(mdl.act), min(nFD, numel(mdl.act))));
            gFD = fdGradient(mdl, x, w, 1e-5, smp);
            m = gradientMetrics(mdl, x, gK(smp), gF(smp), gFD, gK, gF);
            fprintf(['[A] %-8s alpha=%.2f %-9s c=%.6e  |full-FD|/|FD|=%.2e  |gK-FD|/|FD|=%.3f  ' ...
                     '|gF|/|gK|=%.3f  cos(gK,FD)=%.4f  signflip=%.1f%%  gFD>0: %.1f%%  ' ...
                     'filtered: |gF|/|gK|=%.3f cos=%.4f\n'], ...
                name, a, designs{id, 1}, c, m.errFull, m.errK, m.ratio, m.cosK, ...
                100*m.signFlip, 100*m.posFD, m.ratioFilt, m.cosKFilt);
            rowsA(end+1, :) = {name, a, designs{id, 1}, c, m.errFull, m.errK, m.ratio, ...
                m.cosK, m.signFlip, m.posFD, m.posK, m.ratioFilt, m.cosKFilt, m.signFlipFilt}; %#ok<AGROW>
        end

        % --- Part B: optimization with and without g_F ------------------
        resK = optimize(mdl, w, false);
        resF = optimize(mdl, w, true);
        [omK, nVoidK] = structuralOmega(mdl, resK.x);
        [omF, nVoidF] = structuralOmega(mdl, resF.x);
        cKatF = objectiveAndGradients(mdl, resF.x, w);
        fprintf(['[B] %-8s alpha=%.2f  topopt_freq: it=%d c=%.6e om=%.3f/%.3f | ' ...
                 'g_K copy: it=%d c=%.6e om=%.3f/%.3f (|x-xRef|_inf=%.1e) | ' ...
                 'g_K+g_F: it=%d c=%.6e om=%.3f/%.3f, design diff |x_K-x_F|_1/n=%.3f\n'], ...
            name, a, itRef, objRef, omRefS(1), omRefS(2), ...
            resK.iter, resK.c, omK(1), omK(2), max(abs(resK.x - xRef)), ...
            resF.iter, cKatF, omF(1), omF(2), mean(abs(resK.x - resF.x)));
        rowsB(end+1, :) = {name, a, itRef, objRef, omRefS(1), omRefS(2), ...
            resK.iter, resK.c, omK(1), omK(2), max(abs(resK.x - xRef)), resK.stop, ...
            resF.iter, cKatF, omF(1), omF(2), resF.stop, mean(abs(resK.x - resF.x)), ...
            nVoidK, nVoidF}; %#ok<AGROW>
        tag = sprintf('%s_alpha_%s', name, strrep(sprintf('%.2f', a), '.', 'p'));
        R.(tag) = struct('xRef', xRef, 'xK', resK.x, 'xF', resF.x, 'omRef', omRef, ...
            'histK', resK.hist, 'histF', resF.hist);
        saveDesignPng(mdl, {xRef, resK.x, resF.x}, ...
            {'topopt\_freq', 'copy, g_K', 'copy, g_K+g_F'}, fullfile(outDir, [tag '.png']));
    end
end

if doC
    TC = cell2table(rowsC, 'VariableNames', {'case', 'alpha', 'gK_iter', 'gK_c', ...
        'gK_omega1', 'gK_omega2', 'gK_vol', 'gK_stop', 'full_iter', 'full_c', ...
        'full_omega1', 'full_omega2', 'full_vol', 'full_stop', 'mean_abs_dx'});
    writetable(TC, fullfile(outDir, 'c1_partC_mma.csv'));
    save(fullfile(outDir, 'c1_partC_results.mat'), 'R', 'TC');
    disp(TC);
end
if ~doAB, return; end
TA = cell2table(rowsA, 'VariableNames', {'case', 'alpha', 'design', 'c', ...
    'relerr_full_vs_FD', 'relerr_gK_vs_FD', 'norm_gF_over_gK', 'cos_gK_FD', ...
    'signflip_frac', 'frac_FD_positive', 'frac_gK_positive', ...
    'filtered_norm_gF_over_gK', 'filtered_cos_gK_full', 'filtered_signflip_frac'});
TB = cell2table(rowsB, 'VariableNames', {'case', 'alpha', 'ref_iter', 'ref_c', ...
    'ref_omega1', 'ref_omega2', 'gK_iter', 'gK_c', 'gK_omega1', 'gK_omega2', ...
    'gK_vs_ref_maxdx', 'gK_stop', 'full_iter', 'full_c', 'full_omega1', 'full_omega2', ...
    'full_stop', 'mean_abs_dx_K_vs_full', 'gK_nvoid_below', 'full_nvoid_below'});
writetable(TA, fullfile(outDir, 'c1_partA_fd.csv'));
writetable(TB, fullfile(outDir, 'c1_partB_optimization.csv'));
save(fullfile(outDir, 'c1_results.mat'), 'R', 'TA', 'TB');
disp(TA); disp(TB);
end

% ======================================================================
function mdl = buildModel(repoRoot, name)
switch name
    case 'clamped'
        cfg = jsondecode(fileread(fullfile(repoRoot, 'examples', 'ClampedBeam', 'BeamTopOptFreq.json')));
        nelx = 160; nely = 20;
    case 'building'
        cfg = jsondecode(fileread(fullfile(repoRoot, 'examples', 'Building', 'BuildingTopOptFreq.json')));
        nelx = 40; nely = 120;
end
L = cfg.domain.size.length; H = cfg.domain.size.height;
mdl.name = name; mdl.cfg = cfg;
mdl.nelx = nelx; mdl.nely = nely; mdl.L = L; mdl.H = H;
mdl.nEl = nelx * nely;
hx = L / nelx; hy = H / nely;
mdl.E0 = cfg.material.E; mdl.nu = cfg.material.nu; mdl.rho0 = cfg.material.rho;
mdl.Emin = cfg.void_material.E_min_ratio * mdl.E0; mdl.rho_min = cfg.void_material.rho_min;
mdl.penal = cfg.optimization.penalization; mdl.pmass = 1;
mdl.volfrac = cfg.optimization.volume_fraction;
mdl.rmin = cfg.optimization.filter.radius * hx;

% Element matrices, identical to topopt_freq lk/lm (unit thickness).
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

% Supports: clamped beam = both vertical edges fixed; building = bottom edge.
nodeIds = (1:(nelx+1)*(nely+1))';
nx = floor((nodeIds-1) / (nely+1)) * hx;
ny = mod(nodeIds-1, nely+1) * hy;
switch name
    case 'clamped', fixedNodes = nodeIds(abs(nx) < 1e-9 | abs(nx - L) < 1e-9);
    case 'building', fixedNodes = nodeIds(abs(ny) < 1e-9);
end
fixed = sort([2*fixedNodes-1; 2*fixedNodes]);
mdl.free = setdiff((1:mdl.ndof)', fixed);

[pasS, pasV] = parsePassiveRegions(cfg, nelx, nely, L, H);
mdl.pasS = pasS(:); mdl.pasV = pasV(:);
mdl.act = setdiff((1:mdl.nEl)', [mdl.pasS; mdl.pasV]);

% Sensitivity filter, identical to topopt_freq.
rminx = max(1, ceil(mdl.rmin/hx)); rminy = max(1, ceil(mdl.rmin/hy));
iH = []; jH = []; sH = [];
for i = 0:nelx-1
    for j = 0:nely-1
        row = i*nely + j + 1;
        for k = max(i-(rminx-1), 0):min(i+rminx, nelx)-1
            for l = max(j-(rminy-1), 0):min(j+rminy, nely)-1
                fac = mdl.rmin - sqrt(((i-k)*hx)^2 + ((j-l)*hy)^2);
                iH(end+1) = row; jH(end+1) = k*nely + l + 1; sH(end+1) = max(0, fac); %#ok<AGROW>
            end
        end
    end
end
mdl.Hf = sparse(iH, jH, sH, mdl.nEl, mdl.nEl);
mdl.Hs = full(sum(mdl.Hf, 2));

% Frozen solid-domain eigenpairs (x = 1 except passive voids), mass-normalized.
xS = ones(mdl.nEl, 1); xS(mdl.pasV) = 0;
[K, M] = assemble(mdl, xS);
f = mdl.free;
[V, D] = eigs(K(f,f), M(f,f), 2, 'smallestabs', struct('tol', 1e-12, 'v0', startVec(numel(f))));
[lam, ix] = sort(real(diag(D))); V = V(:, ix);
mdl.omega0 = sqrt(lam);
mdl.phi0 = zeros(mdl.ndof, 2);
for k = 1:2
    v = V(:, k) / sqrt(V(:, k)' * M(f,f) * V(:, k));
    mdl.phi0(f, k) = v;
end
mdl.b = mdl.phi0 .* (mdl.omega0'.^2);     % omega0^2 phi0, ndof x 2
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

function [x, omega, nIter, obj] = runReference(repoRoot, mdl, a, outDir)
% topopt_freq through the JSON runner, revision settings, on the small mesh.
cfg = mdl.cfg;
cfg.domain.mesh.nelx = mdl.nelx; cfg.domain.mesh.nely = mdl.nely;
lc = cfg.domain.load_cases;
lc(1).factor = a; lc(2).factor = 1 - a;
cfg.domain.load_cases = lc;
o = cfg.optimization;
o.approach = 'ourApproach'; o.optimizer = 'OC'; o.objective = 'compliance';
o.load_case_weighting = 'compliance'; o.move_limit = 0.2; o.max_iters = 2000;
o.convergence_tol = 0.01; o.stop_criterion = 'stagnation_x';
o.stagnation = struct('window', 10, 'grayness_tol', 1e-2, 'relative_change_tol', 1e-2);
o.semi_harmonic_baseline = 'solid'; o.harmonic_normalize = false;
for fld = {'debug_semi_harmonic', 'semi_harmonic_rho_source'}
    if isfield(o, fld{1}), o = rmfield(o, fld{1}); end
end
cfg.optimization = o;
p = cfg.postprocessing;
p.compute_modes = 0; p.structural_mode_check = true; p.visualize_live = false;
p.visualize_modes = struct('enabled', false, 'count', 3);
p.visualize_topology_modes = struct('enabled', false, 'count', 3);
p.write_correlation_table = false;
p.correlation = struct('enabled', false, 'metric', 'mac', 'write_csv', false, 'best_k', 3);
p.save_snapshot_image = false; p.save_final_image = false; p.save_frequency_iterations = false;
cfg.postprocessing = p;
runDir = fullfile(outDir, 'ref_runs', sprintf('%s_alpha_%s', mdl.name, strrep(sprintf('%.2f', a), '.', 'p')));
if ~exist(runDir, 'dir'), mkdir(runDir); end
jsonPath = fullfile(runDir, 'config.json');
fid = fopen(jsonPath, 'w'); fprintf(fid, '%s', jsonencode(cfg, 'PrettyPrint', true)); fclose(fid);
prev = pwd; cd(runDir); back = onCleanup(@() cd(prev));
[x, omega, ~, nIter, ~, ~, tel] = run_topopt_from_json(jsonPath);
x = x(:);
obj = tel.objective_final;
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
