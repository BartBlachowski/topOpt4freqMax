function out = c2b_benchmark_modes(outDir)
%C2B_BENCHMARK_MODES  AppliSci revision, audit 2.5: validation of the
%   structural-mode selection on the designs of the method comparison
%   (Table 7: simply supported beam 400x50 at V_f = 0.3/0.5/0.7, proposed and
%   Du-Olhoff, Yuksel-Yilmaz at V_f = 0.5; clamped beam and building, Du-Olhoff),
%   and the third threshold (kinetic-energy-weighted density 0.4/0.5/0.6) on
%   these designs and on the ten designs of Tables 3-6.
%
%   For every design, under the paper's evaluator (SIMP p = 3, E_min = 1e-6 E0,
%   linear mass rho_min = 1e-6): structural omega1/omega2 for all 36
%   combinations of the low-density threshold t (0.05/0.1/0.2/0.3), the
%   energy-share cut s (0.3/0.5/0.7) and the participation threshold p
%   (0.4/0.5/0.6); the rejected modes below omega2 with their energy shares;
%   and omega1/omega2 under E_min = 1e-9 E0, the Pedersen-type mass correction
%   (rho = 1e5 x^6 below x = 0.1) and removal of elements with x < 0.1.
%   Results: results/c2b/c2b_summary.csv, c2b_rejected.csv, c2b_log.txt.
here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
if nargin < 1 || isempty(outDir), outDir = fullfile(here, 'results', 'c2b'); end
if ~exist(outDir, 'dir'), mkdir(outDir); end
logFile = fullfile(outDir, 'c2b_log.txt'); if exist(logFile, 'file'), delete(logFile); end
diary(logFile); cleanupDiary = onCleanup(@() diary('off'));
R = fullfile(here, 'results');
D = {};   % {label, case, nelx, nely, vf, x}
for vf = [30 50 70]
    S = load(fullfile(R, 'c8c9', sprintf('c8c9_ss_400x50_vf%d.mat', vf)), 'xP', 'xO');
    D(end+1, :) = {sprintf('SS V_f=0.%d proposed', vf/10), 'ss', 400, 50, vf/100, S.xP}; %#ok<AGROW>
    D(end+1, :) = {sprintf('SS V_f=0.%d Du-Olhoff', vf/10), 'ss', 400, 50, vf/100, S.xO}; %#ok<AGROW>
end
f = fullfile(R, 'c7tol', 'c7tol_400x50.mat');
if exist(f, 'file')
    T = load(f, 'recs');
    for k = 1:numel(T.recs)
        r = T.recs{k};
        if strcmp(r.method, 'yuksel') && abs(r.tol - 1e-2) < 1e-12
            D(end+1, :) = {'SS V_f=0.5 Yuksel-Yilmaz', 'ss', 400, 50, 0.5, r.x(:)}; %#ok<AGROW>
        end
    end
end
S = load(fullfile(R, 'c8c9', 'c8c9_clamped_400x50_vf50.mat'), 'xO');
D(end+1, :) = {'clamped Du-Olhoff', 'clamped', 400, 50, 0.5, S.xO};
S = load(fullfile(R, 'c8c9', 'c8c9_building_80x240_vf30.mat'), 'xO');
D(end+1, :) = {'building Du-Olhoff', 'building', 80, 240, [], S.xO};
tags = {'1p00', '0p75', '0p50', '0p25', '0p00'}; al = [1 0.75 0.5 0.25 0];
for c = {'clamped', 'building'}
    for k = 1:5
        f = fullfile(R, c{1}, ['alpha_' tags{k}], 'result.mat');
        if ~exist(f, 'file'), continue; end
        S = load(f, 'x');
        if strcmp(c{1}, 'clamped'), m = [400 50]; v = 0.5; else, m = [80 240]; v = []; end
        D(end+1, :) = {sprintf('%s alpha=%.2f proposed', c{1}, al(k)), c{1}, m(1), m(2), v, S.x(:)}; %#ok<AGROW>
    end
end
rows = {}; rej = {};
mdlCache = struct();
for id = 1:size(D, 1)
    [lbl, name, nelx, nely, vf, x] = D{id, :};
    key = sprintf('%s_%d', name, nelx);
    if ~isfield(mdlCache, key), mdlCache.(key) = buildModel(repoRoot, name, nelx, nely, vf); end
    mdl = mdlCache.(key); mdl.rho_min = 1e-6;
    x = max(0, min(1, x(:)));
    P = evaluator(mdl, x, 'emin', 1e-6);
    [lam, V, met] = modesWithMetrics(mdl, P, x, 0.1, 40, 2);
    keep = classify(met, 0.5, x);
    sel = find(keep); top = sel(2);
    om0 = sqrt(lam(sel(1:2)));
    for j = find(~keep(1:top))'
        Phi = zeros(mdl.ndof, 1); Phi(mdl.free) = V(:, j);
        b = keBands(mdl, P, x, Phi);
        rej(end+1, :) = {lbl, j, sqrt(lam(j)), met(j, 2), met(j, 3), met(j, 4), b(1), b(2)}; %#ok<AGROW>
    end
    % threshold grid
    oms = [];
    for t = [0.05 0.1 0.2 0.3]
        [lamT, ~, metT] = modesWithMetrics(mdl, P, x, t, 40, 2);
        for s = [0.3 0.5 0.7]
            for p = [0.4 0.5 0.6]
                kT = classify(metT, s, x, p); sT = find(kT);
                o = [NaN NaN]; n2 = min(2, numel(sT)); o(1:n2) = sqrt(lamT(sT(1:n2)));
                oms(end+1, :) = o; %#ok<AGROW>
            end
        end
    end
    % alternative evaluators
    alt = NaN(3, 2);
    Q = evaluator(mdl, x, 'emin', 1e-9); [lq, ~, mq] = modesWithMetrics(mdl, Q, x, 0.1, 40, 2);
    s9 = find(classify(mq, 0.5, x)); alt(1, :) = firstTwo(lq, s9);
    Q = evaluator(mdl, x, 'pedersen', 1e-6); [lq, ~, mq] = modesWithMetrics(mdl, Q, x, 0.1, 40, 2);
    sp = find(classify(mq, 0.5, x)); alt(2, :) = firstTwo(lq, sp);
    Q = evaluator(mdl, x, 'hardkill', 0.1); kq = 12;
    while true
        lq = rawModes(Q, kq); if nnz(lq > 1) >= 2 || kq >= 384, break; end, kq = 2*kq;
    end
    sh = find(lq > 1); alt(3, :) = firstTwo(lq, sh);
    fprintf(['%-34s omega1/2 = %8.3f / %8.3f (raw idx %s, %d rejected below omega2) | grid omega1 %.3f-%.3f, ' ...
        'omega2 %.3f-%.3f | Emin1e-9 %.3f/%.3f, Pedersen %.3f/%.3f, kill<0.1 %.3f/%.3f\n'], lbl, om0(1), om0(2), ...
        mat2str(sel(1:2)'), sum(~keep(1:top)), min(oms(:,1)), max(oms(:,1)), min(oms(:,2)), max(oms(:,2)), alt');
    rows(end+1, :) = {lbl, om0(1), om0(2), sel(1), sel(2), sum(~keep(1:top)), min(oms(:,1)), max(oms(:,1)), ...
        min(oms(:,2)), max(oms(:,2)), alt(1,1), alt(1,2), alt(2,1), alt(2,2), alt(3,1), alt(3,2)}; %#ok<AGROW>
end
TS = cell2table(rows, 'VariableNames', {'design', 'omega1', 'omega2', 'raw_idx1', 'raw_idx2', 'n_rejected_below_omega2', ...
    'grid_omega1_min', 'grid_omega1_max', 'grid_omega2_min', 'grid_omega2_max', 'emin1e9_omega1', 'emin1e9_omega2', ...
    'pedersen_omega1', 'pedersen_omega2', 'kill01_omega1', 'kill01_omega2'});
TR = cell2table(rej, 'VariableNames', {'design', 'raw_index', 'omega', 'ke_share_x_le_0p1', 'se_share_x_le_0p1', ...
    'density_participation', 'ke_x_lt_0p1', 'ke_x_0p1_0p3'});
writetable(TS, fullfile(outDir, 'c2b_summary.csv')); writetable(TR, fullfile(outDir, 'c2b_rejected.csv'));
disp(TS); out = struct('summary', TS, 'rejected', TR);
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

function o = firstTwo(lam, idx)
% omega of the first two selected modes; NaN where fewer were found
o = [NaN NaN]; n = min(2, numel(idx)); o(1:n) = sqrt(lam(idx(1:n)));
end

function P = evaluator(mdl, x, kind, par)
P.kind = kind;
switch kind
    case 'emin'
        Emin = par * mdl.E0;
        P.Ee = Emin + x.^mdl.penal * (mdl.E0 - Emin);
        P.re = mdl.rho_min + x * (mdl.rho0 - mdl.rho_min);
    case 'pedersen'
        Emin = par * mdl.E0;
        P.Ee = Emin + x.^mdl.penal * (mdl.E0 - Emin);
        r = x; lo = x < 0.1; r(lo) = 1e5 * x(lo).^6;
        P.re = mdl.rho_min + r * (mdl.rho0 - mdl.rho_min);
    case 'hardkill'
        on = x >= par;
        P.Ee = double(on) .* (x.^mdl.penal * mdl.E0);
        P.re = double(on) .* (x * mdl.rho0);
end
K = sparse(mdl.iK, mdl.jK, reshape(mdl.KE(:) * P.Ee', [], 1), mdl.ndof, mdl.ndof);
M = sparse(mdl.iK, mdl.jK, reshape(mdl.ME(:) * P.re', [], 1), mdl.ndof, mdl.ndof);
K = (K + K')/2; M = (M + M')/2;
P.M = M;
if strcmp(kind, 'hardkill')
    % DOFs not attached to any kept element are dropped.
    used = false(mdl.ndof, 1); used(unique(mdl.edofMat(P.Ee > 0, :))) = true;
    P.free = mdl.free(used(mdl.free));
else
    P.free = mdl.free;
end
P.Kf = K(P.free, P.free); P.Mf = M(P.free, P.free);
end

function [lam, V] = rawModes(P, k)
opts = struct('tol', 1e-10, 'maxit', 20000, 'v0', sin((1:size(P.Kf,1))' * 0.7071067811865476) + 0.5);
if strcmp(P.kind, 'hardkill')
    % K may be singular (loose fragments): shift below zero.
    [V, D] = eigs(P.Kf, P.Mf, k, -1, opts);
else
    [V, D] = eigs(P.Kf, P.Mf, k, 'smallestabs', opts);
end
[lam, ix] = sort(real(diag(D))); V = V(:, ix);
end

function [lam, V, met] = modesWithMetrics(mdl, P, x, tLow, k0, nStruct)
% Lowest modes, doubling the batch until nStruct structural ones are found
% (under the default share cut 0.5 and this low-density threshold).
k = k0;
while true
    [lam, V] = rawModes(P, k);
    met = metrics(mdl, P, x, lam, V, tLow);
    if nnz(classify(met, 0.5, x)) >= nStruct || k >= 640, break; end
    k = 2*k;
end
end

function met = metrics(mdl, P, x, lam, V, tLow)
% Columns: residual, KE share x<=tLow, SE share x<=tLow, density participation,
% KE share x<=0.01.
low = x <= tLow; vlow = x <= 0.01;
met = NaN(numel(lam), 5);
U = zeros(mdl.ndof, 1);
for j = 1:numel(lam)
    u = V(:, j);
    met(j, 1) = norm(P.Kf*u - lam(j)*(P.Mf*u)) / (norm(P.Kf*u) + abs(lam(j))*norm(P.Mf*u) + eps);
    U(:) = 0; U(P.free) = u;
    ue = U(mdl.edofMat);
    ke = max(P.re .* sum((ue*mdl.ME) .* ue, 2), 0);
    se = max(P.Ee .* sum((ue*mdl.KE) .* ue, 2), 0);
    ken = ke/sum(ke); sen = se/sum(se);
    met(j, 2:5) = [sum(ken(low)), sum(sen(low)), sum(ken .* x), sum(ken(vlow))];
end
end

function b = keBands(mdl, P, x, phi)
% Kinetic-energy shares by element density band.
ue = phi(mdl.edofMat);
ke = max(P.re .* sum((ue*mdl.ME) .* ue, 2), 0); ke = ke / sum(ke);
b = [sum(ke(x < 0.1)), sum(ke(x >= 0.1 & x < 0.3)), sum(ke(x >= 0.3 & x < 0.7)), sum(ke(x >= 0.7))];
end

function keep = classify(met, s, x, pThr)
if nargin < 4, pThr = 0.5; end
uniform = max(x) - min(x) <= 1e-12;
keep = met(:, 1) <= 1e-6 & met(:, 2) < s & met(:, 3) < s & (uniform | met(:, 4) > pThr);
end

function [why, margin] = failReason(m, s, ~)
r = {};
if ~(m(1) <= 1e-6), r{end+1} = 'residual'; end
if ~(m(2) < s), r{end+1} = 'KE_low'; end
if ~(m(3) < s), r{end+1} = 'SE_low'; end
if ~(m(4) > 0.5), r{end+1} = 'participation'; end
if isempty(r), why = 'structural'; else, why = strjoin(r, '+'); end
margin = min(abs([m(2) - s, m(3) - s, m(4) - 0.5]));
end

function v = macMass(a, B, M)
% Classical MAC with the mass inner product: (a'MB)^2 / ((a'Ma)(b'Mb)).
v = zeros(1, size(B, 2));
Ma = M*a; aa = a'*Ma;
for k = 1:size(B, 2)
    b = B(:, k);
    v(k) = (Ma'*b)^2 / (aa * (b'*(M*b)));
end
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

