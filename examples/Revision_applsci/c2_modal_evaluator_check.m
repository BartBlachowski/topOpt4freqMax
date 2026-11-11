function out = c2_modal_evaluator_check(outDir)
%C2_MODAL_EVALUATOR_CHECK  AppliSci revision C2: validate the modal evaluator.
%
%   out = c2_modal_evaluator_check()        writes results under results/c2/
%
%   Works on the ten designs of the alpha sweep (results/<case>/alpha_*/
%   result.mat: clamped beam 400x50, building 80x240).  Nothing is optimized.
%
%   1. Rejected-mode audit.  The lowest modes of each design under the paper's
%      evaluator (E_min = 1e-6 E0, rho_min = 1e-6, p = 3, linear mass) are
%      classified with the criteria of structural_mode_mask, and every metric
%      is written out: eigen-residual, kinetic- and strain-energy shares in
%      elements with x <= 0.1, kinetic-energy-weighted density participation.
%      For each mode: verdict, which criterion failed, margin to the nearest
%      threshold, the largest (Euclidean) MAC with an accepted mode, and the
%      kinetic-energy shares in the density bands <0.1, 0.1-0.3, 0.3-0.7, >=0.7.
%   2. Threshold sensitivity.  The same modes are reclassified with the
%      low-density threshold t in {0.05, 0.1, 0.2, 0.3} and the energy-share
%      cut s in {0.3, 0.5, 0.7}; reported: structural omega1/omega2.
%   3. Evaluator sensitivity.  Structural omega1/omega2 recomputed with
%        E_min / E0 in {1e-9, 1e-6 (paper), 1e-4, 1e-3},
%        Pedersen mass (rho = 1e5 x^6 for x < 0.1, E_min = 1e-6 E0),
%        hard kill (elements with x < 0.1, 0.3 or 0.5 removed, no classifier;
%        rigid-body/hinge modes of loose fragments, omega < 1 rad/s, dropped),
%      and the MAC of their first two structural modes with the paper
%      evaluator's.  A classifier that works gives the same structural
%      frequencies under every evaluator, while void-mode frequencies move
%      with sqrt(E_min).

here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
if nargin < 1 || isempty(outDir)
    outDir = fullfile(here, 'results', 'c2');
end
if ~exist(outDir, 'dir'), mkdir(outDir); end
logFile = fullfile(outDir, 'c2_log.txt');
if exist(logFile, 'file'), delete(logFile); end
diary(logFile); cleanupDiary = onCleanup(@() diary('off'));

cases = {'clamped', 'building'};
tags = {'1p00', '0p75', '0p50', '0p25', '0p00'};
audit = {}; thr = {}; ev = {};
for ic = 1:numel(cases)
    name = cases{ic};
    for it = 1:numel(tags)
        S = load(fullfile(here, 'results', name, ['alpha_' tags{it}], 'result.mat'));
        a = S.a;
        mdl = buildModel(S.cfg);
        x = max(0, min(1, S.x(:)));
        fprintf('\n===== %s alpha=%.2f (%dx%d) =====\n', name, a, mdl.nelx, mdl.nely);

        % --- 1. paper evaluator, all metrics ---------------------------
        P = evaluator(mdl, x, 'emin', 1e-6);
        [lam, V, met] = modesWithMetrics(mdl, P, x, 0.1, 40, 2);
        keep = classify(met, 0.5, x);
        % cross-check against the repository classifier
        keepRepo = structural_mode_mask(V, lam, P.Kf, P.Mf, mdl.free, mdl.ndof, ...
            mdl.edofMat, mdl.KE, mdl.ME, P.Ee, P.re, x);
        assert(isequal(keep, keepRepo), 'c2: classifier copy disagrees with structural_mode_mask');
        sel = find(keep);
        Phi = zeros(mdl.ndof, numel(lam)); Phi(mdl.free, :) = V;
        Mfull = P.M;
        top = sel(min(2, numel(sel)));
        nShow = min(numel(lam), top + 3);
        for j = 1:nShow
            [why, margin] = failReason(met(j, :), 0.5, x);
            % Mass-orthogonal eigenvectors have mass-MAC 0 by construction, so
            % the overlap with the accepted modes uses the Euclidean MAC.
            others = sel(sel ~= j);
            macAcc = 0;
            if ~isempty(others), macAcc = max(macMass(Phi(:, j), Phi(:, others), speye(mdl.ndof))); end
            bands = keBands(mdl, P, x, Phi(:, j));
            audit(end+1, :) = {name, a, j, sqrt(lam(j)), keep(j), why, met(j, 1), ...
                met(j, 2), met(j, 3), met(j, 4), met(j, 5), margin, macAcc, ...
                bands(1), bands(2), bands(3), bands(4)}; %#ok<AGROW>
        end
        nRej = sum(~keep(1:top));
        fprintf('[audit] %d raw modes up to structural omega2; %d rejected; structural omega1/2 = %.4f / %.4f (raw idx %s)\n', ...
            top, nRej, sqrt(lam(sel(1))), sqrt(lam(top)), mat2str(sel(1:min(2, end))'));

        % --- 2. threshold sensitivity on the same evaluator -------------
        for t = [0.05 0.1 0.2 0.3]
            [lamT, ~, metT] = modesWithMetrics(mdl, P, x, t, 40, 2);
            for s = [0.3 0.5 0.7]
                kT = classify(metT, s, x);
                sT = find(kT);
                n2 = min(2, numel(sT));
                om = [NaN NaN]; om(1:n2) = sqrt(lamT(sT(1:n2)));
                if n2 == 2, nr = sum(~kT(1:sT(2))); else, nr = sum(~kT); end
                thr(end+1, :) = {name, a, t, s, om(1), om(2), nr}; %#ok<AGROW>
            end
        end

        % --- 3. evaluator sensitivity -----------------------------------
        refPhi = Phi(:, sel(1:2));
        evs = {'emin_1e-9', 'emin', 1e-9; 'emin_1e-6_paper', 'emin', 1e-6; ...
               'emin_1e-4', 'emin', 1e-4; 'emin_1e-3', 'emin', 1e-3; ...
               'pedersen', 'pedersen', 1e-6; 'hardkill_0.1', 'hardkill', 0.1; ...
               'hardkill_0.3', 'hardkill', 0.3; 'hardkill_0.5', 'hardkill', 0.5};
        for k = 1:size(evs, 1)
            Q = evaluator(mdl, x, evs{k, 2}, evs{k, 3});
            if strcmp(evs{k, 2}, 'hardkill')
                % Rigid-body / hinge mechanisms of loose fragments (omega < 1 rad/s)
                % are not structural modes; they are counted as rejected.
                kq = 12;
                while true
                    [lamQ, VQ] = rawModes(Q, kq);
                    if nnz(lamQ > 1) >= 2 || kq >= 384, break; end
                    kq = 2*kq;
                end
                selQ = find(lamQ > 1); nRejQ = sum(lamQ <= 1); lam1raw = lamQ(1);
                PhiQ = zeros(mdl.ndof, numel(lamQ)); PhiQ(Q.free, :) = VQ;
            else
                [lamQ, VQ, metQ] = modesWithMetrics(mdl, Q, x, 0.1, 40, 2);
                kQ = classify(metQ, 0.5, x); selQ = find(kQ);
                nRejQ = sum(~kQ(1:selQ(min(2, end)))); lam1raw = lamQ(1);
                PhiQ = zeros(mdl.ndof, numel(lamQ)); PhiQ(mdl.free, :) = VQ;
            end
            om = sqrt(lamQ(selQ(1:2)));
            mac1 = macMass(refPhi(:, 1), PhiQ(:, selQ(1)), Mfull);
            mac2 = macMass(refPhi(:, 2), PhiQ(:, selQ(2)), Mfull);
            fprintf('[eval] %-16s omega1=%.4f omega2=%.4f  raw omega1=%.4f  rejected below=%d  MAC=%.4f/%.4f\n', ...
                evs{k, 1}, om(1), om(2), sqrt(lam1raw), nRejQ, mac1, mac2);
            ev(end+1, :) = {name, a, evs{k, 1}, om(1), om(2), sqrt(lam1raw), nRejQ, mac1, mac2}; %#ok<AGROW>
        end
    end
end
TA = cell2table(audit, 'VariableNames', {'case', 'alpha', 'raw_index', 'omega', 'structural', ...
    'reason', 'residual', 'ke_low_share', 'se_low_share', 'density_participation', ...
    'ke_share_x_le_0p01', 'margin_to_threshold', 'max_euclidMAC_with_structural', ...
    'ke_x_lt_0p1', 'ke_x_0p1_0p3', 'ke_x_0p3_0p7', 'ke_x_ge_0p7'});
TT = cell2table(thr, 'VariableNames', {'case', 'alpha', 'low_density_threshold', ...
    'energy_share_cut', 'omega1', 'omega2', 'n_rejected_below_omega2'});
TE = cell2table(ev, 'VariableNames', {'case', 'alpha', 'evaluator', 'omega1', 'omega2', ...
    'raw_omega1', 'n_rejected_below_omega2', 'MAC1_vs_paper', 'MAC2_vs_paper'});
writetable(TA, fullfile(outDir, 'c2_rejected_mode_audit.csv'));
writetable(TT, fullfile(outDir, 'c2_threshold_sensitivity.csv'));
writetable(TE, fullfile(outDir, 'c2_evaluator_sensitivity.csv'));
out = struct('audit', TA, 'thresholds', TT, 'evaluators', TE);
save(fullfile(outDir, 'c2_results.mat'), 'out');
end

% ======================================================================
function mdl = buildModel(cfg)
L = cfg.domain.size.length; H = cfg.domain.size.height;
nelx = cfg.domain.mesh.nelx; nely = cfg.domain.mesh.nely;
mdl.nelx = nelx; mdl.nely = nely; mdl.nEl = nelx*nely;
hx = L/nelx; hy = H/nely;
mdl.E0 = cfg.material.E; mdl.nu = cfg.material.nu; mdl.rho0 = cfg.material.rho;
mdl.rho_min = cfg.void_material.rho_min; mdl.penal = cfg.optimization.penalization;
mdl.KE = lk(hx, hy, mdl.nu); mdl.ME = lm(hx, hy);
edofMat = zeros(mdl.nEl, 8);
for elx = 0:nelx-1
    for ely = 0:nely-1
        el = ely + elx*nely + 1;
        n1 = (nely+1)*elx + ely; n2 = (nely+1)*(elx+1) + ely; n3 = n2 + 1; n4 = n1 + 1;
        edofMat(el, :) = [2*n1+1, 2*n1+2, 2*n2+1, 2*n2+2, 2*n3+1, 2*n3+2, 2*n4+1, 2*n4+2];
    end
end
mdl.edofMat = edofMat;
mdl.iK = reshape(kron(edofMat, ones(1,8))', [], 1);
mdl.jK = reshape(kron(edofMat, ones(8,1))', [], 1);
mdl.ndof = 2*(nelx+1)*(nely+1);
nodeIds = (1:(nelx+1)*(nely+1))';
nx = floor((nodeIds-1)/(nely+1))*hx; ny = mod(nodeIds-1, nely+1)*hy;
fixedNodes = [];
sup = cfg.bc.supports;
if ~iscell(sup), sup = num2cell(sup); end
for k = 1:numel(sup)
    s = sup{k};
    switch s.type
        case 'vertical_line',   fixedNodes = [fixedNodes; nodeIds(abs(nx - s.x) < 1e-6)]; %#ok<AGROW>
        case 'horizontal_line', fixedNodes = [fixedNodes; nodeIds(abs(ny - s.y) < 1e-6)]; %#ok<AGROW>
        otherwise, error('c2: unsupported support type %s', s.type);
    end
end
fixedNodes = unique(fixedNodes);
mdl.free = setdiff((1:mdl.ndof)', sort([2*fixedNodes-1; 2*fixedNodes]));
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

function keep = classify(met, s, x)
uniform = max(x) - min(x) <= 1e-12;
keep = met(:, 1) <= 1e-6 & met(:, 2) < s & met(:, 3) < s & (uniform | met(:, 4) > 0.5);
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

function KE = lk(hx, hy, nu)
D = (1/(1-nu^2)) * [1, nu, 0; nu, 1, 0; 0, 0, 0.5*(1-nu)];
invJ = [2/hx, 0; 0, 2/hy]; detJ = 0.25*hx*hy; gp = 1/sqrt(3);
KE = zeros(8, 8);
for xi = [-gp, gp]
    for eta = [-gp, gp]
        dN = invJ * (0.25 * [-(1-eta), (1-eta), (1+eta), -(1+eta); -(1-xi), -(1+xi), (1+xi), (1-xi)]);
        B = zeros(3, 8);
        B(1, 1:2:end) = dN(1, :); B(2, 2:2:end) = dN(2, :);
        B(3, 1:2:end) = dN(2, :); B(3, 2:2:end) = dN(1, :);
        KE = KE + (B'*D*B)*detJ;
    end
end
KE = (KE + KE')/2;
end

function ME = lm(hx, hy)
ME = kron((hx*hy/36) * [4 2 1 2; 2 4 2 1; 1 2 4 2; 2 1 2 4], eye(2));
end
