function R = c3_olhoff_checks(outDir)
%C3_OLHOFF_CHECKS  AppliSci revision C3, part 1: correctness of the Du-Olhoff
%   comparator (analysis/Olhoff) on small meshes, and the initial-design
%   frequency against Du & Olhoff (2007), Fig. 2a (omega1 = 68.7 rad/s).
%
%   Checks (production preset duOlhoffPedersenAdaptiveBoxSensitivityFiltered
%   unless stated):
%     A  assembly: symmetry, 3 rigid-body modes of the unsupported K, K and M
%        positive definite on the free dofs
%     B  eigenpairs: residual ||K phi - lam M phi||/||K phi||, M-orthonormality,
%        eigs (production) vs dense LAPACK
%     C  simple-eigenvalue gradients f_jj (genGrad, with the Sigmund filter OFF)
%        vs central finite differences, Pedersen and eq. (4b) material laws,
%        designs with elements on both sides of the 0.1 threshold
%     D  generalized off-diagonal gradients f_sk: f_sk'd vs the FD directional
%        derivative phi_s'(dK - lam~ dM)phi_k
%     E  two-mode subspace (deltaLambda with diagonal offsets) on a design with
%        a near-double eigenvalue: predicted lambda_j + dlam_j vs exact
%        eigenvalues along a step t*d; error must be O(t^2)
%     F  initial uniform design (rho = 0.5) omega1 for support idealizations
%        and meshes, vs the printed 68.7 rad/s
%
%   Results: results/c3/c3_checks.txt and c3_checks.mat

here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));
addpath(fullfile(repoRoot, 'analysis', 'Olhoff'));
if nargin < 1 || isempty(outDir), outDir = fullfile(here, 'results', 'c3'); end
if ~exist(outDir, 'dir'), mkdir(outDir); end
logFile = fullfile(outDir, 'c3_checks.txt');
if exist(logFile, 'file'), delete(logFile); end
diary(logFile); cleanupDiary = onCleanup(@() diary('off'));

guard = olhoffcurrent_paths(); %#ok<NASGU>
prod = olhoffcurrent_production_preset();
fprintf('C3 checks, preset %s, %s\n', prod.name, datestr(now, 31));
R = struct(); R.AB = [];

% ---------------------------------------------------------------- A, B
for mesh = [32 4; 48 6; 160 20].'
    nelx = mesh(1); nely = mesh(2);
    [mdl, cfg, flat] = buildModel(nelx, nely, prod.name, {}); %#ok<ASGLU>
    stiff = flat_stiff(cfg); mass = flat_mass(cfg);
    rng(1);
    rho = 0.05 + 0.95*rand(mdl.nele, 1);              % random, both sides of 0.1
    % A: unsupported assembly
    mfull = mdl; mfull.free = (1:mdl.ndof)';
    [Kf, Mf] = assemble2D(mfull, rho, stiff, mass);
    symK = norm(Kf - Kf', 1)/norm(Kf, 1);  symM = norm(Mf - Mf', 1)/norm(Mf, 1);
    if mdl.ndof <= 800
        ev = sort(real(eig(full(Kf), full(Mf))));
        nRigid = sum(abs(ev) < 1e-8*max(abs(ev)));
    else
        ev = NaN; nRigid = NaN;
    end
    [K, M] = assemble2D(mdl, rho, stiff, mass);
    [~, pK] = chol(K); [~, pM] = chol(M);
    fprintf('\n[A] %dx%d: sym(K)=%.1e sym(M)=%.1e, unsupported zero modes=%g (expect 3), chol K ok=%d, M ok=%d\n', ...
        nelx, nely, symK, symM, nRigid, pK == 0, pM == 0);
    % B: eigenpairs, eigs vs dense
    J = 5;
    [w1, P1, l1] = eigSolve(K, M, J, 'eigs');
    res = zeros(J, 1);
    for j = 1:J, res(j) = norm(K*P1(:, j) - l1(j)*M*P1(:, j))/norm(K*P1(:, j)); end
    orth = norm(P1'*M*P1 - eye(J), 'fro');
    if mdl.ndof <= 4000
        w2 = eigSolve(K, M, J, 'dense');
        dEig = max(abs(w1 - w2)./w2);
    else
        dEig = NaN;
    end
    fprintf('[B] %dx%d: max residual %.1e, ||Phi''M Phi - I|| = %.1e, max|w_eigs - w_dense|/w = %.1e\n', ...
        nelx, nely, max(res), orth, dEig);
    R.AB(end+1, :) = [nelx nely symK symM nRigid pK==0 pM==0 max(res) orth dEig]; %#ok<AGROW>
end

% ---------------------------------------------------------------- C, D
nelx = 32; nely = 4;
laws = {'pedersen', {}; ...
        'eq4b_simp', {'material.stiffness.model', 'simp', 'material.mass.model', 'eq4b'}};
R.C = []; R.D = [];
for il = 1:size(laws, 1)
    [mdl, cfg] = buildModel(nelx, nely, prod.name, laws{il, 2});
    stiff = flat_stiff(cfg); mass = flat_mass(cfg);
    rng(2);
    rho = 0.02 + 0.96*rand(mdl.nele, 1);
    rho(abs(rho - 0.1) < 0.01) = 0.12;                % keep FD off the kink
    [K, M] = assemble2D(mdl, rho, stiff, mass);
    [~, Phi, lam] = eigSolve(K, M, 6, 'dense');
    gaps = diff(lam(1:4))./lam(1:3);
    h = 1e-6;
    E = randperm(mdl.nele, 30);
    for j = 1:3
        G = genGrad(mdl, rho, stiff, mass, Phi, lam(j), j);
        g = G(:, 1, 1);
        fd = zeros(numel(E), 1);
        for ie = 1:numel(E)
            e = E(ie);
            rp = rho; rp(e) = rp(e) + h; rm = rho; rm(e) = rm(e) - h;
            [Kp, Mp] = assemble2D(mdl, rp, stiff, mass); [Km, Mm] = assemble2D(mdl, rm, stiff, mass);
            lp = eigSolve(Kp, Mp, 6, 'dense'); lm = eigSolve(Km, Mm, 6, 'dense');
            fd(ie) = (lp(j)^2 - lm(j)^2)/(2*h);
        end
        err = norm(g(E) - fd)/norm(fd);
        fprintf('[C] %s, mode %d (rel. gap to next %.2f): ||f_jj - FD||/||FD|| = %.2e over %d elements\n', ...
            laws{il, 1}, j, gaps(j), err, numel(E));
        R.C(end+1, :) = [il j err];
    end
    % D: off-diagonal generalized gradients along a random direction
    d = randn(mdl.nele, 1);
    t = 1e-6;
    [Kp, Mp] = assemble2D(mdl, rho + t*d, stiff, mass);
    [Km, Mm] = assemble2D(mdl, rho - t*d, stiff, mass);
    dK = (Kp - Km)/(2*t); dM = (Mp - Mm)/(2*t);
    lt = lam(1);
    F = genGrad(mdl, rho, stiff, mass, Phi, lt, 1:3);
    for s = 1:3
        for k = s:3
            ref = Phi(:, s)'*(dK - lt*dM)*Phi(:, k);
            val = F(:, s, k)'*d;
            fprintf('[D] %s, f_%d%d''d = %+.6e, FD = %+.6e, rel err %.1e\n', ...
                laws{il, 1}, s, k, val, ref, abs(val - ref)/max(abs(ref), 1e-12));
            R.D(end+1, :) = [il s k val ref];
        end
    end
end

% ---------------------------------------------------------------- E
% near-double eigenvalue: a symmetric design scaled until omega1 ~ omega2 is
% hard to construct directly; use the saved 160x20 production design if
% available (bimodal at 160x20), else a uniform design with a stiffened
% centre region tuned by bisection.
E = struct();
dsg = fullfile(outDir, 'olhoff_prod_160x20.mat');
if exist(dsg, 'file')
    S = load(dsg); [mdl, cfg] = buildModel(160, 20, prod.name, {}); rho = S.x(:);
    stiff = flat_stiff(cfg); mass = flat_mass(cfg);
    [K, M] = assemble2D(mdl, rho, stiff, mass);
    [w, Phi, lam] = eigSolve(K, M, 5, 'eigs');
    fprintf('\n[E] 160x20 production design: omega = %s rad/s, gap12 = %.2f%%\n', ...
        mat2str(w(1:3)', 6), 100*(w(2) - w(1))/w(1));
    idx = 1:2;
    F = genGrad(mdl, rho, stiff, mass, Phi, lam(1), idx);
    for j = 1:2
        Gj = genGrad(mdl, rho, stiff, mass, Phi, lam(j), j); F(:, j, j) = Gj(:, 1, 1);
    end
    dOff = lam(idx) - lam(1);
    rng(3); d = randn(mdl.nele, 1).*(rho > 0.2 & rho < 0.95); d = d/norm(d, inf);   % stay off the 0.1 kink and the bounds
    fprintf('[E] perturbed elements: %d of %d (0.2 < rho < 0.95)\n', nnz(d), mdl.nele);
    ts = 0.02*2.^-(0:4);
    errs = zeros(numel(ts), 2); errsLin = errs;
    for it = 1:numel(ts)
        t = ts(it);
        dl = deltaLambda(F, t*d, dOff);
        [Kp, Mp] = assemble2D(mdl, rho + t*d, stiff, mass);
        [~, ~, lp] = eigSolve(Kp, Mp, 5, 'eigs');
        errs(it, :) = abs(lam(idx) + dl - lp(idx))'./lam(1);
        dlin = [F(:, 1, 1)'*d; F(:, 2, 2)'*d]*t;           % diagonal-only (simple) prediction
        errsLin(it, :) = abs(sort(lam(idx) + dlin) - lp(idx))'./lam(1);
        fprintf('[E] t=%.5f: rel. error of (25d) prediction %.2e %.2e | diagonal-only %.2e %.2e\n', ...
            t, errs(it, :), errsLin(it, :));
    end
    ord = log2(errs(1:end-1, :)./errs(2:end, :));
    fprintf('[E] observed order of the (25d) prediction error: %s (expect ~2)\n', mat2str(mean(ord), 3));
    E.ts = ts; E.errs = errs; E.errsLin = errsLin; E.w = w;
else
    fprintf('\n[E] skipped: %s not found (run c3_olhoff_reproduce first)\n', dsg);
end
R.E = E;

% ---------------------------------------------------------------- F
fprintf('\n[F] initial uniform design rho = 0.5, omega1 [rad/s] (Du & Olhoff Fig. 2a: 68.7)\n');
sup = {'midHeight', 'bothEnds'; 'midHeight', 'oneEnd'; 'corner', 'bothEnds'; 'corner', 'oneEnd'; 'face', 'bothEnds'};
meshes = [80 10; 160 20; 320 40; 400 50; 800 100];
R.F = NaN(size(sup, 1), size(meshes, 1), 2);
fprintf('%-22s', 'support / mesh');
for im = 1:size(meshes, 1), fprintf('%10s', sprintf('%dx%d', meshes(im, 1), meshes(im, 2))); end
fprintf('\n');
for is = 1:size(sup, 1)
    for el = 1:2
        etype = {'Q4', 'Q6'}; etype = etype{el};
        fprintf('%-22s', sprintf('%s/%s %s', sup{is, 1}, sup{is, 2}, etype));
        for im = 1:size(meshes, 1)
            [mdl, cfg] = buildModel(meshes(im, 1), meshes(im, 2), prod.name, ...
                {'domain.boundary.support', sup{is, 1}, 'domain.boundary.axialRestraint', sup{is, 2}, ...
                 'domain.element.type', etype});
            [K, M] = assemble2D(mdl, 0.5*ones(mdl.nele, 1), flat_stiff(cfg), flat_mass(cfg));
            w = eigSolve(K, M, 3, 'eigs');
            R.F(is, im, el) = w(1);
            fprintf('%10.2f', w(1));
        end
        fprintf('\n');
    end
end
R.sup = sup; R.meshes = meshes;
save(fullfile(outDir, 'c3_checks.mat'), 'R');
end

% ======================================================================
function [mdl, cfg, flat] = buildModel(nelx, nely, preset, overrides)
cfg = olhoffcurrent_config(nelx, nely, 'Preset', preset);
if ~isempty(overrides), cfg = olh.config.assign(cfg, overrides{:}); end
flat = olh.config.toLegacy(cfg);
mdl = model2D(flat);
end

function s = flat_stiff(cfg)
g = @(p) olh.config.getPath(cfg, p);
s = struct('model', g('material.stiffness.model'), 'p', g('material.stiffness.p'), ...
           'linearBelow', g('material.stiffness.linearBelow'));
end

function m = flat_mass(cfg)
m = olh.config.getPath(cfg, 'material.mass');
end
