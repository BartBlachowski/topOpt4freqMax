function T = c8_c9_direct(outDir, jobs)
%C8_C9_DIRECT  AppliSci revision C8 (direct eigenfrequency optimization vs the
%   proposed method at nominal settings) and C9 (beam volume-fraction sweep,
%   building reference-mode choice).
%
%   T = c8_c9_direct()                       all jobs
%   T = c8_c9_direct([], {'c8_clamped'})     a subset
%
%   Jobs
%     c8_clamped     clamped beam 400x50 and 160x20, V_f = 0.5: proposed (alpha = 1)
%                    vs Du-Olhoff (production preset, clamped ends = Du & Olhoff
%                    2007 Fig. 2c, published omega1 = 456.4 rad/s)
%     c9_vf          simply supported beam 400x50, V_f = 0.3 / 0.5 / 0.7, both methods
%     c8_building_tests  checks of the passive-element / base-clamp extension of the
%                    Du-Olhoff implementation (regression, model, smoke run, FD)
%     c8_building    building 80x240, alpha = 1: proposed vs Du-Olhoff with passive
%                    frame (design.passiveSolid) and clamped base (clampedBottom);
%                    Du-Olhoff filter radius 3 elements (0.1875 m), as 0.06 m at 400x50
%     c9_reference   building 80x240, proposed only, alpha = 1, three references:
%                      solid   mode 1 of the solid domain (the paper)
%                      graded  mode 1 of a nonuniform feasible design (density
%                              decreasing linearly with height, mean = V_f)
%                      perturb solid mode + 10% smooth perturbation (BCs kept,
%                              M-normalized; perturbation size printed)
%
%   Proposed: standalone copy of the topopt_freq loop (omitted dF/dx, OC, r = 2
%   elements, move 0.2, stagnation_x W = 10, 1e-2), identical to C4 variant (a),
%   which reproduces the alpha-sweep designs.  Material floors as in the paper
%   (SS 1e-9, clamped / building 1e-6).
%   Du-Olhoff: duOlhoffAdaptivePedersen (production preset of Table 1) with the
%   boundary condition / volume fraction overridden and the same stagnation_x
%   rule (W = 10, 1e-2, guards off), cap 1000 outer iterations.
%   Quality: lowest two structural frequencies of every final design under the
%   common evaluator (SIMP p = 3, E_min = 1e-6 E0, linear mass rho_min = 1e-6,
%   structural_mode_mask); the Du-Olhoff design is also reported in its own model.

here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));
addpath(fullfile(repoRoot, 'tools', 'Matlab'), fullfile(repoRoot, 'analysis', 'Olhoff'));
if nargin < 1 || isempty(outDir), outDir = fullfile(here, 'results', 'c8c9'); end
if nargin < 2 || isempty(jobs), jobs = {'c8_clamped', 'c9_vf', 'c9_reference'}; end
if ~exist(outDir, 'dir'), mkdir(outDir); end
logFile = fullfile(outDir, sprintf('c8c9_log_%s.txt', strjoin(jobs, '_')));
if exist(logFile, 'file'), delete(logFile); end
diary(logFile); cleanupDiary = onCleanup(@() diary('off'));
maxNumCompThreads(1);
fprintf('C8/C9 %s on %s, MATLAB %s\n', datestr(now, 31), strtrim(evalc('system(''hostname'');')), version);

rows = {};
for ij = 1:numel(jobs)
    switch jobs{ij}
        case 'c8_clamped'
            for mesh = [400 50; 160 20]'
                rows = [rows; compareCase(repoRoot, outDir, 'clamped', mesh(1), mesh(2), 0.5, 'C8')]; %#ok<AGROW>
            end
        case 'c9_vf'
            for vf = [0.3 0.5 0.7]
                rows = [rows; compareCase(repoRoot, outDir, 'ss', 400, 50, vf, 'C9')]; %#ok<AGROW>
            end
        case 'c9_reference'
            rows = [rows; referenceStudy(repoRoot, outDir)]; %#ok<AGROW>
        case 'c8_building_tests'
            buildingTests(repoRoot);
        case 'c6_building_start'
            buildingStart(repoRoot);
        case 'c4_equality'
            equalityAblation(repoRoot, outDir);
        case 'c11_history'
            spectralHistory(repoRoot, outDir);
        case 'c8_building'
            rows = [rows; compareCase(repoRoot, outDir, 'building', 80, 240, [], 'C8')]; %#ok<AGROW>
    end
    if isempty(rows), T = table(); continue; end
    T = cell2table(rows, 'VariableNames', {'job', 'case', 'mesh', 'volfrac', 'method', 'reference', ...
        'iterations', 'inner', 'stop', 'time_s', 'volume', 'grey', 'omega1', 'omega2', 'omega1_native', ...
        'omega2_native', 'omega1_initial'});
    writetable(T, fullfile(outDir, sprintf('c8c9_summary_%s.csv', strjoin(jobs, '_'))));
end
disp(T);
end

% ======================================================================
function rows = compareCase(repoRoot, outDir, name, nelx, nely, vf, job)
mdl = buildModel(repoRoot, name, nelx, nely, vf);
vf = mdl.volfrac;
ev = mdl; ev.Emin = 1e-6 * mdl.E0; ev.rho_min = 1e-6;
x0 = initialDesign(mdl);
om0 = structuralOmega(ev, x0);
tag = sprintf('%s_%dx%d_vf%02.0f', name, nelx, nely, 100*vf);
fprintf('\n===== %s %s: omega0(solid) = %.3f, initial uniform omega1 = %.3f =====\n', job, tag, mdl.omega0(1), om0(1));
% proposed
t0 = tic; r = optimize(mdl, [1; 0], false); tP = toc(t0);
omP = structuralOmega(ev, r.x);
fprintf('[%s] %s proposed: it=%d (%s) %.1f s vol=%.4f grey=%.3f omega1/2 = %.3f / %.3f\n', ...
    job, tag, r.iter, r.stop, tP, mean(r.x), 4*mean(r.x.*(1-r.x)), omP(1), omP(2));
rows = {job, name, sprintf('%dx%d', nelx, nely), vf, 'proposed', 'solid', r.iter, NaN, r.stop, tP, ...
    mean(r.x), 4*mean(r.x.*(1-r.x)), omP(1), omP(2), NaN, NaN, om0(1)};
xP = r.x;
% Du-Olhoff
guard = olhoffcurrent_paths(); %#ok<NASGU>
bc = 'simplySupported'; extra = {};
if strcmp(name, 'clamped'), bc = 'clamped'; end
if strcmp(name, 'building')
    bc = 'clampedBottom';
    extra = {'domain.a', mdl.L, 'domain.b', mdl.H, 'design.passiveSolid', toModel2D(mdl, mdl.pasS), ...
        'filter.radiusPhysical', 3*mdl.L/nelx};
end
pre = olhoffcurrent_preset(olhoffcurrent_production_preset().name);
cfg = olh.config.resolve(pre.upstreamPreset, 'domain.mesh.nelx', nelx, 'domain.mesh.nely', nely, ...
    pre.overrides{:}, 'domain.boundary.condition', bc, 'design.volumeFraction', vf, extra{:}, ...
    'stop.rule', 'stagnation_x', 'stop.guards.settledMove', false, 'stop.guards.boxInactiveFraction', 0, ...
    'stop.guards.ladderExhausted', false, 'stop.guards.maxDesignChange', false, ...
    'stop.stagnation.window', 10, 'stop.stagnation.relativeChangeTolerance', 1e-2, ...
    'stop.stagnation.graynessTolerance', 1e-2, 'runtime.maxOuter', 1000, 'runtime.singleThread', true, ...
    'runtime.verbose', false);
t0 = tic; res = olhoffSolve(cfg); tO = toc(t0);
X = reshape(double(res.rho(:)), nely, nelx); xO = reshape(flipud(X), [], 1);   % model2D -> this numbering
omO = structuralOmega(ev, xO);
w = double(res.omega(:));
h = res.hist;
fprintf('[%s] %s Du-Olhoff: outer=%d inner=%d (%s) %.1f s vol=%.4f grey=%.3f omega1/2 common = %.3f / %.3f, native = %.3f / %.3f\n', ...
    job, tag, numel(h.N), sum(h.nInner), char(res.status), tO, mean(xO), 4*mean(xO.*(1-xO)), omO(1), omO(2), w(1), w(2));
rows(end+1, :) = {job, name, sprintf('%dx%d', nelx, nely), vf, 'du-olhoff', '-', numel(h.N), sum(h.nInner), ...
    char(res.status), tO, mean(xO), 4*mean(xO.*(1-xO)), omO(1), omO(2), w(1), w(2), om0(1)};
save(fullfile(outDir, ['c8c9_' tag '.mat']), 'xP', 'xO', 'omP', 'omO', 'w', 'cfg', 'nelx', 'nely', 'vf');
saveDesignPng(mdl, {xP, xO}, {sprintf('proposed \\omega_1=%.1f', omP(1)), ...
    sprintf('Du-Olhoff \\omega_1=%.1f', omO(1))}, fullfile(outDir, ['c8c9_' tag '.png']));
end

function rows = referenceStudy(repoRoot, outDir)
mdl = buildModel(repoRoot, 'building', 80, 240, []);
ev = mdl; ev.Emin = 1e-6 * mdl.E0; ev.rho_min = 1e-6;
f = mdl.free;
om0 = structuralOmega(ev, initialDesign(mdl));
refs = {'solid', 'graded', 'perturb'};
rows = {}; xs = {}; ttl = {};
phiSolid = mdl.phi0(:, 1); omSolid = mdl.omega0(1);
for ir = 1:numel(refs)
    m = mdl;
    switch refs{ir}
        case 'solid'
            phi = phiSolid; om = omSolid;
        case 'graded'
            % density decreasing linearly with height, active mean = that of the uniform start
            xg = initialDesign(mdl);
            yc = repmat(((1:mdl.nely)' - 0.5) / mdl.nely, mdl.nelx, 1);   % el = ely + elx*nely, ely from the base
            prof = 1.6 - 1.2 * yc;                                         % 1.6 at the base, 0.4 at the top
            xg(mdl.act) = xg(mdl.act) .* prof(mdl.act) / mean(prof(mdl.act));
            xg = min(1, max(1e-3, xg));
            [K, M] = assemble(mdl, xg);
            [V, D] = eigs(K(f,f), M(f,f), 1, 'smallestabs', struct('tol', 1e-12, 'v0', startVec(numel(f))));
            phi = zeros(mdl.ndof, 1); phi(f) = V; om = sqrt(real(D));
            [~, Ms] = assemble(mdl, ones(mdl.nEl, 1));   % normalize in the solid-domain mass, as phi0
            phi = phi / sqrt(phi(f)' * Ms(f,f) * phi(f));
            fprintf('graded reference: omega = %.3f (solid %.3f), mean x = %.4f, MAC with solid mode = %.4f\n', ...
                om, omSolid, mean(xg), mac(phi(f), phiSolid(f), Ms(f,f)));
        case 'perturb'
            [~, Ms] = assemble(mdl, ones(mdl.nEl, 1));
            nn = mdl.ndof / 2; nodeIds = (1:nn)';
            nx = floor((nodeIds-1) / (mdl.nely+1)) / mdl.nelx; ny = mod(nodeIds-1, mdl.nely+1) / mdl.nely;
            pert = zeros(mdl.ndof, 1);
            pert(1:2:end) = sin(3*pi*ny) .* cos(pi*nx);    % smooth lateral perturbation
            pert(2:2:end) = 0.5 * sin(2*pi*ny) .* sin(2*pi*nx);
            pert(setdiff(1:mdl.ndof, f)) = 0;
            nP = sqrt(pert(f)' * Ms(f,f) * pert(f));
            phi = phiSolid + 0.10 * pert / nP;               % 10% of the M-norm of phi0 (=1)
            phi = phi / sqrt(phi(f)' * Ms(f,f) * phi(f));
            om = omSolid;
            fprintf('perturbed reference: ||delta||_M / ||phi0||_M = 0.10, MAC with solid mode = %.4f\n', ...
                mac(phi(f), phiSolid(f), Ms(f,f)));
    end
    m.phi0(:, 1) = phi; m.omega0(1) = om; m.b(:, 1) = phi * om^2;
    t0 = tic; r = optimize(m, [1; 0], false); tP = toc(t0);
    omP = structuralOmega(ev, r.x);
    fprintf('[C9 ref] building %s: it=%d (%s) %.1f s vol=%.4f grey=%.3f omega1/2 = %.3f / %.3f\n', ...
        refs{ir}, r.iter, r.stop, tP, mean(r.x), 4*mean(r.x.*(1-r.x)), omP(1), omP(2));
    rows(end+1, :) = {'C9', 'building', '80x240', mdl.volfrac, 'proposed', refs{ir}, r.iter, NaN, r.stop, ...
        tP, mean(r.x), 4*mean(r.x.*(1-r.x)), omP(1), omP(2), NaN, NaN, om0(1)}; %#ok<AGROW>
    xs{end+1} = r.x; ttl{end+1} = sprintf('%s \\omega_1=%.1f', refs{ir}, omP(1)); %#ok<AGROW>
end
save(fullfile(outDir, 'c8c9_building_reference.mat'), 'xs', 'refs');
saveDesignPng(mdl, xs, ttl, fullfile(outDir, 'c8c9_building_reference.png'));
end

function idx = toModel2D(mdl, el)
% element indices of this file's numbering (ely from the bottom) -> model2D (ely from the top)
m = false(mdl.nEl, 1); m(el) = true;
m2 = flipud(reshape(m, mdl.nely, mdl.nelx));
idx = find(m2(:));
end

function buildingTests(repoRoot)
guard = olhoffcurrent_paths(); %#ok<NASGU>
pre = olhoffcurrent_preset(olhoffcurrent_production_preset().name);
% A. regression: production SS beam 160x20 must reproduce the C3 design bit for bit
S = load(fullfile(fileparts(mfilename('fullpath')), 'results', 'c3', 'olhoff_prod_160x20.mat'), 'x', 'omega');
cfg = olh.config.resolve(pre.upstreamPreset, 'domain.mesh.nelx', 160, 'domain.mesh.nely', 20, ...
    'runtime.maxOuter', 400, pre.overrides{:});
res = olhoffSolve(cfg);
fprintf('[test A] SS 160x20 regression: max|x - x_C3| = %.3g, omega1 %.6f vs %.6f\n', ...
    max(abs(res.rho(:) - S.x(:))), res.omega(1), S.omega(1));
% B. model: building solid / uniform frequencies vs the proposed method's model
mdl = buildModel(repoRoot, 'building', 80, 240, []);
pas = toModel2D(mdl, mdl.pasS);
cfg = olh.config.resolve(pre.upstreamPreset, 'domain.mesh.nelx', 80, 'domain.mesh.nely', 240, pre.overrides{:}, ...
    'domain.boundary.condition', 'clampedBottom', 'domain.a', mdl.L, 'domain.b', mdl.H, ...
    'design.volumeFraction', 0.3, 'design.passiveSolid', pas, 'filter.radiusPhysical', 0.1875);
m2 = model2D(olh.config.toLegacy(cfg));
fprintf('[test B] dofs: model2D %d (fixed %d), proposed %d (fixed %d); passive %d = %d elements\n', ...
    m2.ndof, numel(m2.fixed), mdl.ndof, mdl.ndof - numel(mdl.free), numel(pas), numel(mdl.pasS));
st = struct('model', 'simp', 'p', 3, 'linearBelow', 0.1); ms = struct('model', 'eq2', 'q', 1, ...
    'lowDensityExponent', 6, 'cutoff', 0.1);
[K, M] = assemble2D(m2, ones(m2.nele, 1), st, ms); w = eigSolve(K, M, 2, 'eigs');
fprintf('[test B] solid domain omega1/2: model2D %.4f / %.4f, proposed model %.4f / %.4f\n', ...
    w(1), w(2), mdl.omega0(1), mdl.omega0(2));
x0 = initialDesign(mdl); [Kp, Mp] = assemble(mdl, x0); f = mdl.free;
wp = sqrt(sort(real(eigs(Kp(f,f), Mp(f,f), 2, 'smallestabs'))));
r0 = flipud(reshape(x0, mdl.nely, mdl.nelx)); r0 = r0(:);
[K, M] = assemble2D(m2, r0, st, struct('model', 'eq2', 'q', 1, 'lowDensityExponent', 6, 'cutoff', 0.1));
w = eigSolve(K, M, 2, 'eigs');
fprintf('[test B] uniform start (SIMP, E_min %g): model2D %.4f / %.4f, proposed model %.4f / %.4f\n', ...
    mdl.Emin/mdl.E0, w(1), w(2), wp(1), wp(2));
% C. smoke run: 3 outer iterations, passive stays solid, volume kept
cfgS = olh.config.assign(cfg, 'runtime.maxOuter', 3);
res = olhoffSolve(cfgS);
x = res.rho(:);
fprintf('[test C] 3 outer: min rho(passive) = %.17g, mean rho = %.6f (target 0.3), omega = %s, status %s\n', ...
    min(x(pas)), mean(x), mat2str(res.omega(1:2)', 6), char(res.status));
% D. finite differences of lambda_1 w.r.t. active densities (40x120, random design)
mdlC = buildModel(repoRoot, 'building', 40, 120, []); pasC = toModel2D(mdlC, mdlC.pasS);
cfgC = olh.config.resolve(pre.upstreamPreset, 'domain.mesh.nelx', 40, 'domain.mesh.nely', 120, pre.overrides{:}, ...
    'domain.boundary.condition', 'clampedBottom', 'domain.a', mdlC.L, 'domain.b', mdlC.H, ...
    'design.volumeFraction', 0.3, 'design.passiveSolid', pasC);
mC = model2D(olh.config.toLegacy(cfgC)); g = @(p) olh.config.getPath(cfgC, p);
stC = struct('model', g('material.stiffness.model'), 'p', g('material.stiffness.p'), ...
    'linearBelow', g('material.stiffness.linearBelow'));
rng(5); r = 0.2 + 0.6*rand(mC.nele, 1); r(pasC) = 1;
[K, M] = assemble2D(mC, r, stC, g('material.mass')); [~, Phi, lam] = eigSolve(K, M, 2, 'eigs');
G = genGrad(mC, r, stC, g('material.mass'), Phi, lam(1), 1); G = G(:,1,1);
act = setdiff((1:mC.nele)', pasC); smp = act(randperm(numel(act), 6)); h = 1e-6; err = zeros(6, 1);
for i = 1:6
    rp = r; rp(smp(i)) = rp(smp(i)) + h; rm = r; rm(smp(i)) = rm(smp(i)) - h;
    [K, M] = assemble2D(mC, rp, stC, g('material.mass')); [~, ~, lp] = eigSolve(K, M, 2, 'eigs');
    [K, M] = assemble2D(mC, rm, stC, g('material.mass')); [~, ~, lm] = eigSolve(K, M, 2, 'eigs');
    fd = (lp(1) - lm(1)) / (2*h); err(i) = abs(fd - G(smp(i))) / max(abs(fd), 1e-12);
end
fprintf('[test D] FD of lambda_1 (base clamp, 6 active elements): max rel. error %.2e\n', max(err));
end

function buildingStart(repoRoot)
% Reference frequencies of the building gains (Tables 5-6): the actual starting
% design (passive frame solid, uniform density elsewhere, total volume V_f) and,
% for comparison, x = V_f everywhere (what the submitted 19.84 / 90.93 rad/s are).
mdl = buildModel(repoRoot, 'building', 80, 240, []);
ev = mdl; ev.Emin = 1e-6 * mdl.E0; ev.rho_min = 1e-6; f = mdl.free;
[~, Ms] = assemble(ev, ones(mdl.nEl, 1));
designs = {initialDesign(mdl), mdl.volfrac * ones(mdl.nEl, 1)};
names = {'start (frame solid, active uniform)', 'x = V_f everywhere'};
for k = 1:2
    x = designs{k};
    [K, M] = assemble(ev, x);
    [V, D] = eigs(K(f,f), M(f,f), 6, 'smallestabs', struct('tol', 1e-12, 'v0', startVec(numel(f))));
    [lam, ix] = sort(real(diag(D))); V = V(:, ix);
    fprintf('[start] %s: mean x = %.4f, active x = %.4f\n', names{k}, mean(x), x(mdl.act(1)));
    for j = 1:6
        m1 = mac(V(:,j), mdl.phi0(f,1), M(f,f)); m2 = mac(V(:,j), mdl.phi0(f,2), M(f,f));
        fprintf('   mode %d: omega = %.4f rad/s, MAC_M with phi1(0) = %.3f, with phi2(0) = %.3f\n', j, sqrt(lam(j)), m1, m2);
    end
end
fprintf('[start] solid domain: %.4f / %.4f rad/s\n', mdl.omega0(1), mdl.omega0(2));
end

function equalityAblation(repoRoot, outDir)
% Audit 2.2: full derivative + MMA with the volume constraint as an EQUALITY
% (two inequalities), next to the C4 variants (inequality), alpha = 1.
for c = {'ss', 'clamped', 'building'}
    switch c{1}
        case 'ss', mdl = buildModel(repoRoot, 'ss', 400, 50, 0.5);
        case 'clamped', mdl = buildModel(repoRoot, 'clamped', 400, 50, []);
        case 'building', mdl = buildModel(repoRoot, 'building', 80, 240, []);
    end
    ev = mdl; ev.Emin = 1e-6 * mdl.E0; ev.rho_min = 1e-6;
    for v = {'full_MMA_equality', 'omitted_MMA_equality'}
        t0 = tic; r = optimizeMMA(mdl, [1; 0], strcmp(v{1}, 'full_MMA_equality'), true); tt = toc(t0);
        om = structuralOmega(ev, r.x);
        fprintf('[C4eq] %-8s %-22s it=%4d (%s) %.1f s c=%.6e vol=%.4f grey=%.3f omega1/2 = %.3f / %.3f\n', ...
            c{1}, v{1}, r.iter, r.stop, tt, r.c, mean(r.x), 4*mean(r.x.*(1-r.x)), om(1), om(2));
        S.(c{1}).(v{1}) = struct('x', r.x, 'om', om, 'iter', r.iter, 'c', r.c, 'hist', r.hist);
    end
    saveDesignPng(mdl, {S.(c{1}).full_MMA_equality.x, S.(c{1}).omitted_MMA_equality.x}, ...
        {'full, MMA, V = V_f', 'omitted, MMA, V = V_f'}, fullfile(outDir, ['c4eq_' c{1} '.png']));
end
save(fullfile(outDir, 'c4eq.mat'), 'S');
end

function spectralHistory(repoRoot, outDir)
% Audit 2.3: modal mass m_v = v'M(x)v, lambda_hat = s^2 m_v / c (optimization
% model) and the structural omega1 (common evaluator) at every iterate of the
% proposed method, alpha = 1.
H = struct();
for c = {'ss', 'clamped', 'building'}
    switch c{1}
        case 'ss', mdl = buildModel(repoRoot, 'ss', 400, 50, 0.5);
        case 'clamped', mdl = buildModel(repoRoot, 'clamped', 400, 50, []);
        case 'building', mdl = buildModel(repoRoot, 'building', 80, 240, []);
    end
    ev = mdl; ev.Emin = 1e-6 * mdl.E0; ev.rho_min = 1e-6;
    r = optimize(mdl, [1; 0], false, true);
    n = numel(r.xs); f = mdl.free; s = mdl.omega0(1)^2; v = mdl.phi0(:, 1);
    mv = NaN(n, 1); lh = NaN(n, 1); om = NaN(n, 1);
    for k = 1:n
        x = r.xs{k}; [K, M] = assemble(mdl, x);
        F = s * (M * v); u = zeros(mdl.ndof, 1); u(f) = K(f,f) \ F(f);
        cc = F(f)' * u(f); mv(k) = v(f)' * M(f,f) * v(f); lh(k) = s^2 * mv(k) / cc;
        o = structuralOmega(ev, x); om(k) = o(1);
    end
    inc = diff(sqrt(lh)) .* diff(om);
    fprintf(['[hist] %-8s %d iterates: m_v %.4f -> %.4f (%+.1f%%), sqrt(lambda_hat) %.2f -> %.2f, omega1 %.2f -> %.2f; ' ...
        'max |sqrt(lh)-omega1|/omega1 over iterates %.2f%% (last 10: %.2f%%); steps with opposite signs of d sqrt(lh) and d omega1: %d of %d\n'], ...
        c{1}, n, mv(1), mv(end), 100*(mv(end)/mv(1)-1), sqrt(lh(1)), sqrt(lh(end)), om(1), om(end), ...
        100*max(abs(sqrt(lh)-om)./om), 100*max(abs(sqrt(lh(end-9:end))-om(end-9:end))./om(end-9:end)), sum(inc < 0), numel(inc));
    H.(c{1}) = struct('mv', mv, 'lh', lh, 'om', om);
end
save(fullfile(outDir, 'c11_history.mat'), 'H');
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1200 350]);
nm = {'ss', 'clamped', 'building'}; tt = {'simply supported beam', 'clamped beam', 'building'};
for k = 1:3
    subplot(1, 3, k); h = H.(nm{k});
    yyaxis left; plot(sqrt(h.lh), '-', 'LineWidth', 1.2); hold on; plot(h.om, '--', 'LineWidth', 1.2);
    ylabel('rad/s'); yyaxis right; plot(h.mv, ':', 'LineWidth', 1.2); ylabel('m_v');
    xlabel('iteration'); title(tt{k}); legend({'\lambda_hat^{1/2}', '\omega_1', 'm_v'}, 'Location', 'southeast'); grid on;
end
exportgraphics(fig, fullfile(outDir, 'c11_history.png'), 'Resolution', 200); close(fig);
end

function v = mac(a, b, M)
v = (a' * M * b)^2 / ((a' * M * a) * (b' * M * b));
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

function res = optimize(mdl, w, useLoadSens, keepHistory)
% Standalone copy of the Proposed loop of topopt_freq (OC, ft = 0).
if nargin < 4, keepHistory = false; end
x = initialDesign(mdl); xs = {};
g = 0; maxIt = 2000; W = 10; tol = 1e-2;
d = NaN(maxIt, 1); gr = NaN(maxIt, 1); cH = NaN(maxIt, 1);
xPrev = []; pas = [mdl.pasS; mdl.pasV];
res.stop = 'max_iterations';
for it = 1:maxIt
    [c, gK, gF] = objectiveAndGradients(mdl, x, w);
    if keepHistory, xs{end+1} = x; end %#ok<AGROW>
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
if keepHistory, xs{end+1} = x; res.xs = xs; end
end

function res = optimizeMMA(mdl, w, useLoadSens, eqVol)
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
    fval = sum(x) / vScale - 1; dfv = dv(act)' / vScale;
    [xa, ~, ~, ~, ~, ~, ~, ~, ~, low, upp] = mmasub(1, n, it, x(act), zeros(n, 1), ones(n, 1), ...
        xold1, xold2, c / c0, dc(act) / c0, fval, dfv, low, upp, ...
        1, 0, 1e3, 1);
    if nargin >= 4 && eqVol
        % Equality V = V_f: two opposite MMA inequalities admit only the current
        % point, so the MMA step (inequality) is followed by an exact projection
        % onto the volume, a uniform shift within the move box (bisection), as
        % the MMA path of topopt_freq does.
        lo = max(0, x(act) - move); hi = min(1, x(act) + move);
        Vt = mdl.volfrac * mdl.nEl - sum(x) + sum(x(act));     % target active volume
        l1 = -1; l2 = 1;
        for kb = 1:100
            lm = 0.5*(l1 + l2);
            if sum(min(hi, max(lo, xa + lm))) > Vt, l2 = lm; else, l1 = lm; end
        end
        xa = min(hi, max(lo, xa + 0.5*(l1 + l2)));
    end
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

