function T = c3_olhoff_reproduce(outDir, meshes, variants)
%C3_OLHOFF_REPRODUCE  AppliSci revision C3, part 2: the simply supported beam of
%   Du & Olhoff (2007), Fig. 3a (omega1 = 174.7 rad/s, bimodal, omega3 = 284.9
%   rad/s; initial uniform design 68.7 rad/s), with the comparator of the paper
%   and with variants closer to the published settings.
%
%   Variants (upstream preset + overrides, resolved with olh.config.resolve;
%   each with its own native stop rule and a 400-outer-iteration cap):
%     prod       duOlhoffAdaptivePedersen = production preset of the paper
%                (Pedersen stiffness, linear mass, adaptive box, p = 3)
%     prodCorner prod with pins at the bottom corners (as drawn in Fig. 2a)
%     prodRoller prod with the classical pin + roller (ux free at the right end)
%     eq4b       duOlhoffFrozenM4: SIMP p = 3 + eq. (4b) mass (the paper's own
%                low-density treatment), move ladder
%     pCont      pContinuationDecoupled: eq4b + p = 1 -> 2 -> 3 (sec. 2.1)
%
%   Reported: omega1..3 of the final design in the solver's own material
%   model, gap (omega2-omega1)/omega1, outer/inner iterations, time, grey
%   measure, volume.  Results: results/c3/c3_reproduce_*.txt/.csv/.mat;
%   the production 160x20 design is also saved as olhoff_prod_160x20.mat
%   (used by check E of c3_olhoff_checks).

here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));
addpath(fullfile(repoRoot, 'analysis', 'Olhoff'));
if nargin < 1 || isempty(outDir), outDir = fullfile(here, 'results', 'c3'); end
if nargin < 2 || isempty(meshes), meshes = [160 20; 320 40]; end
if nargin < 3 || isempty(variants), variants = {'prod', 'prodCorner', 'prodRoller', 'eq4b', 'pCont'}; end
if ~exist(outDir, 'dir'), mkdir(outDir); end
tag = sprintf('%s_%s', strjoin(variants, '-'), strjoin(arrayfun(@(i) sprintf('%dx%d', meshes(i, 1), meshes(i, 2)), ...
    1:size(meshes, 1), 'UniformOutput', false), '-'));
logFile = fullfile(outDir, ['c3_reproduce_' tag '.txt']);
if exist(logFile, 'file'), delete(logFile); end
diary(logFile); cleanupDiary = onCleanup(@() diary('off'));
guard = olhoffcurrent_paths(); %#ok<NASGU>

defs = struct( ...
    'prod',       {{'duOlhoffAdaptivePedersen', {}}}, ...
    'prodCorner', {{'duOlhoffAdaptivePedersen', {'domain.boundary.support', 'corner'}}}, ...
    'prodRoller', {{'duOlhoffAdaptivePedersen', {'domain.boundary.axialRestraint', 'oneEnd'}}}, ...
    'eq4b',       {{'duOlhoffFrozenM4', {}}}, ...
    'pCont',      {{'pContinuationDecoupled', {}}});

rows = {};
for im = 1:size(meshes, 1)
    nelx = meshes(im, 1); nely = meshes(im, 2);
    for iv = 1:numel(variants)
        v = variants{iv}; d = defs.(v);
        cfg = olh.config.resolve(d{1}, 'domain.mesh.nelx', nelx, 'domain.mesh.nely', nely, ...
            'runtime.maxOuter', 400, d{2}{:});
        fprintf('\n=== %s %dx%d (preset %s) ===\n', v, nelx, nely, d{1});
        t0 = tic;
        res = olhoffSolve(cfg);
        tw = toc(t0);
        h = res.hist;
        w = res.omega(:);
        x = res.rho(:);
        % initial omega1 of this model (rho = 0.5)
        flat = olh.config.toLegacy(cfg); mdl = model2D(flat);
        g = @(p) olh.config.getPath(cfg, p);
        st0 = struct('model', g('material.stiffness.model'), 'p', g('material.stiffness.p'), ...
                     'linearBelow', g('material.stiffness.linearBelow'));
        [K0, M0] = assemble2D(mdl, 0.5*ones(mdl.nele, 1), st0, g('material.mass'));
        w0 = eigSolve(K0, M0, 3, 'eigs');
        gap = (w(2) - w(1))/w(1);
        grey = 4*mean(x.*(1 - x));
        nOut = numel(h.N); nIn = sum(h.nInner);
        fprintf('%s %dx%d: omega0=%.2f  omega1..3 = %.2f %.2f %.2f  gap12=%.1f%%  outer=%d inner=%d  %.0fs  grey=%.3f vol=%.4f status=%s\n', ...
            v, nelx, nely, w0(1), w(1), w(2), w(3), 100*gap, nOut, nIn, tw, grey, mean(x), res.status);
        rows(end+1, :) = {v, sprintf('%dx%d', nelx, nely), w0(1), w(1), w(2), w(3), 100*gap, ...
            nOut, nIn, tw, grey, mean(x), char(res.status)}; %#ok<AGROW>
        S = struct('x', x, 'omega', w, 'omega0', w0, 'hist', h, 'cfg', cfg, 'status', res.status, ...
            'nelx', nelx, 'nely', nely, 'variant', v, 'time', tw);
        save(fullfile(outDir, sprintf('olhoff_%s_%dx%d.mat', v, nelx, nely)), '-struct', 'S');
        f = figure('Visible', 'off', 'Position', [100 100 900 180]);
        imagesc(1 - reshape(x, nely, nelx)); colormap gray; axis equal tight off; caxis([0 1]);
        title(sprintf('%s %dx%d: \\omega_1=%.1f, \\omega_2=%.1f, \\omega_3=%.1f rad/s', v, nelx, nely, w(1), w(2), w(3)));
        exportgraphics(f, fullfile(outDir, sprintf('olhoff_%s_%dx%d.png', v, nelx, nely)), 'Resolution', 150);
        close(f);
    end
end
T = cell2table(rows, 'VariableNames', {'variant', 'mesh', 'omega0', 'omega1', 'omega2', 'omega3', ...
    'gap12_pct', 'outer', 'inner', 'time_s', 'grey', 'volume', 'status'});
disp(T);
fprintf('Du & Olhoff (2007): omega0 = 68.7, omega1 = omega2 = 174.7, omega3 = 284.9 rad/s\n');
writetable(T, fullfile(outDir, ['c3_reproduce_' tag '.csv']));
end
