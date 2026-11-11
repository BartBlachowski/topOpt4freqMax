function T = c7_matched_benchmark(meshes, nRep, methods, settings, outDir)
%C7_MATCHED_BENCHMARK  AppliSci revision C7: simply supported beam benchmark at
%   native and at matched settings, with a repeated-timing protocol.
%
%   T = c7_matched_benchmark()                         160x20, 400x50, 800x100, 3 reps
%   T = c7_matched_benchmark([160 20], 1)              quick check
%   T = c7_matched_benchmark(meshes, nRep, {'proposed','yuksel'}, {'matched'})
%
%   settings
%     native   the Table 1 settings of the paper: Proposed r = 2 el, move 0.2,
%              SIMP p = 3, E_min = 1e-9 E0, rho_min = 1e-9; Yuksel r = 2.5 el,
%              move 0.1, x^6 mass below 0.1; Du-Olhoff production preset
%     matched  every method at the Du-Olhoff production settings:
%              sensitivity filter R = 0.06 m (physical), move limit 0.1 (the
%              ceiling of the Du-Olhoff adaptive box), Pedersen (2000)
%              stiffness (x^3 above 0.1, linear below) with linear mass,
%              floor E = 1e-5 E0, rho = 1e-3 rho0 (= Du-Olhoff at rho_min = 1e-3),
%              uniform start x = 0.5.  Du-Olhoff is unchanged: its native
%              settings ARE the matched ones, so it is run once per repetition
%              and reported under both labels.
%   All runs: stagnation_x stop (W = 10, eps_x = eps_M = 1e-2), single thread,
%   one discarded warm-up per method, nRep timed repetitions, median and range.
%
%   Quality: lowest structural omega1/omega2 of the final design under
%     (a) the paper's common evaluator (SIMP p = 3, E_min = 1e-6 E0, linear mass
%         rho_min = 1e-6, structural_mode_mask)
%     (b) the matched model (Pedersen + linear mass, floor 1e-3), raw spectrum
%   both outside all timing.
%
%   Results: results/c7/c7_<tag>.csv (one row per run), c7_<tag>_summary.csv
%   (median/min/max per method x setting x mesh), designs in c7_<tag>.mat.

here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));
perf = fullfile(repoRoot, 'examples', 'Performance');
addpath(perf, fullfile(perf, 'conference_bench'), fullfile(perf, 'benchmark_profile'), ...
    fullfile(repoRoot, 'tools', 'Matlab'), fullfile(repoRoot, 'analysis', 'Olhoff'));
olhoffcurrent_scrub_forbidden_paths(repoRoot);
if nargin < 1 || isempty(meshes), meshes = [160 20; 400 50; 800 100]; end
if nargin < 2 || isempty(nRep), nRep = 3; end
if nargin < 3 || isempty(methods), methods = {'proposed', 'yuksel', 'olhoff'}; end
if nargin < 4 || isempty(settings), settings = {'native', 'matched'}; end
if nargin < 5 || isempty(outDir), outDir = fullfile(here, 'results', 'c7'); end
if ~exist(outDir, 'dir'), mkdir(outDir); end
tag = sprintf('%s_%s_%s_rep%d', strjoin(methods, '-'), strjoin(settings, '-'), ...
    strjoin(arrayfun(@(i) sprintf('%dx%d', meshes(i, 1), meshes(i, 2)), 1:size(meshes, 1), ...
    'UniformOutput', false), '-'), nRep);
logFile = fullfile(outDir, ['c7_' tag '.txt']);
if exist(logFile, 'file'), delete(logFile); end
diary(logFile); cleanupDiary = onCleanup(@() diary('off'));
maxNumCompThreads(1);
fprintf('C7 matched benchmark %s on %s, MATLAB %s\n', datestr(now, 31), ...
    strtrim(evalc('system(''hostname'');')), version);

stop = struct('criterion', 'stagnation_x', 'window', 10, 'graynessTol', 1e-2, ...
    'relativeChangeTol', 1e-2);
stopOlh = stop; stopOlh.maxOuter = 1000;
stopYuk = stop; stopYuk.maxIters = 5000;

% ---- warm-up (discarded) ------------------------------------------------
for im = 1:numel(methods)
    for is = 1:numel(settings)
        if strcmp(methods{im}, 'olhoff') && is > 1, continue; end
        mc = methodConfig(methods{im}, settings{is}, 48, 6, stop, stopOlh, stopYuk);
        w = confbench_run_case(methods{im}, mc, struct('max_outer_override', 5, 'warmup', true));
        fprintf('warm-up %s/%s: %s\n', methods{im}, settings{is}, w.status);
    end
end

rows = {}; designs = struct('method', {}, 'setting', {}, 'mesh', {}, 'x', {});
for ir = 1:size(meshes, 1)
    nelx = meshes(ir, 1); nely = meshes(ir, 2);
    for im = 1:numel(methods)
        m = methods{im};
        for is = 1:numel(settings)
            s = settings{is};
            if strcmp(m, 'olhoff') && is > 1, continue; end   % native == matched
            mc = methodConfig(m, s, nelx, nely, stop, stopOlh, stopYuk);
            xRef = [];
            for rep = 1:nRep
                rec = confbench_run_case(m, mc, struct('label', sprintf('C7 %s %s rep%d', m, s, rep)));
                t = rec.times;
                c = rec.counts;
                x = double(rec.x(:));
                if rep == 1
                    xRef = x;
                    [qa, qb] = evaluate(x, nelx, nely);
                    designs(end+1) = struct('method', m, 'setting', s, 'mesh', [nelx nely], 'x', x); %#ok<AGROW>
                end
                same = isequal(x, xRef);
                lbl = s; if strcmp(m, 'olhoff'), lbl = 'native=matched'; end
                fprintf(['%s %dx%d %-14s rep %d: %s  count1=%g count2=%g  T1=%.2f T2=%.2f stage=%.2f ' ...
                    'total=%.2f s  omega1 common=%.2f matched=%.2f  identical design=%d\n'], ...
                    m, nelx, nely, lbl, rep, rec.status, getf(c, 'count1'), getf(c, 'count2'), ...
                    getf(t, 'time1'), getf(t, 'time2'), getf(t, 'stage_time_s'), ...
                    getf(t, 'total_wall_time_s'), qa(1), qb(1), same);
                rows(end+1, :) = {m, lbl, sprintf('%dx%d', nelx, nely), nelx*nely, rep, rec.status, ...
                    getf(c, 'count1'), getf(c, 'count2'), getf(t, 'time1'), getf(t, 'time2'), ...
                    getf(t, 'stage_time_s'), getf(t, 'total_wall_time_s'), getf(t, 'eigen_time_s'), ...
                    getf(t, 'gradient_time_s'), qa(1), qa(2), qb(1), qb(2), mean(x), ...
                    4*mean(x.*(1 - x)), same}; %#ok<AGROW>
            end
            save(fullfile(outDir, ['c7_' tag '.mat']), 'designs', 'rows');
        end
    end
end
T = cell2table(rows, 'VariableNames', {'method', 'setting', 'mesh', 'Ne', 'rep', 'status', ...
    'count1', 'count2', 'time1_s', 'time2_s', 'stage_s', 'total_s', 'eigen_s', 'gradient_s', ...
    'omega1_common', 'omega2_common', 'omega1_matched', 'omega2_matched', 'volume', 'grey', ...
    'identical_design'});
writetable(T, fullfile(outDir, ['c7_' tag '.csv']));
G = groupsummary(T, {'method', 'setting', 'mesh', 'Ne'}, {'median', 'min', 'max'}, ...
    {'total_s', 'stage_s', 'count1', 'count2', 'omega1_common', 'omega1_matched'});
G = sortrows(G, {'Ne', 'setting', 'method'});
disp(G);
writetable(G, fullfile(outDir, ['c7_' tag '_summary.csv']));
end

% ======================================================================
function mc = methodConfig(m, s, nelx, nely, stop, stopOlh, stopYuk)
switch m
    case 'olhoff',   mc = confbench_method_config(m, nelx, nely, '', stopOlh);
    case 'yuksel',   mc = confbench_method_config(m, nelx, nely, '', stopYuk);
    otherwise,       mc = confbench_method_config(m, nelx, nely, '', stop);
end
if strcmp(m, 'olhoff') || strcmp(s, 'native'), return; end
assert(strcmp(s, 'matched'), 'c7:setting', 'unknown setting %s', s);
mc.optimization.filter.radius = 0.06;
mc.optimization.filter.radius_units = 'physical';
mc.optimization.move_limit = 0.1;
mc.optimization.stiffness_model = 'pedersen';
mc.optimization.pedersen_threshold = 0.1;
mc.void_material.E_min_ratio = 1e-5;     % = Pedersen g(1e-3) = 1e-3 * 0.1^2
mc.void_material.rho_min = 1e-3;
if strcmp(m, 'yuksel'), mc.optimization.yuksel.mass_cut = 0; end   % linear mass
mc.meta.profile_id = [mc.meta.profile_id '_C7matched'];
end

function [qa, qb] = evaluate(x, nelx, nely)
%EVALUATE  (a) paper's common evaluator, (b) matched Pedersen model (raw).
guard = olhoffcurrent_paths(); %#ok<NASGU>
cfg = olhoffcurrent_config(nelx, nely, 'Preset', olhoffcurrent_production_preset().name);
mdl = model2D(olh.config.toLegacy(cfg));
x = min(1, max(0, x(:)));
% (a) SIMP p = 3, E_min = 1e-6 E0, linear mass rho_min = 1e-6, structural modes
E0 = 1e7; rho0 = 1;
Ee = 1e-6*E0 + x.^3*(E0 - 1e-6*E0);
re = 1e-6*rho0 + x*(rho0 - 1e-6*rho0);
sK = mdl.K0(:)/E0*Ee'; sM = mdl.M0(:)/rho0*re';
K = sparse(mdl.iK, mdl.jK, sK(:), mdl.ndof, mdl.ndof); K = (K + K')/2;
M = sparse(mdl.iK, mdl.jK, sM(:), mdl.ndof, mdl.ndof); M = (M + M')/2;
f = mdl.free;
qa = lowest_structural_modes(K(f, f), M(f, f), f, mdl.ndof, mdl.edofMat, ...
    mdl.K0/E0, mdl.M0/rho0, Ee, re, x, 2, 6);
% (b) matched model: Pedersen stiffness + linear mass at rho = max(x, 1e-3)
st = struct('model', 'pedersen', 'p', 3, 'linearBelow', 0.1);
[K, M] = assemble2D(mdl, max(x, 1e-3), st, struct('model', 'eq2', 'q', 1, ...
    'lowDensityExponent', 6, 'cutoff', 0.1));
qb = eigSolve(K, M, 2, 'eigs');
end

function v = getf(s, f)
if isstruct(s) && isfield(s, f) && ~isempty(s.(f)), v = double(s.(f)); else, v = NaN; end
end
