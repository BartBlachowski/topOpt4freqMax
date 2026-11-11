function T = c7_tolerance_check(nelx, nely, tols, methods, outDir)
%C7_TOLERANCE_CHECK  AppliSci revision (audit 2.6 / plan T6.5): the simply supported
%   beam with the native settings of Table 1 and the stagnation_x rule at two
%   tolerances (eps_x = eps_M = 1e-2 as in Table 1, and 1e-3), with the cost
%   decomposition every implementation reports.
%
%   T = c7_tolerance_check()                 400x50, tols [1e-2 1e-3], all three methods
%
%   Per run: status, counts, every timer of confbench_run_case (Du-Olhoff:
%   eigensolve, gradients, inner MMA, bookkeeping, overhead; Yuksel: stage 1/2;
%   proposed: reference eigenanalysis, static loop, overhead), volume, grey
%   measure, lowest structural omega1/omega2 under the common evaluator.
%   One discarded warm-up per method; one timed run per tolerance (single thread).
%   Results: results/c7tol/c7tol_<mesh>.txt/.csv/.mat
here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));
perf = fullfile(repoRoot, 'examples', 'Performance');
addpath(perf, fullfile(perf, 'conference_bench'), fullfile(perf, 'benchmark_profile'), ...
    fullfile(repoRoot, 'tools', 'Matlab'), fullfile(repoRoot, 'analysis', 'Olhoff'));
olhoffcurrent_scrub_forbidden_paths(repoRoot);
if nargin < 1 || isempty(nelx), nelx = 400; nely = 50; end
if nargin < 3 || isempty(tols), tols = [1e-2 1e-3]; end
if nargin < 4 || isempty(methods), methods = {'proposed', 'yuksel', 'olhoff'}; end
if nargin < 5 || isempty(outDir), outDir = fullfile(here, 'results', 'c7tol'); end
if ~exist(outDir, 'dir'), mkdir(outDir); end
tag = sprintf('%dx%d', nelx, nely);
logFile = fullfile(outDir, ['c7tol_' tag '.txt']);
if exist(logFile, 'file'), delete(logFile); end
diary(logFile); cleanupDiary = onCleanup(@() diary('off'));
maxNumCompThreads(1);
fprintf('C7 tolerance check %s on %s, MATLAB %s\n', datestr(now, 31), strtrim(evalc('system(''hostname'');')), version);
rows = {}; recs = {};
for im = 1:numel(methods)
    m = methods{im};
    st = struct('criterion', 'stagnation_x', 'window', 10, 'graynessTol', 1e-2, 'relativeChangeTol', 1e-2);
    mc = methodConfig(m, 'native', 48, 6, st, setfield(st, 'maxOuter', 1000), setfield(st, 'maxIters', 5000)); %#ok<SFLD>
    w = confbench_run_case(m, mc, struct('max_outer_override', 5, 'warmup', true));
    fprintf('warm-up %s: %s\n', m, w.status);
    for it = 1:numel(tols)
        tol = tols(it);
        st = struct('criterion', 'stagnation_x', 'window', 10, 'graynessTol', tol, 'relativeChangeTol', tol);
        mc = methodConfig(m, 'native', nelx, nely, st, setfield(st, 'maxOuter', 1000), setfield(st, 'maxIters', 5000)); %#ok<SFLD>
        rec = confbench_run_case(m, mc, struct('label', sprintf('tol %g %s', tol, m)));
        x = double(rec.x(:));
        qa = evaluate(x, nelx, nely);
        t = rec.times; c = rec.counts;
        fprintf('\n%s tol=%g: %s count1=%g count2=%g total=%.2f s vol=%.4f grey=%.3f omega1/2 common = %.2f / %.2f\n', ...
            m, tol, rec.status, getf(c, 'count1'), getf(c, 'count2'), getf(t, 'total_wall_time_s'), mean(x), ...
            4*mean(x.*(1-x)), qa(1), qa(2));
        disp(t);
        rows(end+1, :) = {m, tol, rec.status, getf(c, 'count1'), getf(c, 'count2'), getf(t, 'time1'), getf(t, 'time2'), ...
            getf(t, 'eigen_time_s'), getf(t, 'gradient_time_s'), getf(t, 'inner_time_total_s'), ...
            getf(t, 'outer_bookkeeping_time_s'), getf(t, 'overhead_time_s'), getf(t, 'total_wall_time_s'), ...
            mean(x), 4*mean(x.*(1-x)), qa(1), qa(2)}; %#ok<AGROW>
        recs{end+1} = struct('method', m, 'tol', tol, 'x', x, 'times', t, 'counts', c, 'status', rec.status); %#ok<AGROW>
        save(fullfile(outDir, ['c7tol_' tag '.mat']), 'rows', 'recs');
    end
end
T = cell2table(rows, 'VariableNames', {'method', 'tol', 'status', 'count1', 'count2', 'time1_s', 'time2_s', ...
    'eigen_s', 'gradient_s', 'inner_mma_s', 'bookkeeping_s', 'overhead_s', 'total_s', 'volume', 'grey', ...
    'omega1_common', 'omega2_common'});
writetable(T, fullfile(outDir, ['c7tol_' tag '.csv']));
disp(T);
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
