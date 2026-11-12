function test_ports_against_research_code(researchRepo)
%TEST_PORTS_AGAINST_RESEARCH_CODE  Checks that the compact solvers of this branch
%   reproduce the research code used for the paper (tag applsci-revision-1).
%   Needs a checkout of that tag:  test_ports_against_research_code('/path/to/topOpt4freqMax')
%   Not needed to reproduce the paper; it documents that the compact code is the same method.
here = fileparts(fileparts(mfilename('fullpath')));
run(fullfile(here, 'setup_paths.m'));
addpath(fullfile(researchRepo, 'analysis', 'Olhoff'));
fprintf('--- Du-Olhoff: compact port vs analysis/Olhoff (production preset, stagnation_x) ---\n');
cases = {problem_ss_beam(160, 20), problem_clamped_beam(160, 20), problem_building(40, 120)};
for k = 1:numel(cases)
    prob = cases{k};
    R = 0.06;  if strcmp(prob.name, 'building'), R = 3*prob.L/prob.nelx; end
    a = du_olhoff(prob, struct('filterRadius', R));
    b = researchDuOlhoff(researchRepo, prob, R);
    run(fullfile(here, 'setup_paths.m'));     % the research path guard restores its own path
    fprintf('%-13s outer %d / %d, max|dx| = %.3g, omega1 %.10g / %.10g\n', prob.name, ...
        a.nOuter, b.nOuter, max(abs(a.xNative - b.rho)), a.omega(1), b.omega(1));
end
% proposed and Yuksel-Yilmaz: the 400x50 designs of the research code (Table 2 settings),
% stored by examples/Revision_applsci/c7_tolerance_check.m in the research repository
f = fullfile(researchRepo, 'examples', 'Revision_applsci', 'results', 'c7tol', 'c7tol_400x50.mat');
if exist(f, 'file')
    S = load(f, 'recs');  prob = problem_ss_beam(400, 50);
    for k = 1:numel(S.recs)
        r = S.recs{k};
        if abs(r.tol - 1e-2) > 1e-12 || strcmp(r.method, 'olhoff'), continue; end
        if strcmp(r.method, 'proposed')
            a = proposed_freq(prob);  it = a.iterations;
        else
            a = yuksel_freq(prob);  it = a.iterations;
        end
        xr = r.x(:);  xf = reshape(flipud(reshape(xr, 50, 400)), [], 1);
        fprintf('%-9s iterations %s / research %g+%g, max|dx| = %.3g (as stored) / %.3g (flipped)\n', r.method, ...
            mat2str(it), r.counts.count1, r.counts.count2, max(abs(a.x - xr)), max(abs(a.x - xf)));
    end
end
end

function res = researchDuOlhoff(repo, prob, R)
% the research code refuses to run when another mmasub is visible: hide this branch's solvers
here = fileparts(fileparts(mfilename('fullpath')));
d = {fullfile(here,'solvers','du_olhoff'), fullfile(here,'solvers','proposed')};
rmpath(d{:});  restore = onCleanup(@() addpath(d{:}));
guard = olhoffcurrent_paths(); %#ok<NASGU>
pre = olhoffcurrent_preset(olhoffcurrent_production_preset().name);
bc = prob.support;
extra = {};
if strcmp(prob.name, 'building')
    m = false(prob.nelx*prob.nely,1); m(prob.passive) = true;
    m = flipud(reshape(m, prob.nely, prob.nelx));
    extra = {'domain.a', prob.L, 'domain.b', prob.H, 'design.passiveSolid', find(m(:))};
end
cfg = olh.config.resolve(pre.upstreamPreset, 'domain.mesh.nelx', prob.nelx, 'domain.mesh.nely', prob.nely, ...
    pre.overrides{:}, 'domain.boundary.condition', bc, 'design.volumeFraction', prob.volfrac, extra{:}, ...
    'filter.radiusPhysical', R, 'stop.rule', 'stagnation_x', 'stop.guards.settledMove', false, ...
    'stop.guards.boxInactiveFraction', 0, 'stop.guards.ladderExhausted', false, 'stop.guards.maxDesignChange', false, ...
    'stop.stagnation.window', 10, 'stop.stagnation.relativeChangeTolerance', 1e-2, ...
    'stop.stagnation.graynessTolerance', 1e-2, 'runtime.maxOuter', 1000, 'runtime.singleThread', true);
r = olhoffSolve(cfg);
res = struct('rho', r.rho(:), 'omega', r.omega, 'nOuter', numel(r.hist.N));
end
