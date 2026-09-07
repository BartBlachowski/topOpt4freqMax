function nFail = test_preset_equivalence(meshes)
%TEST_PRESET_EQUIVALENCE  The production preset must reproduce the frozen
%   conference realization.
%
%   nFail = TEST_PRESET_EQUIVALENCE()              160x20 only (about 2 min)
%   nFail = TEST_PRESET_EQUIVALENCE([160 20;320 40])  both promotion meshes
%
%   The reference is the SAVED conference campaign,
%   examples/Performance/conference_benchmark/campaign_9mesh_r2/benchmark_records.mat,
%   produced by analysis/OlhoffM4Reconstruction -- the realization that is
%   production today.  Existing evidence is reused rather than regenerated.
%
%   The comparison standard is BITWISE on the design vector: rho is compared
%   with isequal on doubles, not with a tolerance.  A tolerance here would
%   hide exactly the class of defect this test exists to catch.
%
%   Scientific equivalence is only asserted at meshes >= 160x20; the mesh list
%   is refused below that floor.

if nargin < 1 || isempty(meshes), meshes = [160 20]; end

here = fileparts(mfilename('fullpath'));
root = fileparts(here);
repo = fileparts(fileparts(root));
addpath(root);

recFile = fullfile(repo,'examples','Performance','conference_benchmark', ...
                   'campaign_9mesh_r2','benchmark_records.mat');
assert(exist(recFile,'file') == 2, 'test_preset_equivalence:NoReference', ...
    'Frozen conference records not found: %s', recFile);
R = load(recFile);
rec = R.records(strcmp({R.records.method_key},'olhoff'));

maxNumCompThreads(1);
nFail = 0;
fprintf('\n%s\nTEST_PRESET_EQUIVALENCE  (production preset vs frozen conference)\n%s\n', ...
    repmat('=',1,72), repmat('=',1,72));

for r = 1:size(meshes,1)
    nelx = meshes(r,1); nely = meshes(r,2);
    assert(nelx*nely >= 3200, 'test_preset_equivalence:MeshFloor', ...
        ['Scientific equivalence is not asserted below 160x20 (got %dx%d). ' ...
         'Cut the iteration cap, never the mesh.'], nelx, nely);

    j = find(arrayfun(@(s) isequal(s.mesh(:).', [nelx nely]), rec), 1);
    if isempty(j)
        fprintf('  [SKIP] %dx%d: no frozen conference record\n', nelx, nely); continue
    end
    ref = rec(j);

    guard = olhoffcurrent_paths(); %#ok<NASGU>
    cfg = olhoffcurrent_config(nelx, nely);
    res = olhoffSolve(cfg);

    got = struct('rho', double(res.rho(:)), 'omega1', double(res.omega(1)), ...
        'volume', mean(double(res.rho(:))), 'nOuter', numel(res.hist.N), ...
        'inner', sum(res.hist.nInner), 'converged', ...
        any(contains(res.log,'converged at outer')));
    want = struct('rho', double(ref.x(:)), 'omega1', double(ref.omega(1)), ...
        'volume', mean(double(ref.x(:))), 'nOuter', double(ref.counts.outer_iterations), ...
        'inner', double(ref.counts.inner_iterations_total), 'converged', ...
        strcmp(ref.status,'NATIVE_CONVERGED'));

    checks = { ...
        'design vector rho (BITWISE)', isequal(got.rho, want.rho); ...
        'first eigenfrequency (BITWISE)', isequal(got.omega1, want.omega1); ...
        'volume (BITWISE)', isequal(got.volume, want.volume); ...
        'outer iteration count', isequal(got.nOuter, want.nOuter); ...
        'inner iteration count', isequal(got.inner, want.inner); ...
        'convergence status', isequal(got.converged, want.converged)};

    fprintf('  --- %dx%d ---\n', nelx, nely);
    for k = 1:size(checks,1)
        if checks{k,2}, fprintf('    [PASS] %s\n', checks{k,1});
        else, fprintf('    [FAIL] %s\n', checks{k,1}); nFail = nFail + 1; end
    end
    fprintf('    outer %d/%d  inner %d/%d  omega1 %.17g / %.17g\n', ...
        got.nOuter, want.nOuter, got.inner, want.inner, got.omega1, want.omega1);
    if ~isequal(got.rho, want.rho)
        fprintf('    max|drho| = %.17g over %d elements\n', ...
            max(abs(got.rho - want.rho)), numel(got.rho));
    end
    clear guard
end

fprintf('%s\n  failures: %d\n\n', repmat('-',1,72), nFail);
end
