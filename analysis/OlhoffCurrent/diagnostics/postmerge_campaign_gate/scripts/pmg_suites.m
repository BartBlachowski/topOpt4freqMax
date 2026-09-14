function R = pmg_suites(repo, kind, outJson)
%PMG_SUITES  Run one block of the post-merge gate test suite in checkout REPO and
%   save failure counts (and per-study finalization verdicts) to OUTJSON.
%   Scientifically inert apart from the 160x20 solves the named suites contain.
%
%   kind:
%     'gates'       test_path_isolation, test_currentness, test_source_integrity,
%                   test_evidence_retention, test_finalization_gate   (run ALONE:
%                   two of them perturb +impl temporarily and restore it)
%     'presets'     test_preset_identity, test_pedersen_adaptive_units, test_cost_reporting
%     'solves160'   test_preset_equivalence, test_named_preset_reproduction('pedersen'),
%                   test_named_preset_reproduction('stageExhaustion')          (160x20)
%     'studies'     olhoffcurrent_finalization_gate over every diagnostics study
%     'selftest'    confbench_selftest (mechanics; sub-160x20 or no solve)
%     'upstream'    the six root-independent upstream suites (253069 git archive in
%                   UPDIR) against OlhoffCurrent +impl under its fail-closed gate
%     'harness'     confbench_preflight 160x20 (no solve) + Olhoff cap-3 smoke + export
oc = fullfile(repo, 'analysis', 'OlhoffCurrent');
R = struct('repo', repo, 'kind', kind, 'when', char(datetime('now')), 'matlab', version, ...
    'head', strtrim(local_sys(sprintf('git -C "%s" rev-parse HEAD', repo))), ...
    'suites', struct('name', {}, 'nFail', {}, 'error', {}));
restoredefaultpath;
addpath(fileparts(mfilename('fullpath')));
maxNumCompThreads(1);
switch kind
    case {'gates', 'presets', 'solves160'}
        addpath(oc); addpath(fullfile(oc, 'tests'));
        switch kind
            case 'gates',   L = {'test_path_isolation', 'test_currentness', 'test_source_integrity', ...
                                 'test_evidence_retention', 'test_finalization_gate'};
            case 'presets', L = {'test_preset_identity', 'test_pedersen_adaptive_units', 'test_cost_reporting'};
            case 'solves160', L = {'test_preset_equivalence', 'test_named_preset_reproduction:pedersen', ...
                                   'test_named_preset_reproduction:stageExhaustion'};
        end
        entry = path();
        for s = L
            path(entry);            % some suites reset the path (test_path_isolation)
            R = local_run(R, s{1});
        end
    case 'studies'
        addpath(oc);
        D = dir(fullfile(oc, 'diagnostics')); D = D([D.isdir] & ~startsWith({D.name}, '.'));
        R.studies = struct('name', {}, 'ok', {}, 'gates', {}, 'nMissing', {}, 'nMismatched', {}, ...
            'historical', {}, 'current', {}, 'nSuperseded', {});
        for i = 1:numel(D)
            st = olhoffcurrent_finalization_gate(fullfile(D(i).folder, D(i).name), 'Verbose', false, 'RepoRoot', repo);
            hs = 'n/a'; cs = 'n/a';
            if isfield(st, 'historicalSource'), hs = st.historicalSource.status; end
            if isfield(st, 'currentSource'),    cs = st.currentSource.status; end
            nsup = 0; if isfield(st, 'supersededSource'), nsup = numel(st.supersededSource); end
            R.studies(end+1) = struct('name', D(i).name, 'ok', st.ok, 'gates', st.gates, ...
                'nMissing', numel(st.missing), 'nMismatched', numel(st.mismatched), ...
                'historical', hs, 'current', cs, 'nSuperseded', nsup);
            fprintf('STUDY %-45s ok=%d hist=%s cur=%s\n', D(i).name, st.ok, hs, cs);
        end
        R.failing = {R.studies(~[R.studies.ok]).name};
    case 'selftest'
        local_harnessPath(repo);
        st = confbench_selftest([outJson(1:end-5) '_selftest_report.json']);
        R.selftest = st.tests;
        R.failed_ids = {st.tests(~[st.tests.pass]).id};
        R.suites(end+1) = struct('name', 'confbench_selftest', 'nFail', numel(R.failed_ids), 'error', '');
        for i = 1:numel(st.tests), fprintf('  [%d] %s %s\n', st.tests(i).pass, st.tests(i).id, st.tests(i).detail); end
    case 'upstream'
        upDir = getenv('UPDIR');
        addpath(oc); guard = olhoffcurrent_paths(); %#ok<NASGU>
        impl = fullfile(oc, '+impl');
        assert(strncmp(which('olhoffSolve'), impl, numel(impl)) && strncmp(which('olh.move.limit'), impl, numel(impl)), ...
            'pmg:suite', 'target not dispatched');
        R.tree = olhoffcurrent_source_manifest('Verify', true).treeHash;
        addpath(fullfile(upDir, 'architecture', 'tests'), '-end');
        addpath(fullfile(upDir, 'architecture', 'anchors', 'code'), '-end');
        R.solver = which('olhoffSolve'); R.limit = which('olh.move.limit');
        entry = path();
        for s = {'test_config', 'test_mass', 'test_modules', 'test_preset_resolution_unchanged', ...
                 'test_stage_exhaustion', 'test_outer_timing'}
            path(entry);
            R = local_run(R, s{1});
        end
    case 'harness'
        local_harnessPath(repo);
        R.harness = pmg_harness(repo);
        R.suites(end+1) = struct('name', 'harness', 'nFail', double(~R.harness.pass), 'error', '');
    otherwise
        error('pmg:kind', 'unknown kind %s', kind);
end
fid = fopen(outJson, 'w'); fwrite(fid, jsonencode(R, 'PrettyPrint', true)); fclose(fid);
for i = 1:numel(R.suites)
    fprintf('SUITE %-45s failures=%d %s\n', R.suites(i).name, R.suites(i).nFail, R.suites(i).error);
end
end

function R = local_run(R, spec)
parts = strsplit(spec, ':');
err = '';
try
    if numel(parts) == 2, n = feval(parts{1}, parts{2}); else, n = feval(parts{1}); end
catch ME
    n = NaN; err = [ME.identifier ': ' ME.message];
end
R.suites(end+1) = struct('name', spec, 'nFail', n, 'error', err);
fprintf('SUITE %-45s failures=%g %s\n', spec, n, err);
end

function local_harnessPath(repo)
scriptDir = fullfile(repo, 'examples', 'Performance');
addpath(scriptDir); addpath(fullfile(scriptDir, 'conference_bench'));
addpath(fullfile(repo, 'tools', 'Matlab'));
addpath(fullfile(repo, 'analysis', 'three_method_parametric_study'));
addpath(fullfile(repo, 'analysis', 'OlhoffCurrent'));
olhoffcurrent_scrub_forbidden_paths(repo);
end

function s = local_sys(cmd)
[~, s] = system(cmd);
end
