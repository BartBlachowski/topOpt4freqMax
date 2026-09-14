function mig_run_suite(kind, outMat)
%MIG_RUN_SUITE  Run one test suite of TEST_REPORT.md and save its failure count.
%   kind:
%     an OlhoffCurrent test name ('test_preset_identity', ...), optionally
%       'name:arg' for a single char argument ('test_named_preset_reproduction:pedersen')
%     'upstream_suites'  the six root-independent upstream architecture/tests
%       suites (253069 snapshot), run AGAINST the migrated OlhoffCurrent +impl
%       under its own fail-closed gate
%     'upstream_suites_snapshot'  the same six suites against the snapshot itself
P = mig_paths();
maxNumCompThreads(1);
R = struct('kind', kind, 'when', char(datetime('now')), 'matlab', version, 'suites', struct('name', {}, 'nFail', {}));
switch kind
    case {'upstream_suites', 'upstream_suites_snapshot'}
        if strcmp(kind, 'upstream_suites')
            restoredefaultpath; addpath(P.scripts); addpath(P.oc);
            guard = olhoffcurrent_paths(); %#ok<NASGU>
            impl = fullfile(P.oc, '+impl');
            assert(strncmp(which('olhoffSolve'), impl, numel(impl)) && ...
                   strncmp(which('olh.move.limit'), impl, numel(impl)), 'mig:suite', 'target not dispatched');
            R.tree = olhoffcurrent_source_manifest('Verify', true).treeHash;
        else
            mig_use_snapshot(P.up, false);
            R.tree = 'snapshot 253069';
        end
        addpath(fullfile(P.up, 'architecture', 'tests'), '-end');
        addpath(fullfile(P.up, 'architecture', 'anchors', 'code'), '-end');
        R.solver = which('olhoffSolve'); R.limit = which('olh.move.limit');
        for s = {'test_config', 'test_mass', 'test_modules', 'test_preset_resolution_unchanged', ...
                 'test_stage_exhaustion', 'test_outer_timing'}
            n = feval(s{1});
            R.suites(end+1) = struct('name', s{1}, 'nFail', n);
            fprintf('SUITE %-36s failures=%d\n', s{1}, n);
        end
    case 'gates'
        % run ALONE: test_source_integrity and test_currentness perturb +impl and the
        % manifest temporarily (and restore them); nothing else may run concurrently
        restoredefaultpath; addpath(P.scripts); addpath(P.oc); addpath(fullfile(P.oc, 'tests'));
        for s = {'test_path_isolation', 'test_currentness', 'test_source_integrity', ...
                 'test_evidence_retention', 'test_finalization_gate'}
            restoredefaultpath; addpath(P.scripts); addpath(P.oc); addpath(fullfile(P.oc, 'tests'));
            n = feval(s{1});
            R.suites(end+1) = struct('name', s{1}, 'nFail', n);
            fprintf('SUITE %-36s failures=%d\n', s{1}, n);
        end
        R.manifestAfter = olhoffcurrent_source_manifest('Verify', true).ok;
    otherwise
        restoredefaultpath; addpath(P.scripts); addpath(P.oc); addpath(fullfile(P.oc, 'tests'));
        parts = strsplit(kind, ':');
        if numel(parts) == 2, n = feval(parts{1}, parts{2}); else, n = feval(parts{1}); end
        R.suites(end+1) = struct('name', kind, 'nFail', n);
        fprintf('SUITE %-36s failures=%d\n', kind, n);
end
R.nFail = sum([R.suites.nFail]);
save(outMat, 'R');
end
