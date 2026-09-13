function R = pgh_run(kind, outJson)
%PGH_RUN  One logged block of the provenance-gate hardening audit (PREREGISTRATION.md).
%   kind 'probes'  : gate_provenance_probes on this checkout -> probe table (JSON)
%        'tests'   : the worktree suites of PREREGISTRATION.md section 5
%        'real'    : R1 (two_branch_controller_validation) and R2 (every study)
%        'discrimination' : the same probes against the two EARLIER gate versions
%                   (9b30ec4 from git; adf86a3 from selfreview/*.m.txt), to show
%                   each probe separates a defective gate from the repaired one
%   Paths are derived from this file, so the script runs in whichever checkout
%   holds it.  Scientifically inert.
here = fileparts(mfilename('fullpath'));
study = fileparts(here);
oc = fileparts(fileparts(study));
repo = fileparts(fileparts(oc));
restoredefaultpath;
addpath(here); addpath(oc); addpath(fullfile(oc, 'tests'));
setenv('GIT_PAGER', 'cat');
[~, head] = system(sprintf('git --no-pager -C "%s" rev-parse HEAD', repo));
R = struct('kind', kind, 'repo', repo, 'head', strtrim(head), 'when', char(datetime('now')), ...
           'matlab', version, 'gate', which('olhoffcurrent_finalization_gate'));
switch kind
    case 'probes'
        [T, meta] = gate_provenance_probes(repo);
        R.probes = T; R.meta = meta;
        R.allAsExpected = all([T.pass]) && numel(T) == 39;
        fprintf('PROBES %d/%d as expected\n', sum([T.pass]), numel(T));
    case 'tests'
        entry = path();
        R.suites = struct('name', {}, 'nFail', {}, 'error', {});
        for s = {'test_path_isolation', 'test_currentness', 'test_source_integrity', ...
                 'test_evidence_retention', 'test_preset_identity', 'test_finalization_gate'}
            path(entry);
            err = '';
            try, n = feval(s{1}); catch ME, n = NaN; err = [ME.identifier ': ' ME.message]; end
            R.suites(end+1) = struct('name', s{1}, 'nFail', n, 'error', err);
            fprintf('SUITE %-28s failures=%g %s\n', s{1}, n, err);
        end
    case 'real'
        S = fullfile(oc, 'diagnostics', 'two_branch_controller_validation');
        st = olhoffcurrent_finalization_gate(S, 'Verbose', true, 'RepoRoot', repo);
        R.R1 = struct('ok', st.ok, 'gates', st.gates, 'historicalSource', st.historicalSource, ...
            'currentSource', rmfield(st.currentSource, 'reasons'), 'currentReasons', {st.currentSource.reasons}, ...
            'sourceLines', st.sourceLines);
        D = dir(fullfile(oc, 'diagnostics')); D = D([D.isdir] & ~startsWith({D.name}, '.'));
        R.R2 = struct('name', {}, 'ok', {}, 'G', {}, 'historical', {}, 'current', {});
        for i = 1:numel(D)
            s = olhoffcurrent_finalization_gate(fullfile(D(i).folder, D(i).name), 'Verbose', false, 'RepoRoot', repo);
            g = s.gates;
            R.R2(end+1) = struct('name', D(i).name, 'ok', s.ok, ...
                'G', sprintf('%d%d%d%d%d%d', g.G1, g.G2, g.G3, g.G4, g.G5, g.G6), ...
                'historical', s.historicalSource.status, 'current', s.currentSource.status);
            fprintf('STUDY %-45s ok=%d G=%s %s %s\n', D(i).name, s.ok, R.R2(end).G, ...
                s.historicalSource.status, s.currentSource.status);
        end
        R.failing = {R.R2(~[R.R2.ok]).name};
    case 'discrimination'
        V = {'9b30ec4', ''; 'adf86a3', fullfile(study, 'selfreview', 'olhoffcurrent_finalization_gate.adf86a3.m.txt')};
        R.versions = struct('version', {}, 'gateSha256', {}, 'probes', {}, 'nAsExpected', {});
        for v = 1:size(V, 1)
            d = [tempname() '_gate_' V{v, 1}]; mkdir(d);
            g = fullfile(d, 'olhoffcurrent_finalization_gate.m');
            if isempty(V{v, 2})
                system(sprintf('git --no-pager -C "%s" show %s:analysis/OlhoffCurrent/olhoffcurrent_finalization_gate.m > "%s"', repo, V{v, 1}, g));
            else
                copyfile(V{v, 2}, g);
            end
            addpath(d);
            assert(strcmp(which('olhoffcurrent_finalization_gate'), g), 'pgh:dispatch', 'old gate not dispatched');
            T = gate_provenance_probes(repo);
            rmpath(d); rmdir(d, 's');
            R.versions(end+1) = struct('version', V{v, 1}, 'gateSha256', '', 'probes', T, 'nAsExpected', sum([T.pass]));
            fprintf('DISCRIMINATION %s: %d/%d probes as expected by the REPAIRED rule\n', V{v, 1}, sum([T.pass]), numel(T));
        end
        assert(contains(which('olhoffcurrent_finalization_gate'), fullfile(oc, 'olhoffcurrent_finalization_gate.m')));
    otherwise
        error('pgh:kind', 'unknown kind %s', kind);
end
fid = fopen(outJson, 'w'); fwrite(fid, jsonencode(R, 'PrettyPrint', true)); fclose(fid);
end
