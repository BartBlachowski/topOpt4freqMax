function report = confbench_selftest(outFile)
%CONFBENCH_SELFTEST  Mechanics tests for the conference benchmark plumbing.
%
%   Everything here runs at sub-160x20 scale or with no solve at all.  It is
%   MECHANICS ONLY: scientific_evidence = false, performance_campaign = false.
%   Nothing it produces may be cited as a result.
%
%   What it proves:
%     T1  a solver that raises produces a RUN_ERROR record, not a crash
%     T2  the Olhoff dispatch gate REFUSES when a superseded implementation
%         shadows the import  (this is the fail-closed property, tested by
%         actually shadowing it)
%     T3  every method's record carries the same top-level field set, so the
%         driver can build one struct array from all three
%     T4  the timing-accounting identity and its FAIL flag both behave: a
%         consistent record passes, a deliberately inconsistent one is flagged
%     T5  the exported artifacts contain no memory column beyond the one
%         explicitly marked deprecated and unmeasured
%     T6  the frozen Olhoff configuration rejects an odd nely
%     T7  the preflight refuses a mesh above 160x20 without acknowledgement
%
%   See also PERFORMANCE_COMPARISON, CONFBENCH_PREFLIGHT.

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(here)));
if nargin < 1 || isempty(outFile)
    outFile = fullfile(repo,'examples','Performance','conference_benchmark','smoke', ...
        'mechanics_selftest.json');
end

report = struct('generated', char(string(datetime('now','TimeZone','local', ...
    'Format','yyyy-MM-dd''T''HH:mm:ssXXX'))), ...
    'scientific_evidence', false, 'performance_campaign', false, ...
    'note', 'MECHANICS ONLY. Sub-160x20 or no solve. Not citable as a result.', ...
    'tests', struct('id', {}, 'name', {}, 'pass', {}, 'detail', {}));

% ---- T1: failure handling ----------------------------------------------
try
    bad = confbench_method_config('proposed', 40, 6);
    bad.optimization.filter.type = 'no_such_filter';
    rec = confbench_run_case('proposed', bad, struct('max_outer_override', 2));
    ok = strcmp(rec.status, 'RUN_ERROR') && ~rec.ok && ~isempty(rec.error);
    report = addT(report, 'T1', 'a raising solver yields RUN_ERROR, not a crash', ok, ...
        sprintf('status=%s ok=%d note=%s', rec.status, rec.ok, firstLine(rec.status_note)));
catch ME
    report = addT(report, 'T1', 'a raising solver yields RUN_ERROR, not a crash', false, ...
        ['the error escaped: ' ME.message]);
end

% ---- T2: fail-closed dispatch -------------------------------------------
entryPath = path();
try
    superseded = fullfile(repo, 'Matlab', 'reproduction2007', 'algo');
    addpath(superseded, '-begin');
    refused = false; msg = '';
    try
        olhoffm4_assert_dispatch({'olhoffOpt'}, olhoffm4_root());
    catch ME
        refused = strcmp(ME.identifier, 'olhoffm4_assert_dispatch:WrongImplementation');
        msg = firstLine(ME.message);
    end
    path(entryPath);
    % ... and the guard must resolve it correctly again once installed
    g = olhoffm4_paths(); %#ok<NASGU>
    correct = strncmp(which('olhoffOpt'), olhoffm4_root(), numel(olhoffm4_root()));
    clear g
    report = addT(report, 'T2', 'dispatch gate refuses a shadowing superseded implementation', ...
        refused && correct, sprintf('refused=%d, guard then resolves correctly=%d | %s', ...
        refused, correct, msg));
catch ME
    path(entryPath);
    report = addT(report, 'T2', 'dispatch gate refuses a shadowing superseded implementation', ...
        false, ME.message);
end
path(entryPath);

% ---- T3: uniform record shape -------------------------------------------
recs = {};
keys = {'proposed','yuksel','olhoff'};
for i = 1:numel(keys)
    mc = confbench_method_config(keys{i}, 40, 6);
    recs{end+1} = confbench_run_case(keys{i}, mc, struct('max_outer_override', 3)); %#ok<AGROW>
end
f1 = fieldnames(orderfields(recs{1}));
same = true; detail = '';
for i = 2:numel(recs)
    fi = fieldnames(orderfields(recs{i}));
    if ~isequal(f1, fi)
        same = false;
        detail = sprintf('%s: only here {%s}, missing {%s}', keys{i}, ...
            strjoin(setdiff(fi,f1),','), strjoin(setdiff(f1,fi),','));
    end
end
arrOk = false;
try
    arr = orderfields(recs{1});
    for i = 2:numel(recs); arr(end+1) = orderfields(recs{i}); end %#ok<AGROW>
    arrOk = numel(arr) == numel(recs);
catch ME
    detail = [detail ' | struct array assembly failed: ' ME.message];
end
report = addT(report, 'T3', 'all three methods produce the same record field set', ...
    same && arrOk, sprintf('%d fields, struct array built=%d. %s', numel(f1), arrOk, detail));

% ---- T4: timing accounting identity and its FAIL flag -------------------
good = struct('time1', 1.0, 'time2', 2.0, 'overhead_time_s', 0.5, ...
    'total_wall_time_s', 3.5, 'independent_crosscheck_residual_s', 1e-4);
bad = good; bad.overhead_time_s = 0.9;           % deliberately inconsistent
aG = confbench_accounting(good, 1e-6, 1e-9, 0.05);
aB = confbench_accounting(bad,  1e-6, 1e-9, 0.05);
report = addT(report, 'T4', 'timing-accounting identity passes and TIMING_ACCOUNTING_FAIL fires', ...
    ~aG.timing_accounting_fail && aB.timing_accounting_fail, ...
    sprintf('consistent: residual %.3e, fail=%d | inconsistent: residual %.3e, fail=%d', ...
        aG.timing_accounting_residual_s, aG.timing_accounting_fail, ...
        aB.timing_accounting_residual_s, aB.timing_accounting_fail));

% ---- T5: no memory column -----------------------------------------------
smokeDir = fullfile(repo,'examples','Performance','conference_benchmark','smoke');
memHits = {};
files = {'conference_performance_table.csv','conference_performance_table.tex', ...
         'conference_performance_detailed.csv'};
for i = 1:numel(files)
    p = fullfile(smokeDir, files{i});
    if exist(p,'file') ~= 2; continue; end
    txt = fileread(p);
    lines = strsplit(txt, newline);
    for k = 1:numel(lines)
        L = lower(lines{k});
        if (contains(L,'ram') || contains(L,'memory') || contains(L,'rss')) && ...
           ~contains(L,'deprecated') && ~contains(L,'not_measured') && ...
           ~contains(L,'peak-memory measurement was not available')
            memHits{end+1} = sprintf('%s:%d', files{i}, k); %#ok<AGROW>
        end
    end
end
report = addT(report, 'T5', 'no memory column in the exported tables', isempty(memHits), ...
    strjoin(memHits, '; '));

% ---- T6: odd nely rejected ----------------------------------------------
rejected = false; msg6 = '';
try
    olhoffm4_config(160, 21);
catch ME
    rejected = strcmp(ME.identifier, 'olhoffm4_config:OddNely');
    msg6 = firstLine(ME.message);
end
report = addT(report, 'T6', 'the frozen Olhoff configuration rejects an odd nely', rejected, msg6);

% ---- T7: large-mesh acknowledgement -------------------------------------
c = struct('resolutions', [240 30], 'methods', struct('proposed',true,'yuksel',false,'olhoff',false), ...
    'singleThread', true, 'maxOuterOverride', [], 'confirmLongCampaign', false, ...
    'outputDir', tempname(), 'runLabel', 'selftest');
c.scientificEvidence = true; c.performanceCampaign = false;
mcT7 = struct('proposed', confbench_method_config('proposed', 240, 30));
p7 = confbench_preflight(c, mcT7);
idx = find(strcmp({p7.checks.name}, 'meshes above 160x20 are explicitly acknowledged'), 1);
report = addT(report, 'T7', 'preflight refuses a mesh above 160x20 without acknowledgement', ...
    ~isempty(idx) && ~p7.checks(idx).pass && ~p7.pass, ...
    sprintf('check present=%d, refused=%d, overall preflight pass=%d', ...
        ~isempty(idx), ~isempty(idx) && ~p7.checks(idx).pass, p7.pass));

% ---- verdict -------------------------------------------------------------
report.pass = all([report.tests.pass]);
od = fileparts(outFile);
if exist(od,'dir') ~= 7; mkdir(od); end
fid = fopen(outFile,'w'); cl = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(report, 'PrettyPrint', true));

fprintf('\n---- CONFERENCE BENCHMARK MECHANICS SELF-TEST ----\n');
for i = 1:numel(report.tests)
    t = report.tests(i);
    fprintf('  [%s] %-4s %s\n', pf(t.pass), t.id, t.name);
    if ~isempty(t.detail); fprintf('              %s\n', t.detail); end
end
fprintf('  RESULT: %s   (scientific_evidence=false, performance_campaign=false)\n', pf(report.pass));
fprintf('  written to %s\n', outFile);
end

function r = addT(r, id, name, ok, detail)
r.tests(end+1) = struct('id', id, 'name', name, 'pass', logical(ok), ...
    'detail', char(string(detail)));
end
function s = pf(ok)
if ok; s = 'PASS'; else; s = 'FAIL'; end
end
function s = firstLine(t)
t = char(string(t));
parts = strsplit(t, newline);
s = parts{1};
end
