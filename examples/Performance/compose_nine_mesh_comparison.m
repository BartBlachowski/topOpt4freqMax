%COMPOSE_NINE_MESH_COMPARISON  Three-method table and complexity plots from two campaigns.
%
%   Proposed and Yuksel have not changed since the 2026-09-11 campaign
%   (campaign_9mesh_r2); the 2026-09-14 campaign (nine_mesh_pedersen_b21483b)
%   re-ran only the Du-Olhoff column, at the Pedersen/adaptive-box preset.  This
%   script composes the comparison from the two RECORDED campaigns -- nothing is
%   re-solved and nothing is re-timed:
%
%     Proposed, Yuksel   rows of campaign_9mesh_r2/benchmark_records.mat
%     Du-Olhoff          rows of nine_mesh_pedersen_b21483b/benchmark_records.mat
%
%   The historical "Du-Olhoff reconstruction (M4)" rows of campaign_9mesh_r2 are
%   NOT used.
%
%   The artifacts are written by the campaign's own writers
%   (CONFBENCH_SCALING_FIT, CONFBENCH_EXPORT, CONFBENCH_COMPLEXITY_PLOTS), so the
%   table, the fits and the four figures have exactly the format of a single
%   campaign.  Output: conference_benchmark/<outLabel>/.
%
%   Nothing is written unless
%     - both source record files hash to their pinned SHA-256,
%     - both sources are complete performance campaigns (scientific evidence),
%     - both ran on the same host, MATLAB version, BLAS and thread count, with
%       the same run settings and the same meshes,
%     - the solver sources and frozen profiles of Proposed and Yuksel hash
%       identically in the two campaign manifests,
%     - every (method, mesh) row exists exactly once and carries the label the
%       current benchmark code gives that method.
%
%   What the composition cannot remove, and the notes therefore say: the rows
%   were timed in two separate MATLAB sessions three days apart on the same
%   machine, not interleaved mesh by mesh in one session.
%
%   TIMING SCHEMA 2.  campaign_9mesh_r2 recorded the Proposed rows under timing
%   schema 1, whose Time 1 folded the solver preparation into the reference
%   eigenanalysis (0.43 s at 160x20 for a 0.03 s eigensolve).  Schema 2 makes
%   Time 1 the eigenanalysis alone and moves the preparation to overhead --
%   the table's "Other" column -- where the Yuksel and Du-Olhoff records already
%   place their setup.  The Proposed rows are brought to schema 2 here by
%   CONFBENCH_PROPOSED_TIMES from their RECORDED sub-interval timers
%   (stage1_time_s, stage1_reference_eigen_time_s, stage2_time_s,
%   total_wall_time_s); the accounting identity is re-checked and gated like
%   every other composition check.  Nothing is re-timed.
%
%   See also PERFORMANCE_COMPARISON, CONFBENCH_EXPORT, CONFBENCH_COMPLEXITY_PLOTS,
%   CONFBENCH_PROPOSED_TIMES.

clear; clc;

%% ============================================================
%  COMPOSITION
%  ============================================================
sources = struct( ...
    'label',          {'campaign_9mesh_r2', 'nine_mesh_pedersen_b21483b'}, ...
    'methods',        {{'proposed', 'yuksel'}, {'olhoff'}}, ...
    'records_sha256', {'873125858df02664d9a7d37bd09e4e9b0b1ccff5c37940d088f9b1e424921fae', ...
                       '261bf8fc94a1efb7da223aae64ac72da336a4d8f48341f0d8bf79b0a4d2abe82'});
outLabel  = 'nine_mesh_comparison_pedersen_b21483b';
overwrite = true;    % the output folder is regenerated in place from the two recorded campaigns

% Files whose manifest hashes must agree between the two campaigns for the
% Proposed and Yuksel rows to stand next to the new Olhoff rows.  These are
% KEYS into the two recorded (pre-2026-09-14) campaign manifests, which name
% the sources by their paths at the time; they are never opened here.  Since
% the repository cleanup the files live at analysis/Proposed, analysis/Yuksel
% and examples/Performance/benchmark_profile.
unchangedSources = { ...
    'analysis/ourApproach/Matlab/topopt_freq.m', ...
    'analysis/YukselApproach/Matlab/top99neo_inertial_freq.m', ...
    'analysis/three_method_parametric_study/results/profile_freeze_manifest.json', ...
    'analysis/three_method_parametric_study/study_base_config.m', ...
    'analysis/three_method_parametric_study/study_evaluate_design.m', ...
    'tools/Matlab/run_topopt_from_json.m'};
envFields   = {'hostname', 'computer', 'matlab_version', 'blas_library', 'max_num_comp_threads'};
methodOrder = {'proposed', 'yuksel', 'olhoff'};   % row order within a mesh, as in campaign tables

%% ============================================================
%  PATHS
%  ============================================================
scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(fileparts(scriptDir));
addpath(scriptDir);
addpath(fullfile(scriptDir, 'conference_bench'));
addpath(fullfile(repoRoot, 'analysis', 'Olhoff'));   % display name and caveat of the Olhoff preset
benchRoot = fullfile(scriptDir, 'conference_benchmark');
benchRel  = 'examples/Performance/conference_benchmark';
outDir    = fullfile(benchRoot, outLabel);

%% ============================================================
%  LOAD AND CHECK -- nothing is written before every check passes
%  ============================================================
checks = struct('name', {}, 'pass', {}, 'detail', {});
S = cell(1, numel(sources));
for k = 1:numel(sources)
    d = fullfile(benchRoot, sources(k).label);
    recPath = fullfile(d, 'benchmark_records.mat');
    h = fileSha256(recPath);
    checks = addCheck(checks, sprintf('%s records hash to the pinned SHA-256', sources(k).label), ...
        strcmp(h, sources(k).records_sha256), h);
    L = load(recPath, 'records', 'cfg', 'manifest');
    checks = addCheck(checks, sprintf('%s is a complete performance campaign', sources(k).label), ...
        L.cfg.performanceCampaign && L.cfg.scientificEvidence, ...
        sprintf('performanceCampaign=%d scientificEvidence=%d', L.cfg.performanceCampaign, L.cfg.scientificEvidence));
    checks = addCheck(checks, sprintf('%s ran single-threaded', sources(k).label), ...
        L.manifest.environment.max_num_comp_threads == 1, ...
        sprintf('max_num_comp_threads=%d', L.manifest.environment.max_num_comp_threads));
    L.records_sha256  = h;
    L.manifest_sha256 = fileSha256(fullfile(d, 'benchmark_manifest.json'));
    S{k} = L;
end

settingsOf = @(c) rmfield(c, {'methods', 'outputDir', 'runLabel'});
for k = 2:numel(S)
    pair = sprintf('%s vs %s', sources(1).label, sources(k).label);
    for f = envFields
        a = S{1}.manifest.environment.(f{1});
        b = S{k}.manifest.environment.(f{1});
        checks = addCheck(checks, sprintf('environment.%s identical (%s)', f{1}, pair), ...
            isequal(a, b), sprintf('%s | %s', valText(a), valText(b)));
    end
    checks = addCheck(checks, sprintf('identical run settings and meshes (%s)', pair), ...
        isequal(settingsOf(S{1}.cfg), settingsOf(S{k}.cfg)), mat2str(S{1}.cfg.resolutions));
    for p = unchangedSources
        a = sourceHash(S{1}.manifest, p{1});
        b = sourceHash(S{k}.manifest, p{1});
        checks = addCheck(checks, sprintf('%s unchanged (%s)', p{1}, pair), ...
            ~isempty(a) && strcmp(a, b), sprintf('%s | %s', a, b));
    end
end

res = S{1}.cfg.resolutions;
records = [];
rowOrigin = struct('method', {}, 'mesh', {}, 'source', {});
for i = 1:size(res, 1)
    for m = methodOrder
        k = find(cellfun(@(c) any(strcmp(c, m{1})), {sources.methods}));
        if numel(k) ~= 1
            error('compose_nine_mesh_comparison:MethodSource', ...
                'Method %s must come from exactly one source (found %d).', m{1}, numel(k));
        end
        R = S{k}.records;
        sel = find(strcmp({R.method_key}, m{1}) & arrayfun(@(r) isequal(r.mesh(:).', res(i, :)), R));
        mesh = sprintf('%dx%d', res(i, 1), res(i, 2));
        checks = addCheck(checks, sprintf('%s %s present exactly once in %s', m{1}, mesh, sources(k).label), ...
            numel(sel) == 1, sprintf('%d row(s)', numel(sel)));
        if numel(sel) ~= 1; continue; end
        r = R(sel);
        checks = addCheck(checks, sprintf('%s %s carries the current display name', m{1}, mesh), ...
            strcmp(r.method, confbench_display_name(m{1})), r.method);
        records = [records, r]; %#ok<AGROW>
        rowOrigin(end+1) = struct('method', r.method, 'mesh', mesh, 'source', sources(k).label); %#ok<SAGROW>
    end
end
olhRows = records(strcmp({records.method_key}, 'olhoff'));
checks = addCheck(checks, 'Olhoff rows ran the preset the benchmark names', ...
    all(strcmp({olhRows.production_preset}, confbench_olhoff_preset())), confbench_olhoff_preset());

% ---- Proposed rows: recorded schema-1 times -> timing schema 2 --------------
% Time 1 becomes the reference eigenanalysis alone; the preparation goes to
% overhead_time_s.  Every input is a recorded timer of the row; the identity
% Total = Time 1 + Time 2 + Other and the independent cross-check are re-run
% through the same accounting code the campaign used, and gated below.
olhSource = find(cellfun(@(c) any(strcmp(c, 'olhoff')), {sources.methods}));
tolCfg = S{olhSource}.cfg;
reaccounted = struct('method', {}, 'mesh', {}, 'time1_before_s', {}, 'time1_after_s', {}, ...
    'overhead_before_s', {}, 'overhead_after_s', {}, 'total_s', {});
for i = find(strcmp({records.method_key}, 'proposed'))
    T = records(i).times;
    mesh = sprintf('%dx%d', records(i).mesh(1), records(i).mesh(2));
    hasInputs = all(isfield(T, {'stage1_time_s', 'stage1_reference_eigen_time_s', ...
        'stage2_time_s', 'total_wall_time_s', 'solver_self_report_wall_s'}));
    checks = addCheck(checks, sprintf('proposed %s carries the recorded timers schema 2 needs', mesh), ...
        hasInputs, 'stage1_time_s, stage1_reference_eigen_time_s, stage2_time_s, total_wall_time_s, solver_self_report_wall_s');
    if ~hasInputs; continue; end
    new = confbench_proposed_times(T.stage1_time_s, T.stage1_reference_eigen_time_s, ...
        T.stage2_time_s, T.total_wall_time_s, T.solver_self_report_wall_s);
    acc = confbench_accounting(new, tolCfg.timingTolAbs, tolCfg.timingTolRel, tolCfg.crosscheckTolRel);
    checks = addCheck(checks, sprintf('proposed %s re-accounted to schema 2: identity and cross-check hold', mesh), ...
        ~acc.timing_accounting_fail && ~acc.independent_crosscheck_fail && ...
        abs(new.total_wall_time_s - T.total_wall_time_s) == 0, ...
        sprintf('time1 %.4f -> %.4f s, overhead %.4f -> %.4f s, residual %.1e s', ...
            T.time1, new.time1, T.overhead_time_s, new.overhead_time_s, acc.timing_accounting_residual_s));
    reaccounted(end+1) = struct('method', records(i).method, 'mesh', mesh, ...
        'time1_before_s', T.time1, 'time1_after_s', new.time1, ...
        'overhead_before_s', T.overhead_time_s, 'overhead_after_s', new.overhead_time_s, ...
        'total_s', T.total_wall_time_s); %#ok<SAGROW>
    records(i).times = new;
    records(i).accounting = acc;
end

fprintf('Composition checks:\n');
for i = 1:numel(checks)
    fprintf('  [%s] %s  (%s)\n', passText(checks(i).pass), checks(i).name, checks(i).detail);
end
if ~all([checks.pass])
    error('compose_nine_mesh_comparison:ChecksFailed', ...
        '%d composition check(s) failed; nothing was written.', sum(~[checks.pass]));
end
if exist(outDir, 'dir') == 7 && ~overwrite
    error('compose_nine_mesh_comparison:OutputExists', ...
        'Output folder exists: %s (set overwrite = true to replace its files).', outDir);
end
if exist(outDir, 'dir') ~= 7; mkdir(outDir); end

%% ============================================================
%  CONFIGURATION, SCALING, MANIFEST
%  ============================================================
cfg = S{olhSource}.cfg;
cfg.methods   = struct('proposed', true, 'yuksel', true, 'olhoff', true);
cfg.outputDir = outDir;
cfg.runLabel  = outLabel;

scaling = confbench_scaling_fit(cfg, records);

srcInfo = struct([]);
for k = 1:numel(sources)
    M = S{k}.manifest;
    srcInfo(k).label              = sources(k).label;
    srcInfo(k).dir                = [benchRel '/' sources(k).label];
    srcInfo(k).methods_taken      = strjoin(cellfun(@confbench_display_name, sources(k).methods, ...
                                        'UniformOutput', false), '; ');
    srcInfo(k).campaign_generated = M.generated_datetime;
    srcInfo(k).repository_head    = M.repository.head;
    srcInfo(k).repository_dirty   = M.repository.dirty;
    srcInfo(k).records_sha256     = S{k}.records_sha256;
    srcInfo(k).manifest_sha256    = S{k}.manifest_sha256;
end

compositionNote = sprintf(['Rows are composed from two recorded campaigns run on the same host ' ...
    '(%s, MATLAB %s, %s, %d thread): Proposed and Yuksel from %s (generated %s), the %s from %s ' ...
    '(generated %s). Nothing was re-solved or re-timed. The Proposed/Yuksel rows and the ' ...
    'Du-Olhoff rows were timed in separate MATLAB sessions, not interleaved mesh by mesh in one ' ...
    'session. The historical "Du-Olhoff reconstruction (M4)" rows of %s are not used. ' ...
    'The Proposed rows are re-accounted to timing schema 2 from their recorded timers: Time 1 ' ...
    'is the reference eigenanalysis alone and the solver preparation is in Other, as for the ' ...
    'other two methods.'], ...
    S{1}.manifest.environment.hostname, S{1}.manifest.environment.matlab_release, ...
    S{1}.manifest.environment.blas_library, S{1}.manifest.environment.max_num_comp_threads, ...
    sources(1).label, srcInfo(1).campaign_generated, confbench_display_name('olhoff'), ...
    sources(olhSource).label, srcInfo(olhSource).campaign_generated, sources(1).label);

manifest = struct();
manifest.manifest_schema    = 'conference_performance_composite/1';
manifest.generated_datetime = char(string(datetime('now', 'TimeZone', 'local', ...
                                  'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX')));
manifest.generated_by       = 'examples/Performance/compose_nine_mesh_comparison.m';
manifest.manifest_role      = ['OUTPUT. A composition of two recorded campaigns; nothing was ' ...
                               're-solved or re-timed. The source manifests remain authoritative ' ...
                               'for what each row ran.'];
manifest.composition_note   = compositionNote;
manifest.configuration      = cfg;
manifest.active_resolutions = res;
manifest.active_resolutions_count = size(res, 1);
manifest.active_methods     = methodOrder;
manifest.element_counts     = (res(:, 1) .* res(:, 2)).';
manifest.sources            = srcInfo;
manifest.row_origin         = rowOrigin;
manifest.composition_checks = checks;
manifest.unchanged_sources  = unchangedSources;
manifest.environment        = S{olhSource}.manifest.environment;
manifest.olhoff_implementation = S{olhSource}.manifest.olhoff_implementation;
manifest.caveats            = confbench_caveats();
manifest.timing_schema      = confbench_timing_schema();
manifest.timing_reaccounting = struct( ...
    'rule', ['Proposed rows recorded under timing schema 1 are brought to schema 2 by ' ...
             'confbench_proposed_times: time1 = stage1_reference_eigen_time_s, ' ...
             'overhead_time_s = total_wall_time_s - time1 - stage2_time_s. Inputs are ' ...
             'recorded timers; nothing was re-timed. Identity and cross-check re-run and gated.'], ...
    'rows', reaccounted);
manifest.cap_summary        = struct('any_cap_hit', any(~[records.ok]), ...
                                     'n_not_ok', sum(~[records.ok]), 'n_records', numel(records));

%% ============================================================
%  ARTIFACTS
%  ============================================================
files = confbench_export(cfg, records, manifest, scaling);
annotateOutputs(files, manifest, srcInfo, compositionNote);
save(fullfile(outDir, 'benchmark_records.mat'), 'records', 'cfg', 'manifest', 'scaling', '-v7.3');
plotFiles = confbench_complexity_plots(cfg, records, scaling);
pdfPath = writePreviewTable(outDir);

fprintf('\nWritten to %s\n', outDir);
fn = [fieldnames(files); fieldnames(plotFiles)];
for i = 1:numel(fn)
    if isfield(files, fn{i}); p = files.(fn{i}); else; p = plotFiles.(fn{i}); end
    fprintf('  %-26s %s\n', fn{i}, p);
end
fprintf('  %-26s %s\n', 'table_pdf', pdfPath);
fprintf('\nTotal-time fits T(Ne) = C * Ne^p:\n');
for i = 1:numel(scaling.methods)
    s = scaling.methods(i);
    fprintf('  %-75s C = %.6e  p = %.4f  R^2 = %.4f  (%d points)\n', s.method, s.C, s.p, s.R2, s.n);
end

%% ============================================================
%  LOCAL FUNCTIONS
%  ============================================================
function annotateOutputs(files, manifest, srcInfo, note)
%ANNOTATEOUTPUTS  Say in every human-facing file that the rows are composed.
generatedLine = sprintf('Generated %s from `examples/Performance/performance_comparison.m`.', ...
    manifest.generated_datetime);
notes = fileread(files.notes_md);
if ~contains(notes, generatedLine)
    error('compose_nine_mesh_comparison:NotesFormat', ...
        'BENCHMARK_NOTES.md no longer has the expected "Generated" line; not annotating blindly.');
end
tbl = sprintf(['| Method(s) | Source campaign | Campaign generated | Repository HEAD | Records SHA-256 |\n' ...
    '|---|---|---|---|---|\n']);
for k = 1:numel(srcInfo)
    dirty = ''; if srcInfo(k).repository_dirty; dirty = ' (dirty)'; end
    tbl = [tbl sprintf('| %s | `%s` | %s | `%s`%s | `%s` |\n', srcInfo(k).methods_taken, ...
        srcInfo(k).dir, srcInfo(k).campaign_generated, srcInfo(k).repository_head(1:7), dirty, ...
        srcInfo(k).records_sha256)]; %#ok<AGROW>
end
replacement = sprintf(['Generated %s by `examples/Performance/compose_nine_mesh_comparison.m` ' ...
    'from two recorded campaigns.\n\n## Composition\n\n%s\n\n%s'], ...
    manifest.generated_datetime, note, tbl);
writeText(files.notes_md, strrep(notes, generatedLine, replacement));

prependLine(files.primary_csv,  ['# Composition: ' note]);
prependLine(files.detailed_csv, ['# Composition: ' note]);
% Not the LaTeX table: it is the paper-facing fragment, and the note names the
% internal Du-Olhoff realization.  Its composition is recorded above and in the CSVs.
end

function prependLine(path, line)
writeText(path, sprintf('%s\n%s', line, fileread(path)));
end

function writeText(path, txt)
fid = fopen(path, 'w');
c = onCleanup(@() fclose(fid));
fwrite(fid, txt, 'char');
end

function pdfPath = writePreviewTable(outDir)
%WRITEPREVIEWTABLE  Standalone wrapper around the table fragment, compiled if pdflatex exists.
wrapper = strjoin({ ...
    '% Standalone preview wrapper for conference_performance_table.tex.', ...
    '%', ...
    '% The generated table file is a FRAGMENT meant to be \input{} into a paper.', ...
    '% This wrapper supplies a document class so that', ...
    '%', ...
    '%     pdflatex preview_table.tex', ...
    '%', ...
    '% renders it.  Nothing here is part of the benchmark evidence.', ...
    '%', ...
    '% The table renders no caption; its widest element is the eight-column', ...
    '% tabular at \scriptsize, which fits an A4 landscape page.', ...
    '\documentclass[11pt]{article}', ...
    '\usepackage[a4paper,landscape,margin=12mm]{geometry}', ...
    '\usepackage[T1]{fontenc}', ...
    '\usepackage{lmodern}', ...
    '\usepackage{amsmath}', ...
    '\pagestyle{empty}', ...
    '\begin{document}', ...
    '\small', ...
    '\input{conference_performance_table.tex}', ...
    '\end{document}', ''}, newline);
writeText(fullfile(outDir, 'preview_table.tex'), wrapper);

pdfPath = 'NOT PRODUCED (pdflatex not found; run pdflatex preview_table.tex)';
candidates = {'/Library/TeX/texbin/pdflatex', '/usr/local/bin/pdflatex', '/opt/homebrew/bin/pdflatex'};
exe = candidates(cellfun(@(p) exist(p, 'file') == 2, candidates));
if isempty(exe); return; end
[st, out] = system(sprintf('cd "%s" && "%s" -interaction=nonstopmode -halt-on-error preview_table.tex', ...
    outDir, exe{1}));
if st ~= 0
    warning('compose_nine_mesh_comparison:PdfFailed', 'pdflatex failed:\n%s', out);
    pdfPath = 'NOT PRODUCED (pdflatex failed; see warning)';
    return
end
for ext = {'.aux', '.log'}
    p = fullfile(outDir, ['preview_table' ext{1}]);
    if exist(p, 'file') == 2; delete(p); end
end
pdfPath = fullfile(outDir, 'preview_table.pdf');
end

function h = fileSha256(path)
fid = fopen(path, 'r');
if fid < 0
    error('compose_nine_mesh_comparison:MissingFile', 'Cannot open %s.', path);
end
c = onCleanup(@() fclose(fid));
md = java.security.MessageDigest.getInstance('SHA-256');
while true
    b = fread(fid, 2^24, '*uint8');
    if isempty(b); break; end
    md.update(typecast(b, 'int8'));
end
h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'), 2).', 1, []));
end

function h = sourceHash(manifest, relPath)
h = '';
sh = manifest.source_hashes;
fn = fieldnames(sh);
for i = 1:numel(fn)
    if strcmp(sh.(fn{i}).path, relPath); h = sh.(fn{i}).sha256; return; end
end
end

function checks = addCheck(checks, name, pass, detail)
checks(end+1) = struct('name', name, 'pass', logical(pass), 'detail', char(string(detail)));
end

function s = valText(v)
if isnumeric(v) || islogical(v); s = mat2str(v); else; s = char(string(v)); end
end

function s = passText(ok)
if ok; s = 'PASS'; else; s = 'FAIL'; end
end
