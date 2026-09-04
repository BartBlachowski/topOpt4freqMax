function files = confbench_export(cfg, records, manifest, scaling)
%CONFBENCH_EXPORT  Write every benchmark artifact.  Runs OUTSIDE all solver timing.
%
%   files = CONFBENCH_EXPORT(cfg, records, manifest, scaling)
%
%   Produces, in cfg.outputDir:
%     conference_performance_table.csv       the primary, method-native table
%     conference_performance_table.tex       the same table for the slide
%     conference_performance_detailed.csv    explicit method-specific fields
%     benchmark_results.json                 every record, full precision
%     benchmark_manifest.json                exactly what was run
%     timing_schema.json                     what each count and time means
%     BENCHMARK_NOTES.md                     the caveats, ready to paste
%
%   There is no memory column anywhere.  See CONFBENCH_CAVEATS.

if nargin < 4; scaling = struct(); end
od = cfg.outputDir;
if exist(od, 'dir') ~= 7; mkdir(od); end
cav = confbench_caveats();
files = struct();

files.primary_csv  = fullfile(od, 'conference_performance_table.csv');
files.primary_tex  = fullfile(od, 'conference_performance_table.tex');
files.detailed_csv = fullfile(od, 'conference_performance_detailed.csv');
files.results_json = fullfile(od, 'benchmark_results.json');
files.manifest_json= fullfile(od, 'benchmark_manifest.json');
files.schema_json  = fullfile(od, 'timing_schema.json');
files.notes_md     = fullfile(od, 'BENCHMARK_NOTES.md');

writePrimaryCsv(files.primary_csv, records, cav);
writeLatex(files.primary_tex, records, cav);
writeDetailedCsv(files.detailed_csv, records);
writeResultsJson(files.results_json, cfg, records, scaling, cav);
writeJson(files.manifest_json, manifest);
writeJson(files.schema_json, confbench_timing_schema());
writeNotes(files.notes_md, cfg, records, manifest, scaling, cav);
end

% =========================================================================
function writePrimaryCsv(path, R, cav)
fid = fopen(path, 'w');
c = onCleanup(@() fclose(fid));
fprintf(fid, '# %s\n', cav.table_caption);
fprintf(fid, '# %s\n', cav.olhoff_label);
fprintf(fid, '# Memory: %s\n', cav.memory);
fprintf(fid, ['Method,Mesh,Count1,Count2,Time1_s,Time2_s,Total_s,omega1,' ...
    'Count1_meaning,Count2_meaning,Time1_meaning,Time2_meaning,' ...
    'Olhoff_inner_per_outer,Olhoff_inner_time_share_pct,Status\n']);
for i = 1:numel(R)
    r = R(i);
    [c1, c2, t1, t2, tt] = primaryCells(r);
    if strcmp(r.method_key, 'olhoff') && isfield(r.counts, 'inner_iterations_per_outer_mean')
        ipo = num(r.counts.inner_iterations_per_outer_mean, '%.4f');
        shr = num(r.times.inner_time_share_pct, '%.4f');
    else
        ipo = 'N/A'; shr = 'N/A';
    end
    fprintf(fid, '%s,%dx%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', ...
        csvText(r.method), r.mesh(1), r.mesh(2), c1, c2, t1, t2, tt, ...
        num(r.omega1_native, '%.10g'), ...
        csvText(nameOr(r.counts, 'count1_name')), csvText(nameOr(r.counts, 'count2_name')), ...
        csvText(nameOr(r.times, 'time1_name')),  csvText(nameOr(r.times, 'time2_name')), ...
        ipo, shr, csvText(r.status));
end
end

% =========================================================================
function writeLatex(path, R, cav)
fid = fopen(path, 'w');
c = onCleanup(@() fclose(fid));
fprintf(fid, '%% Conference performance table -- generated, do not edit by hand.\n');
fprintf(fid, '%% %s\n', cav.olhoff_label);
fprintf(fid, '\\begin{table}[t]\n\\centering\n');
fprintf(fid, '\\begin{tabular}{llrrrrrr}\n\\hline\n');
fprintf(fid, ['Method & Mesh & Count 1 & Count 2 & Time 1 [s] & Time 2 [s] & ' ...
    'Total [s] & $\\omega_1$ \\\\\n\\hline\n']);
for i = 1:numel(R)
    r = R(i);
    [c1, c2, t1, t2, tt] = primaryCells(r, '%.2f');
    fprintf(fid, '%s & $%d\\times%d$ & %s & %s & %s & %s & %s & %s \\\\\n', ...
        texEscape(r.method), r.mesh(1), r.mesh(2), c1, c2, t1, t2, tt, ...
        num(r.omega1_native, '%.2f'));
end
fprintf(fid, '\\hline\n\\end{tabular}\n');
fprintf(fid, '\\caption{%s\n', texEscape(cav.table_caption));
fprintf(fid, '%s\n', texEscape(interpretationSentence(R)));
fprintf(fid, '%s}\n', texEscape(cav.olhoff));
fprintf(fid, '\\label{tab:conference-performance}\n\\end{table}\n');

% Olhoff-specific exposure, required alongside the primary table.
olh = R(strcmp({R.method_key}, 'olhoff'));
if ~isempty(olh)
    fprintf(fid, '\n%% Nested-scheme detail for the Du-Olhoff reconstruction (M4):\n');
    for i = 1:numel(olh)
        r = olh(i);
        if isfield(r.counts, 'inner_iterations_per_outer_mean')
            fprintf(fid, '%%   %dx%d: %.2f inner MMA iterations per outer, inner time share %.1f%%\n', ...
                r.mesh(1), r.mesh(2), r.counts.inner_iterations_per_outer_mean, ...
                r.times.inner_time_share_pct);
        end
    end
end
end

% =========================================================================
function writeDetailedCsv(path, R)
%WRITEDETAILEDCSV  Explicit method-specific field names, full precision.
fid = fopen(path, 'w');
c = onCleanup(@() fclose(fid));
fprintf(fid, ['method,method_key,nelx,nely,n_elements,status,status_note,ok,' ...
    'scientific_observation,' ...
    'omega1_native,omega2_native,omega3_native,' ...
    'proposed_stage1_solves,proposed_stage2_iterations,' ...
    'proposed_stage1_time_s,proposed_stage1_reference_eigen_time_s,proposed_stage2_time_s,' ...
    'yuksel_stage1_iterations,yuksel_stage2_iterations,yuksel_iterations_total,' ...
    'yuksel_stage1_time_s,yuksel_stage2_time_s,' ...
    'olhoff_outer_iterations,olhoff_inner_iterations_total,' ...
    'olhoff_inner_iterations_per_outer_mean,' ...
    'olhoff_outer_time_excluding_inner_s,olhoff_inner_time_total_s,' ...
    'olhoff_inner_time_per_outer_mean_s,olhoff_inner_time_per_inner_iteration_mean_s,' ...
    'olhoff_inner_time_share_pct,olhoff_eigen_time_s,olhoff_gradient_time_s,' ...
    'olhoff_outer_bookkeeping_time_s,' ...
    'overhead_time_s,total_wall_time_s,' ...
    'timing_accounting_residual_s,timing_accounting_relative_residual,' ...
    'timing_accounting_fail,independent_crosscheck_residual_s,' ...
    'independent_crosscheck_fail,' ...
    'stop_reason,volume,grayness,' ...
    'omega1_common_raw_E1,omega1_common_raw_E2,omega1_common_raw_E3,' ...
    'max_ram_mb_DEPRECATED_UNMEASURED\n']);
for i = 1:numel(R)
    r = R(i);
    C = r.counts; T = r.times; A = r.accounting;
    ev = evalOr(r);
    % The method-specific blocks are GATED on the method.  Proposed and Yuksel
    % both carry a field called stage2_iterations and both carry stage1_time_s,
    % and they mean different things; without the gate a Yuksel row would fill
    % the Proposed columns with Yuksel numbers.
    isP = strcmp(r.method_key, 'proposed');
    isY = strcmp(r.method_key, 'yuksel');
    isO = strcmp(r.method_key, 'olhoff');
    fprintf(fid, '%s,%s,%d,%d,%d,%s,%s,%d,%d,', csvText(r.method), csvText(r.method_key), ...
        r.mesh(1), r.mesh(2), r.mesh(1)*r.mesh(2), csvText(r.status), ...
        csvText(r.status_note), r.ok, isfield(r,'scientific_observation') && r.scientific_observation);
    fprintf(fid, '%s,%s,%s,', num(r.omega(1)), num(vecOr(r.omega,2)), num(vecOr(r.omega,3)));
    fprintf(fid, '%s,%s,%s,%s,%s,', g(isP,C,'stage1_solves'), g(isP,C,'stage2_iterations'), ...
        g(isP,T,'stage1_time_s'), g(isP,T,'stage1_reference_eigen_time_s'), g(isP,T,'stage2_time_s'));
    fprintf(fid, '%s,%s,%s,%s,%s,', g(isY,C,'stage1_iterations'), g(isY,C,'stage2_iterations'), ...
        g(isY,C,'iterations_total_generic'), g(isY,T,'stage1_time_s'), g(isY,T,'stage2_time_s'));
    fprintf(fid, '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,', ...
        g(isO,C,'outer_iterations'), g(isO,C,'inner_iterations_total'), ...
        g(isO,C,'inner_iterations_per_outer_mean'), g(isO,T,'outer_time_excluding_inner_s'), ...
        g(isO,T,'inner_time_total_s'), g(isO,T,'inner_time_per_outer_mean_s'), ...
        g(isO,T,'inner_time_per_inner_iteration_mean_s'), g(isO,T,'inner_time_share_pct'), ...
        g(isO,T,'eigen_time_s'), g(isO,T,'gradient_time_s'), g(isO,T,'outer_bookkeeping_time_s'));
    fprintf(fid, '%s,%s,', f(T,'overhead_time_s'), f(T,'total_wall_time_s'));
    fprintf(fid, '%s,%s,%d,%s,%d,', f(A,'timing_accounting_residual_s'), ...
        f(A,'timing_accounting_relative_residual'), logicalOr(A,'timing_accounting_fail'), ...
        f(A,'independent_crosscheck_residual_s'), logicalOr(A,'independent_crosscheck_fail'));
    fprintf(fid, '%s,%s,%s,', csvText(f(r.stopping,'stop_reason')), ...
        f(r.stopping,'volume'), f(r.stopping,'final_grayness'));
    fprintf(fid, '%s,%s,%s,', ev.E1, ev.E2, ev.E3);
    fprintf(fid, 'NOT_MEASURED\n');
end
end

% =========================================================================
function writeResultsJson(path, cfg, R, scaling, cav)
out = struct();
out.schema = 'conference_performance_benchmark_results/1';
out.generated = char(string(datetime('now','TimeZone','local','Format','yyyy-MM-dd''T''HH:mm:ssXXX')));
out.run_label = cfg.runLabel;
out.scientific_evidence = cfg.scientificEvidence;
out.performance_campaign = cfg.performanceCampaign;
out.caveats = cav;
out.timing_schema = confbench_timing_schema();
out.memory = struct('measured', false, 'reported', false, 'reason', cav.memory);
out.scaling = scaling;
recs = cell(numel(R),1);
for i = 1:numel(R)
    r = R(i);
    if isfield(r, 'x'); r = rmfield(r, 'x'); end
    if isfield(r, 'telemetry'); r = rmfield(r, 'telemetry'); end
    if isfield(r, 'effective_config'); r = rmfield(r, 'effective_config'); end
    recs{i} = r;
end
out.runs = recs;
writeJson(path, out);
end

% =========================================================================
function writeNotes(path, cfg, R, manifest, scaling, cav)
fid = fopen(path, 'w');
c = onCleanup(@() fclose(fid));
fprintf(fid, '# Conference performance benchmark -- notes\n\n');
fprintf(fid, 'Generated %s from `examples/Performance/performance_comparison.m`.\n\n', ...
    manifest.generated_datetime);
fprintf(fid, '- run label: `%s`\n', cfg.runLabel);
fprintf(fid, '- scientific evidence: **%s**\n', tf(cfg.scientificEvidence));
fprintf(fid, '- performance campaign: **%s**\n', tf(cfg.performanceCampaign));
fprintf(fid, '- resolutions: %s\n', meshList(cfg.resolutions));
fprintf(fid, '- threads: %d\n\n', manifest.environment.max_num_comp_threads);

fprintf(fid, '## How to read the table\n\n%s\n\n', cav.table_caption);
fprintf(fid, '%s\n\n', interpretationSentence(R));

fprintf(fid, '## Du-Olhoff reconstruction (M4)\n\n');
fprintf(fid, '%s\n\n', cav.olhoff_label);
fprintf(fid, '> %s\n\n', cav.olhoff);
fprintf(fid, '%s\n\n', cav.olhoff_iteration_counts);

fprintf(fid, '## Memory\n\n%s\n\n', cav.memory);

fprintf(fid, '## Scaling\n\n%s\n\n', cav.scaling);
if isfield(scaling, 'fitted') && scaling.fitted
    fprintf(fid, '| Method | C | p | R^2 | points |\n|---|---|---|---|---|\n');
    for i = 1:numel(scaling.methods)
        s = scaling.methods(i);
        fprintf(fid, '| %s | %.6e | %.4f | %.4f | %d |\n', s.method, s.C, s.p, s.R2, s.n);
    end
    fprintf(fid, '\n');
else
    fprintf(fid, '_No scaling fit was performed for this run: %s_\n\n', scalingReason(scaling));
end

fprintf(fid, '## Results\n\n');
fprintf(fid, '| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Total [s] | omega1 | Status |\n');
fprintf(fid, '|---|---|---|---|---|---|---|---|---|\n');
for i = 1:numel(R)
    r = R(i);
    [c1, c2, t1, t2, tt] = primaryCells(r, '%.3f');
    fprintf(fid, '| %s | %dx%d | %s | %s | %s | %s | %s | %s | %s |\n', ...
        r.method, r.mesh(1), r.mesh(2), c1, c2, t1, t2, tt, ...
        num(r.omega1_native, '%.4f'), r.status);
end
fprintf(fid, '\n');
end

% =========================================================================
function [c1, c2, t1, t2, tt] = primaryCells(r, tfmt)
if nargin < 2; tfmt = '%.9g'; end
c1 = f(r.counts, 'count1', '%.6g');
c2 = f(r.counts, 'count2', '%.6g');
t1 = f(r.times, 'time1', tfmt);
t2 = f(r.times, 'time2', tfmt);
tt = f(r.times, 'total_wall_time_s', tfmt);
end

function s = interpretationSentence(R)
keys = unique({R.method_key});
parts = {};
if any(strcmp(keys,'proposed'))
    parts{end+1} = ['Proposed: Count 1 = reference eigenanalysis solves (always 1, ' ...
        'not an optimization iteration), Count 2 = SIMP iterations, Time 1 = ' ...
        'eigenanalysis and preparation, Time 2 = SIMP.'];
end
if any(strcmp(keys,'yuksel'))
    parts{end+1} = ['Yuksel: Count 1 and Count 2 are the Stage-1 and Stage-2 ' ...
        'iteration counts, Time 1 and Time 2 the corresponding stage times.'];
end
if any(strcmp(keys,'olhoff'))
    parts{end+1} = ['Du-Olhoff reconstruction (M4): Count 1 = outer iterations, ' ...
        'Count 2 = cumulative nested MMA iterations, Time 1 = outer work ' ...
        'excluding the nested MMA solve, Time 2 = nested MMA total. The two ' ...
        'counts are never added.'];
end
s = strjoin(parts, ' ');
end

function ev = evalOr(r)
ev = struct('E1','N/A','E2','N/A','E3','N/A');
if ~isfield(r,'evaluator') || isempty(r.evaluator) || ~isstruct(r.evaluator); return; end
e = r.evaluator;
models = {'E1','E2','E3'};
for i = 1:numel(models)
    key = ['selected_omega_raw_' models{i}];
    if isfield(e, key); ev.(models{i}) = num(e.(key)); end
end
end

function v = vecOr(a, i)
if numel(a) >= i; v = a(i); else; v = NaN; end
end

function s = g(gate, S, name)
%G  Method-gated cell: 'N/A' unless this row belongs to the method that owns
%   the field.  Two methods can legitimately use the same field name for
%   different quantities; the gate is what keeps the columns honest.
if gate; s = f(S, name); else; s = 'N/A'; end
end

function s = f(S, name, fmt)
if nargin < 3; fmt = '%.17g'; end
if isstruct(S) && isfield(S, name) && ~isempty(S.(name))
    v = S.(name);
    if ischar(v) || isstring(v); s = char(string(v)); else; s = num(v, fmt); end
else
    s = 'N/A';
end
end

function b = logicalOr(S, name)
if isstruct(S) && isfield(S, name) && ~isempty(S.(name)); b = logical(S.(name)); else; b = false; end
end

function s = num(v, fmt)
if nargin < 2; fmt = '%.17g'; end
if isempty(v) || (isnumeric(v) && ~isfinite(v)); s = 'N/A'; else; s = sprintf(fmt, double(v)); end
end

function s = nameOr(S, name)
if isstruct(S) && isfield(S, name); s = char(string(S.(name))); else; s = 'N/A'; end
end

function s = csvText(v)
s = char(string(v));
s = strrep(s, newline, ' ');
s = strrep(s, ',', ';');
s = strrep(s, '"', '''');
end

function s = texEscape(v)
s = char(string(v));
s = strrep(s, '\', '\textbackslash{}');
s = strrep(s, '_', '\_');
s = strrep(s, '%', '\%');
s = strrep(s, '&', '\&');
s = strrep(s, '#', '\#');
end

function s = meshList(Rm)
parts = arrayfun(@(i) sprintf('%dx%d', Rm(i,1), Rm(i,2)), 1:size(Rm,1), 'UniformOutput', false);
s = strjoin(parts, ', ');
end

function s = tf(b)
if b; s = 'true'; else; s = 'false'; end
end

function s = scalingReason(scaling)
if isfield(scaling, 'reason'); s = char(string(scaling.reason)); else; s = 'not requested'; end
end

function writeJson(path, s)
fid = fopen(path, 'w');
c = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(s, 'PrettyPrint', true));
end
