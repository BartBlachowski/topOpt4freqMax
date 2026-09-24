function files = confbench_export(cfg, records, manifest, scaling, opts)
%CONFBENCH_EXPORT  Write every benchmark artifact.  Runs OUTSIDE all solver timing.
%
%   files = CONFBENCH_EXPORT(cfg, records, manifest, scaling)
%   Optional opts.tables_only writes just CSV/LaTeX views; opts.latex_only
%   writes just the LaTeX table; opts.caveats carries the recorded method
%   descriptions when refreshing a historical campaign.
%
%   Produces, in cfg.outputDir:
%     conference_performance_table.csv       the primary, method-native table
%     conference_performance_table.tex       the same table, paper-ready: methods
%                                            carry CONFBENCH_PAPER_LABEL and no
%                                            internal caveat is rendered
%     conference_performance_detailed.csv    explicit method-specific fields
%     benchmark_results.json                 every record, full precision
%     benchmark_manifest.json                exactly what was run
%     timing_schema.json                     what each count and time means
%     BENCHMARK_NOTES.md                     the caveats, ready to paste
%
%   Table columns (timing schema 2):
%     Count 1, Count 2, Time 1, Time 2   method-native stages (see the schema)
%     Stage time                         Time 1 + Time 2
%     Other                              work inside the solver timer but
%                                        outside the two named stages
%     Total wall time                    caller-side wall time;
%                                        Stage time + Other = Total wall time
%     omega_1 native                     the solver's own material model; a
%                                        dagger marks a value that deviates
%                                        from E1 by more than the tolerance in
%                                        CONFBENCH_CAVEATS
%     omega_1 E1                         the frozen common evaluator, the only
%                                        omega_1 comparable across methods
%
%   There is no memory column anywhere.  See CONFBENCH_CAVEATS.

if nargin < 4; scaling = struct(); end
% Table-only refreshes preserve the original campaign evidence and caveats.
if nargin < 5; opts = struct(); end
od = cfg.outputDir;
if exist(od, 'dir') ~= 7; mkdir(od); end
cav = confbench_caveats();
if isfield(opts, 'caveats'); cav = opts.caveats; end
for i = 1:numel(records)
    records(i).times.stage_time_s = confbench_stage_time(records(i).times);
end
files = struct();

files.primary_csv  = fullfile(od, 'conference_performance_table.csv');
files.primary_tex  = fullfile(od, 'conference_performance_table.tex');
files.detailed_csv = fullfile(od, 'conference_performance_detailed.csv');
files.results_json = fullfile(od, 'benchmark_results.json');
files.manifest_json= fullfile(od, 'benchmark_manifest.json');
files.schema_json  = fullfile(od, 'timing_schema.json');
files.notes_md     = fullfile(od, 'BENCHMARK_NOTES.md');

if isfield(opts, 'latex_only') && opts.latex_only
    writeLatex(files.primary_tex, records, cav);
    files = struct('primary_tex', files.primary_tex);
    return
end
writePrimaryCsv(files.primary_csv, records, cav);
writeLatex(files.primary_tex, records, cav);
writeDetailedCsv(files.detailed_csv, records, cav);
if isfield(opts, 'tables_only') && opts.tables_only
    files = rmfield(files, {'results_json','manifest_json','schema_json','notes_md'});
    return
end
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
fprintf(fid, '# %s\n', cav.other_column);
fprintf(fid, '# %s\n', cav.omega1_columns);
fprintf(fid, '# %s\n', cav.olhoff_label);
fprintf(fid, '# Memory: %s\n', cav.memory);
fprintf(fid, ['Method,Mesh,Count1,Count2,Time1_s,Time2_s,Stage_time_s,Other_s,Total_s,' ...
    'omega1_native,omega1_native_flag,omega1_common_E1,' ...
    'Count1_meaning,Count2_meaning,Time1_meaning,Time2_meaning,' ...
    'Olhoff_inner_per_outer,Olhoff_inner_time_share_pct,Status\n']);
for i = 1:numel(R)
    r = R(i);
    [c1, c2, t1, t2, st, ov, tt] = primaryCells(r);
    [nat, e1, flagged] = omega1Cells(r, cav.omega1_native_flag_tol, '%.10g');
    if strcmp(r.method_key, 'olhoff') && isfield(r.counts, 'inner_iterations_per_outer_mean')
        ipo = num(r.counts.inner_iterations_per_outer_mean, '%.4f');
        shr = num(r.times.inner_time_share_pct, '%.4f');
    else
        ipo = 'N/A'; shr = 'N/A';
    end
    fprintf(fid, '%s,%dx%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', ...
        csvText(r.method), r.mesh(1), r.mesh(2), c1, c2, t1, t2, st, ov, tt, ...
        nat, flagText(flagged), e1, ...
        csvText(nameOr(r.counts, 'count1_name')), csvText(nameOr(r.counts, 'count2_name')), ...
        csvText(nameOr(r.times, 'time1_name')),  csvText(nameOr(r.times, 'time2_name')), ...
        ipo, shr, csvText(r.status));
end
end

% =========================================================================
function writeLatex(path, R, cav)
%WRITELATEX  The paper-facing table.  Rows carry CONFBENCH_PAPER_LABEL, and no
%   caption or notes block is rendered: the method caveats (formulation,
%   provenance, count/time semantics) are internal and live in the CSV tables
%   and BENCHMARK_NOTES.md, never in what the reader sees.  Columns: counts
%   and stage times only; Other, total wall time and both omega_1 values
%   (native and E1) stay in the CSV tables.  Rows are grouped by mesh, in the
%   order Du-Olhoff, Yuksel-Yilmaz, Proposed; the mesh is printed once per
%   group and a thin rule separates consecutive groups.
fid = fopen(path, 'w');
c = onCleanup(@() fclose(fid));
fprintf(fid, '%% Conference performance table -- generated, do not edit by hand.\n');
fprintf(fid, '%% Method provenance and caveats: conference_performance_table.csv, BENCHMARK_NOTES.md.\n');
fprintf(fid, '%% Columns: Stage time [s] = Time 1 + Time 2, summed before rounding. Other, total wall time and omega_1 (native and common evaluator E1) are in the CSV only.\n');
fprintf(fid, '\\begin{table}[t]\n\\centering\n');
fprintf(fid, '\\begingroup\n\\setlength{\\tabcolsep}{3pt}\n\\scriptsize\n');
% Thin (0.2 pt) rule between mesh groups; \arrayrulewidth must be set
% globally for \hline to see it, and is restored to the LaTeX default 0.4 pt.
fprintf(fid, ['\\providecommand{\\thinhline}{\\noalign{\\global\\setlength{\\arrayrulewidth}{0.2pt}}' ...
    '\\hline\\noalign{\\global\\setlength{\\arrayrulewidth}{0.4pt}}}\n']);
fprintf(fid, '\\begin{tabular}{llrrrrr}\n\\hline\n');
fprintf(fid, ['Mesh & Method & Count 1 & Count 2 & Time 1 [s] & Time 2 [s] & ' ...
    'Stage time [s] \\\\\n\\hline\n']);
methodOrder = {'olhoff', 'yuksel', 'proposed'};
rank = cellfun(@(k) find(strcmp(methodOrder, k)), {R.method_key});
[~, order] = sortrows([arrayfun(@(r) prod(r.mesh), R(:)), rank(:)]);
prevMesh = [];
for i = order(:).'
    r = R(i);
    [c1, c2, t1, t2, st] = primaryCells(r, '%.2f');
    meshCell = '';
    if ~isequal(r.mesh(:).', prevMesh)
        if ~isempty(prevMesh); fprintf(fid, '\\thinhline\n'); end
        meshCell = sprintf('$%d\\times%d$', r.mesh(1), r.mesh(2));
        prevMesh = r.mesh(:).';
    end
    fprintf(fid, '%s & %s & %s & %s & %s & %s & %s \\\\\n', ...
        meshCell, texEscape(confbench_paper_label(r.method_key)), c1, c2, t1, t2, st);
end
fprintf(fid, '\\hline\n\\end{tabular}\n\\endgroup\n');
fprintf(fid, '\\label{tab:conference-performance}\n\\end{table}\n');

% Olhoff-specific exposure, required alongside the primary table.
olh = R(strcmp({R.method_key}, 'olhoff'));
if ~isempty(olh)
    fprintf(fid, '\n%% Nested-scheme detail for %s:\n', confbench_paper_label('olhoff'));
    for i = 1:numel(olh)
        r = olh(i);
        if isfield(r.counts, 'inner_iterations_per_outer_mean')
            fprintf(fid, '%%   %dx%d: %.2f inner MMA iterations per outer, inner time share %.1f%%\n', ...
                r.mesh(1), r.mesh(2), r.counts.inner_iterations_per_outer_mean, ...
                r.times.inner_time_share_pct);
        end
    end
end

fprintf(fid, '\n%% Scaling caveat: %s\n', cav.sparse_step);
end

% =========================================================================
function writeDetailedCsv(path, R, cav)
%WRITEDETAILEDCSV  Explicit method-specific field names, full precision.
fid = fopen(path, 'w');
c = onCleanup(@() fclose(fid));
fprintf(fid, ['method,method_key,nelx,nely,n_elements,status,status_note,ok,' ...
    'scientific_observation,' ...
    'omega1_native,omega2_native,omega3_native,' ...
    'proposed_stage1_solves,proposed_stage2_iterations,' ...
    'proposed_stage1_time_s,proposed_stage1_reference_eigen_time_s,' ...
    'proposed_stage1_preparation_time_s,proposed_stage2_time_s,' ...
    'yuksel_stage1_iterations,yuksel_stage2_iterations,yuksel_iterations_total,' ...
    'yuksel_stage1_time_s,yuksel_stage2_time_s,' ...
    'olhoff_outer_iterations,olhoff_inner_iterations_total,' ...
    'olhoff_inner_iterations_per_outer_mean,' ...
    'olhoff_outer_time_excluding_inner_s,olhoff_inner_time_total_s,' ...
    'olhoff_inner_time_per_outer_mean_s,olhoff_inner_time_per_inner_iteration_mean_s,' ...
    'olhoff_inner_time_share_pct,olhoff_eigen_time_s,olhoff_gradient_time_s,' ...
    'olhoff_outer_bookkeeping_time_s,' ...
    'olhoff_preset,olhoff_outer_time_total_s,olhoff_outer_time_per_outer_mean_s,' ...
    'olhoff_outer_time_per_outer_median_s,olhoff_outer_time_excluding_inner_per_outer_mean_s,' ...
    'olhoff_eigen_time_per_outer_mean_s,olhoff_gradient_time_per_outer_mean_s,' ...
    'olhoff_total_wall_time_per_outer_s,' ...
    'time1_s,time2_s,stage_time_s,overhead_time_s,total_wall_time_s,' ...
    'timing_accounting_residual_s,timing_accounting_relative_residual,' ...
    'timing_accounting_fail,independent_crosscheck_residual_s,' ...
    'independent_crosscheck_fail,' ...
    'stop_reason,volume,grayness,' ...
    'omega1_common_raw_E1,omega1_common_raw_E2,omega1_common_raw_E3,' ...
    'omega1_native_vs_E1_rel_dev,omega1_native_flag,E1_selected_mode_voidKE,' ...
    'max_ram_mb_DEPRECATED_UNMEASURED\n']);
for i = 1:numel(R)
    r = R(i);
    C = r.counts; T = r.times; A = r.accounting;
    ev = evalOr(r);
    [~, ~, flagged, dev] = omega1Cells(r, cav.omega1_native_flag_tol);
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
    fprintf(fid, '%s,%s,%s,%s,%s,%s,', g(isP,C,'stage1_solves'), g(isP,C,'stage2_iterations'), ...
        g(isP,T,'stage1_time_s'), g(isP,T,'stage1_reference_eigen_time_s'), ...
        g(isP,T,'stage1_preparation_time_s'), g(isP,T,'stage2_time_s'));
    fprintf(fid, '%s,%s,%s,%s,%s,', g(isY,C,'stage1_iterations'), g(isY,C,'stage2_iterations'), ...
        g(isY,C,'iterations_total_generic'), g(isY,T,'stage1_time_s'), g(isY,T,'stage2_time_s'));
    fprintf(fid, '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,', ...
        g(isO,C,'outer_iterations'), g(isO,C,'inner_iterations_total'), ...
        g(isO,C,'inner_iterations_per_outer_mean'), g(isO,T,'outer_time_excluding_inner_s'), ...
        g(isO,T,'inner_time_total_s'), g(isO,T,'inner_time_per_outer_mean_s'), ...
        g(isO,T,'inner_time_per_inner_iteration_mean_s'), g(isO,T,'inner_time_share_pct'), ...
        g(isO,T,'eigen_time_s'), g(isO,T,'gradient_time_s'), g(isO,T,'outer_bookkeeping_time_s'));
    if isO && isfield(r, 'production_preset') && ~isempty(r.production_preset)
        presetCell = csvText(r.production_preset);
    else
        presetCell = 'N/A';
    end
    fprintf(fid, '%s,%s,%s,%s,%s,%s,%s,%s,', presetCell, ...
        g(isO,T,'outer_time_total_s'), g(isO,T,'outer_time_per_outer_mean_s'), ...
        g(isO,T,'outer_time_per_outer_median_s'), g(isO,T,'outer_time_excluding_inner_per_outer_mean_s'), ...
        g(isO,T,'eigen_time_per_outer_mean_s'), g(isO,T,'gradient_time_per_outer_mean_s'), ...
        g(isO,T,'total_wall_time_per_outer_s'));
    fprintf(fid, '%s,%s,%s,%s,%s,', f(T,'time1'), f(T,'time2'), f(T,'stage_time_s'), f(T,'overhead_time_s'), f(T,'total_wall_time_s'));
    fprintf(fid, '%s,%s,%d,%s,%d,', f(A,'timing_accounting_residual_s'), ...
        f(A,'timing_accounting_relative_residual'), logicalOr(A,'timing_accounting_fail'), ...
        f(A,'independent_crosscheck_residual_s'), logicalOr(A,'independent_crosscheck_fail'));
    fprintf(fid, '%s,%s,%s,', csvText(f(r.stopping,'stop_reason')), ...
        f(r.stopping,'volume'), f(r.stopping,'final_grayness'));
    fprintf(fid, '%s,%s,%s,', ev.E1, ev.E2, ev.E3);
    fprintf(fid, '%s,%s,%s,', num(dev), flagText(flagged), num(voidKEOf(r)));
    fprintf(fid, 'NOT_MEASURED\n');
end
end

% =========================================================================
function writeResultsJson(path, cfg, R, scaling, cav)
out = struct();
out.schema = 'conference_performance_benchmark_results/2';
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
    [~, ~, flagged, dev] = omega1Cells(r, cav.omega1_native_flag_tol);
    r.omega1_common_E1 = e1Of(r);
    r.omega1_native_vs_E1_rel_dev = dev;
    r.omega1_native_flag = flagged;
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
fprintf(fid, '- threads: %d\n', manifest.environment.max_num_comp_threads);
fprintf(fid, '- timing schema: `%s`\n\n', confbench_timing_schema().schema_version);

fprintf(fid, '## How to read the table\n\n%s\n\n', cav.table_caption);
fprintf(fid, '%s\n\n', interpretationSentence(R));
fprintf(fid, '%s\n\n', cav.other_column);
fprintf(fid, '%s\n\n', cav.omega1_columns);

writeProposedStage1Section(fid, R, cav);
writeFlaggedOmegaSection(fid, R, cav);

fprintf(fid, '## %s\n\n', confbench_display_name('olhoff'));
fprintf(fid, '%s\n\n', cav.olhoff_label);
fprintf(fid, '> %s\n\n', cav.olhoff);
fprintf(fid, '%s\n\n', cav.olhoff_iteration_counts);

olh = R(strcmp({R.method_key}, 'olhoff'));
if ~isempty(olh)
    fprintf(fid, '### Cost per outer iteration\n\n');
    fprintf(fid, ['Total wall time and per-outer-iteration cost, from the same per-iteration ' ...
        'timers. eig/outer includes FE assembly.\n\n']);
    fprintf(fid, '| Mesh | Outer | Total [s] | Total/outer [s] | Outer excl. inner/outer [s] | eig/outer [s] | Inner/outer [s] | Inner MMA/outer | Per inner it. [s] | Status |\n');
    fprintf(fid, '|---|---|---|---|---|---|---|---|---|---|\n');
    for i = 1:numel(olh)
        r = olh(i); T = r.times; C = r.counts;
        fprintf(fid, '| %dx%d | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n', r.mesh(1), r.mesh(2), ...
            f(C,'outer_iterations','%.6g'), f(T,'total_wall_time_s','%.3f'), ...
            f(T,'total_wall_time_per_outer_s','%.4f'), f(T,'outer_time_excluding_inner_per_outer_mean_s','%.4f'), ...
            f(T,'eigen_time_per_outer_mean_s','%.4f'), f(T,'inner_time_per_outer_mean_s','%.4f'), ...
            f(C,'inner_iterations_per_outer_mean','%.2f'), f(T,'inner_time_per_inner_iteration_mean_s','%.4f'), ...
            r.status);
    end
    fprintf(fid, '\n');
end

fprintf(fid, '## Memory\n\n%s\n\n', cav.memory);

fprintf(fid, '## Scaling\n\n%s\n\n', cav.scaling);
if isfield(scaling, 'fitted') && scaling.fitted
    fprintf(fid, '| Method | C | p | R^2 | points |\n|---|---|---|---|---|\n');
    for i = 1:numel(scaling.methods)
        s = scaling.methods(i);
        fprintf(fid, '| %s | %.6e | %.4f | %.4f | %d |\n', s.method, s.C, s.p, s.R2, s.n);
    end
    fprintf(fid, '\n');
    if isfield(scaling, 'per_outer') && ~isempty(scaling.per_outer.methods)
        fprintf(fid, 'Per-outer-iteration cost, %s:\n\n', scaling.per_outer.model);
        fprintf(fid, '| Method | Quantity | C | p | R^2 | points |\n|---|---|---|---|---|---|\n');
        for i = 1:numel(scaling.per_outer.methods)
            s = scaling.per_outer.methods(i);
            fprintf(fid, '| %s | %s | %.6e | %.4f | %.4f | %d |\n', s.method, s.quantity, s.C, s.p, s.R2, s.n);
        end
        fprintf(fid, '\n');
    end
else
    fprintf(fid, '_No scaling fit was performed for this run: %s_\n\n', scalingReason(scaling));
end
writePerIterationSection(fid, R, cav);

fprintf(fid, '## Results\n\n');
fprintf(fid, '| Method | Mesh | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Stage time [s] | Other [s] | Total wall time [s] | omega1 native | omega1 E1 | Status |\n');
fprintf(fid, '|---|---|---|---|---|---|---|---|---|---|---|---|\n');
for i = 1:numel(R)
    r = R(i);
    [c1, c2, t1, t2, st, ov, tt] = primaryCells(r, '%.3f');
    [nat, e1, flagged] = omega1Cells(r, cav.omega1_native_flag_tol, '%.4f');
    if flagged; nat = [nat ' †']; end
    fprintf(fid, '| %s | %dx%d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n', ...
        r.method, r.mesh(1), r.mesh(2), c1, c2, t1, t2, st, ov, tt, nat, e1, r.status);
end
fprintf(fid, '\n† native omega1 deviates from E1 by more than %g%% (see "Native omega_1 values flagged").\n\n', ...
    100*cav.omega1_native_flag_tol);
end

% =========================================================================
function writeProposedStage1Section(fid, R, cav)
P = R(strcmp({R.method_key}, 'proposed'));
P = P(arrayfun(@(r) isfield(r.times, 'stage1_preparation_time_s'), P));
if isempty(P); return; end
fprintf(fid, '### Proposed Stage 1: eigenanalysis versus preparation\n\n%s\n\n', cav.proposed_preparation);
fprintf(fid, '| Mesh | Initialization incl. eigenanalysis [s] | Time 1 = reference eigenanalysis [s] | Preparation [s] (in Other) | Other [s] | Total [s] |\n');
fprintf(fid, '|---|---|---|---|---|---|\n');
prep = zeros(1, numel(P));
for i = 1:numel(P)
    r = P(i); T = r.times; prep(i) = T.stage1_preparation_time_s;
    fprintf(fid, '| %dx%d | %s | %s | %s | %s | %s |\n', r.mesh(1), r.mesh(2), ...
        f(T,'stage1_time_s','%.3f'), f(T,'time1','%.3f'), f(T,'stage1_preparation_time_s','%.3f'), ...
        f(T,'overhead_time_s','%.3f'), f(T,'total_wall_time_s','%.3f'));
end
[pmax, imax] = max(prep);
rest = prep; rest(imax) = [];
if numel(rest) >= 1
    fprintf(fid, ['\nThe largest preparation value, %.3f s at %dx%d, is the first measured row of its ' ...
        'session; the other %d rows lie between %.3f and %.3f s, independent of the mesh.\n\n'], ...
        pmax, P(imax).mesh(1), P(imax).mesh(2), numel(rest), min(rest), max(rest));
else
    fprintf(fid, '\n');
end
end

% =========================================================================
function writeFlaggedOmegaSection(fid, R, cav)
fl = flaggedRows(R, cav.omega1_native_flag_tol);
fprintf(fid, '### Native omega_1 values flagged\n\n');
if isempty(fl)
    fprintf(fid, 'No native omega_1 deviates from its E1 value by more than %g%%.\n\n', ...
        100*cav.omega1_native_flag_tol);
    return
end
fprintf(fid, ['%d row(s) carry a native omega_1 that deviates from E1 by more than %g%%. ' ...
    'For each, the first three native modes and the E1 selected structural mode are listed; a ' ...
    'cluster of native modes within a few percent of each other, and an E1 mode whose kinetic ' ...
    'energy in elements with rho < 0.1 (void-KE share) is near zero, is the signature of ' ...
    'localized near-void modes in the native model, not of a different structure.\n\n'], ...
    numel(fl), 100*cav.omega1_native_flag_tol);
fprintf(fid, '| Method | Mesh | native omega_1, omega_2, omega_3 | E1 omega_1 | deviation | E1 selected mode void-KE share |\n');
fprintf(fid, '|---|---|---|---|---|---|\n');
for i = 1:numel(fl)
    r = fl(i);
    [~, ~, ~, dev] = omega1Cells(r, cav.omega1_native_flag_tol);
    fprintf(fid, '| %s | %dx%d | %s | %s | %.1f%% | %s |\n', r.method, r.mesh(1), r.mesh(2), ...
        omegaTriple(r, '%.2f'), num(e1Of(r), '%.2f'), 100*dev, num(voidKEOf(r), '%.3f'));
end
fprintf(fid, '\n');
end

% =========================================================================
function writePerIterationSection(fid, R, cav)
M = zeros(0, 2);
for i = 1:numel(R)
    if ~ismember(R(i).mesh(:).', M, 'rows'); M(end+1, :) = R(i).mesh(:).'; end %#ok<AGROW>
end
if isempty(M); return; end
fprintf(fid, '### Per-iteration cost across meshes\n\n%s\n\n', cav.sparse_step);
fprintf(fid, '| Mesh | DOFs | Proposed SIMP [s/it] | Yuksel Stage 1 [s/it] | Yuksel Stage 2 [s/it] | Du-Olhoff eig+assembly [s/outer] | Du-Olhoff inner MMA [s/it] |\n');
fprintf(fid, '|---|---|---|---|---|---|---|\n');
for k = 1:size(M, 1)
    nelx = M(k, 1); nely = M(k, 2);
    p = pickRow(R, 'proposed', M(k, :)); y = pickRow(R, 'yuksel', M(k, :)); o = pickRow(R, 'olhoff', M(k, :));
    fprintf(fid, '| %dx%d | %d | %s | %s | %s | %s | %s |\n', nelx, nely, 2*(nelx+1)*(nely+1), ...
        perIter(p, 'time2', 'count2'), perIter(y, 'time1', 'count1'), perIter(y, 'time2', 'count2'), ...
        rowField(o, 'eigen_time_per_outer_mean_s'), rowField(o, 'inner_time_per_inner_iteration_mean_s'));
end
fprintf(fid, '\n');
end

function r = pickRow(R, key, mesh)
r = [];
for i = 1:numel(R)
    if strcmp(R(i).method_key, key) && isequal(R(i).mesh(:).', mesh(:).'); r = R(i); return; end
end
end

function s = perIter(r, tname, cname)
s = 'N/A';
if isempty(r) || ~isfield(r.times, tname) || ~isfield(r.counts, cname); return; end
n = r.counts.(cname);
if ~isnumeric(n) || ~isfinite(n) || n <= 0; return; end
s = num(r.times.(tname)/n, '%.4f');
end

function s = rowField(r, name)
s = 'N/A';
if isempty(r); return; end
s = f(r.times, name, '%.4f');
end

% =========================================================================
function [c1, c2, t1, t2, st, ov, tt] = primaryCells(r, tfmt)
if nargin < 2; tfmt = '%.9g'; end
c1 = f(r.counts, 'count1', '%.6g');
c2 = f(r.counts, 'count2', '%.6g');
t1 = f(r.times, 'time1', tfmt);
t2 = f(r.times, 'time2', tfmt);
st = f(r.times, 'stage_time_s', tfmt);
ov = f(r.times, 'overhead_time_s', tfmt);
tt = f(r.times, 'total_wall_time_s', tfmt);
end

function [nat, e1, flagged, dev] = omega1Cells(r, tol, fmt)
%OMEGA1CELLS  Native omega_1, common-evaluator E1 omega_1, and the dagger rule.
if nargin < 3; fmt = '%.17g'; end
nat = num(r.omega1_native, fmt);
v1 = e1Of(r);
e1 = num(v1, fmt);
dev = NaN; flagged = false;
if isnumeric(r.omega1_native) && isfinite(r.omega1_native) && isfinite(v1) && v1 > 0
    dev = abs(r.omega1_native - v1)/v1;
    flagged = dev > tol;
end
end

function fl = flaggedRows(R, tol)
keep = false(1, numel(R));
for i = 1:numel(R)
    [~, ~, keep(i)] = omega1Cells(R(i), tol);
end
fl = R(keep);
end

function v = e1Of(r)
%E1OF  The E1 classifier-selected structural omega_1, NaN when the evaluator did not run.
v = NaN;
if isfield(r, 'evaluator') && isstruct(r.evaluator) && isfield(r.evaluator, 'selected_omega_raw_E1')
    x = r.evaluator.selected_omega_raw_E1;
    if isnumeric(x) && ~isempty(x); v = double(x(1)); end
end
end

function v = voidKEOf(r)
%VOIDKEOF  Void kinetic-energy share of the E1 selected mode, NaN when absent.
v = NaN;
if isfield(r, 'evaluator') && isstruct(r.evaluator) && isfield(r.evaluator, 'modal_raw_E1') ...
        && isstruct(r.evaluator.modal_raw_E1) && isfield(r.evaluator.modal_raw_E1, 'selected_voidKE')
    x = r.evaluator.modal_raw_E1.selected_voidKE;
    if isnumeric(x) && ~isempty(x); v = double(x(1)); end
end
end

function s = omegaTriple(r, fmt)
parts = cell(1, 3);
for k = 1:3; parts{k} = num(vecOr(r.omega, k), fmt); end
s = strjoin(parts, ', ');
end

function s = flagText(flagged)
if flagged; s = 'DAGGER'; else; s = ''; end
end

function s = interpretationSentence(R)
keys = unique({R.method_key});
parts = {};
if any(strcmp(keys,'proposed'))
    parts{end+1} = ['Proposed: Count 1 = reference eigenanalysis solves (always 1, ' ...
        'not an optimization iteration), Count 2 = SIMP iterations, Time 1 = ' ...
        'that single eigenanalysis (K0/M0 assembly and the eigensolve, nothing ' ...
        'else; solver preparation is in Other), Time 2 = SIMP.'];
end
if any(strcmp(keys,'yuksel'))
    parts{end+1} = ['Yuksel: Count 1 and Count 2 are the Stage-1 and Stage-2 ' ...
        'iteration counts, Time 1 and Time 2 the corresponding stage times.'];
end
if any(strcmp(keys,'olhoff'))
    olhoffRows = R(strcmp({R.method_key}, 'olhoff'));
    parts{end+1} = [olhoffRows(1).method ': Count 1 = outer iterations, ' ...
        'Count 2 = cumulative nested MMA iterations, Time 1 = outer work ' ...
        'excluding the nested MMA solve (FE assembly, the eigenproblem, ' ...
        'sensitivities, filtering, the design update), Time 2 = nested MMA ' ...
        'total. The two counts are never added.'];
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
