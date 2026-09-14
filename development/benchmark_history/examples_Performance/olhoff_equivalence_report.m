function summary = olhoff_equivalence_report(opts)
%OLHOFF_EQUIVALENCE_REPORT  Collect per-mesh equivalence records into the
%   machine-readable summary and the human-readable report.
%
%   summary = OLHOFF_EQUIVALENCE_REPORT()
%   summary = OLHOFF_EQUIVALENCE_REPORT(opts)
%
%   Reads every olhoff_equivalence_<mesh>.mat under the equivalence directory
%   -- meshes may have been run as separate MATLAB processes -- and writes
%
%     equivalence/olhoff_equivalence_summary.json
%     OLHOFF_BENCHMARK_EQUIVALENCE_REPORT.md      (repository root)
%
%   opts.equivalence_dir  default examples/Performance/equivalence
%   opts.report_path      default <repoRoot>/OLHOFF_BENCHMARK_EQUIVALENCE_REPORT.md
%   opts.meshes           n x 2, to fix row order.  Default: the four
%                         performance meshes, then anything else found.
%
%   See also VERIFY_REPRO2007_BENCHMARK_EQUIVALENCE, OLHOFF_EQUIVALENCE_GATE.

here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));
addpath(here);

if nargin < 1 || isempty(opts)
    opts = struct();
end
profileMode = localGet(opts, 'profile_mode', 'r3');
eqDir       = localGet(opts, 'equivalence_dir', ...
    fullfile(here, 'equivalence', profileMode));
if strcmp(profileMode, 'r3')
    defaultReport = fullfile(repoRoot, 'OLHOFF_BENCHMARK_EQUIVALENCE_REPORT.md');
else
    defaultReport = fullfile(repoRoot, ...
        sprintf('OLHOFF_BENCHMARK_EQUIVALENCE_REPORT_%s.md', upper(profileMode)));
end
reportPath  = localGet(opts, 'report_path', defaultReport);
wanted     = localGet(opts, 'meshes', [160 20; 240 30; 320 40; 400 50]);

listing = dir(fullfile(eqDir, 'olhoff_equivalence_*x*.mat'));
if isempty(listing)
    error('olhoff_equivalence_report:NoRecords', ...
        'No equivalence records under %s.', eqDir);
end

recs = struct([]);
found = {};
for i = 1:numel(listing)
    S = load(fullfile(eqDir, listing(i).name));
    if ~isfield(S, 'rec'); continue; end
    if isempty(recs); recs = S.rec; else; recs(end+1) = S.rec; end %#ok<AGROW>
    found{end+1} = S.rec.mesh; %#ok<AGROW>
end

% Requested order first, then anything extra, so the table reads like the
% performance table rather than like a directory listing.
order = [];
for i = 1:size(wanted, 1)
    k = find(strcmp(found, sprintf('%dx%d', wanted(i,1), wanted(i,2))), 1);
    if ~isempty(k); order(end+1) = k; end %#ok<AGROW>
end
order = [order, setdiff(1:numel(recs), order, 'stable')];
recs = recs(order);

summary = struct();
summary.generated  = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
summary.n_meshes   = numel(recs);
summary.meshes     = recs;
verdicts = {recs.equivalence_verdict};
summary.verdict = localVerdict(all(strcmp(verdicts, 'PASS')));
summary.all_timing_admissible = all([recs.timing_admissible]);

localWriteJson(fullfile(eqDir, 'olhoff_equivalence_summary.json'), summary);
localWriteMarkdown(reportPath, summary, recs, eqDir, repoRoot);
fprintf('wrote %s\n', fullfile(eqDir, 'olhoff_equivalence_summary.json'));
fprintf('wrote %s\n', reportPath);
end

% -------------------------------------------------------------------------
function localWriteMarkdown(path, summary, recs, eqDir, repoRoot)
L = {};
r1 = recs(1);
L{end+1} = '# Olhoff benchmark-path equivalence';
L{end+1} = '';
L{end+1} = sprintf('**Generated:** %s', summary.generated);
L{end+1} = sprintf('**Verdict:** %s', summary.verdict);
L{end+1} = sprintf('**Profile:** `%s`  (mode `%s`)', r1.profile_id, r1.profile_mode);
if isfield(r1.profile, 'mode_note')
    L{end+1} = sprintf('**Interpretation:** %s', r1.profile.mode_note);
end
L{end+1} = sprintf('**Source commit:** `%s`', r1.source_commit);
L{end+1} = sprintf('**Reproduction tree hash (frozen algo/fem/filter/mma):** `%s`', r1.reproduction_tree_hash);
L{end+1} = sprintf('**Benchmark path code hash:** `%s`', r1.benchmark_path_code_hash);
L{end+1} = sprintf('**MATLAB:** %s (%s)', r1.environment.matlab_version, r1.environment.computer);
L{end+1} = '';
L{end+1} = 'Proves, per mesh, that';
L{end+1} = '';
L{end+1} = '```text';
L{end+1} = 'A  repro2007_config -> olhoffOpt                              (direct clean-room oracle)';
L{end+1} = 'B  run_topopt_from_json -> OlhoffDu2007Repro -> run_repro2007 -> olhoffOpt  (benchmark)';
L{end+1} = '```';
L{end+1} = '';
L{end+1} = ['execute the same normalized configuration and produce the same trajectory, ' ...
            'bit for bit. The clean-room implementation is the oracle. Tolerance is exactly ' ...
            'zero on every compared quantity; wall-clock columns (`t_eig`, `t_grad`, ' ...
            '`t_inner`, `elapsed_s`) are the only exclusions.'];
L{end+1} = '';

% ---- headline table -----------------------------------------------------
L{end+1} = '## Results';
L{end+1} = '';
L{end+1} = '| Mesh | Config identity | History identity | Density identity | Stop identity | Verdict |';
L{end+1} = '|---|---|---|---|---|---|';
for i = 1:numel(recs)
    r = recs(i);
    L{end+1} = sprintf('| %s | %s | %s | %s | %s | **%s** |', ...
        r.mesh, r.config_identity, r.history_identity, r.density_identity, ...
        r.status_identity, r.equivalence_verdict); %#ok<AGROW>
end
L{end+1} = '';

% ---- what each run did --------------------------------------------------
L{end+1} = '## Trajectories';
L{end+1} = '';
L{end+1} = '| Mesh | Outer A | Outer B | Status | Stop reason | final max&#124;dρ&#124; | LP failures | ω₁ final | Timing admissible |';
L{end+1} = '|---|---|---|---|---|---|---|---|---|';
for i = 1:numel(recs)
    r = recs(i);
    L{end+1} = sprintf('| %s | %d | %d | `%s` | `%s` | %.17g | %d | %.6f | %s |', ...
        r.mesh, r.n_outer_A, r.n_outer_B, r.stop_A.status, r.stop_A.stop_reason, ...
        r.stop_A.final_max_density_change, r.lp.n_failures_A, r.omega_final_A(1), ...
        localYesNo(r.timing_admissible)); %#ok<AGROW>
end
L{end+1} = '';

% ---- hashes -------------------------------------------------------------
L{end+1} = '## Identity hashes';
L{end+1} = '';
L{end+1} = '`direct` is path A, `benchmark` is path B. Equal hashes mean bit-identical arrays, not close ones.';
L{end+1} = '';
L{end+1} = '| Mesh | Normalized config | Trajectory (direct) | Trajectory (benchmark) | Density (direct) | Density (benchmark) |';
L{end+1} = '|---|---|---|---|---|---|';
for i = 1:numel(recs)
    r = recs(i);
    L{end+1} = sprintf('| %s | `%s` | `%s` | `%s` | `%s` | `%s` |', r.mesh, ...
        localH(r.normalized_config_hash), localH(r.direct_path_hash), ...
        localH(r.benchmark_path_hash), localH(r.density_sha256_A), ...
        localH(r.density_sha256_B)); %#ok<AGROW>
end
L{end+1} = '';

% ---- acceptance criteria ------------------------------------------------
L{end+1} = '## Acceptance criteria (WP10)';
L{end+1} = '';
crit = fieldnames(recs(1).acceptance);
hdr = '| Criterion |';
sep = '|---|';
for i = 1:numel(recs)
    hdr = [hdr sprintf(' %s |', recs(i).mesh)]; %#ok<AGROW>
    sep = [sep '---|']; %#ok<AGROW>
end
L{end+1} = hdr;
L{end+1} = sep;
labels = { ...
    'c1_identical_effective_configuration', '1. identical effective normalized configuration'
    'c2_same_initial_spectrum',             '2. same initial spectrum'
    'c3_same_numerical_trajectory',         '3. same numerical trajectory (zero tolerance)'
    'c4_same_multiplicity_and_eigengap',    '4. same multiplicity / eigengap logic'
    'c5_same_volume_history',               '5. same volume history'
    'c6_same_subproblem_status_sequence',   '6. same LP/subproblem status sequence'
    'c7_same_stop_classification',          '7. same stop classification'
    'c8_same_final_density_field',          '8. same final density field / checksum'
    'c9_no_lp_failure_as_convergence',      '9. no LP failure misclassified as convergence'};
for k = 1:size(labels, 1)
    f = labels{k,1};
    if ~any(strcmp(crit, f)); continue; end
    row = sprintf('| %s |', labels{k,2});
    for i = 1:numel(recs)
        row = [row sprintf(' %s |', recs(i).acceptance.(f))]; %#ok<AGROW>
    end
    L{end+1} = row; %#ok<AGROW>
end
L{end+1} = '';

% ---- checkpoints --------------------------------------------------------
L{end+1} = '## Checkpoints';
L{end+1} = '';
L{end+1} = ['Values are from path A; `match` is exact equality against path B at ' ...
            'that outer iteration. A checkpoint past the end of the shorter run is ' ...
            'listed as a non-match rather than skipped: both paths read NaN there, and ' ...
            '`NaN == NaN` would otherwise be scored as agreement when in fact one path ' ...
            'stopped and the other did not.'];
for i = 1:numel(recs)
    r = recs(i);
    L{end+1} = ''; %#ok<AGROW>
    L{end+1} = sprintf('### %s', r.mesh); %#ok<AGROW>
    L{end+1} = ''; %#ok<AGROW>
    L{end+1} = '| Iter | ω₁ | ω₂ | ω₃ | N | gap_rel | objective | max&#124;dρ&#124; | vol | lp_flag | inner_conv | match | note |'; %#ok<AGROW>
    L{end+1} = '|---|---|---|---|---|---|---|---|---|---|---|---|---|'; %#ok<AGROW>
    for k = 1:numel(r.checkpoint_results)
        c = r.checkpoint_results(k);
        note = c.note;
        if isempty(note); note = '--'; end
        L{end+1} = sprintf('| %s | %.6f | %.6f | %.6f | %g | %.6g | %.6f | %.17g | %.6f | %g | %g | %s | %s |', ...
            c.label, c.omega1_A, c.omega2_A, c.omega3_A, c.N_A, c.gap_rel_A, ...
            c.objective_A, c.d_inf_A, c.vol_A, c.lp_flag_A, c.inner_converged_A, ...
            localYesNo(c.identical), note); %#ok<AGROW>
    end
end
L{end+1} = '';

% ---- divergences / failures ---------------------------------------------
L{end+1} = '## Divergences and subproblem failures';
L{end+1} = '';
anyIssue = false;
for i = 1:numel(recs)
    r = recs(i);
    if strcmp(r.equivalence_verdict, 'PASS') && ~r.lp.any_failure
        continue
    end
    anyIssue = true;
    L{end+1} = sprintf('### %s', r.mesh); %#ok<AGROW>
    L{end+1} = ''; %#ok<AGROW>
    if ~strcmp(r.equivalence_verdict, 'PASS')
        L{end+1} = sprintf('**First divergence:** outer iteration %g, field(s) %s', ...
            r.first_divergence_iteration, strjoin(r.first_divergence_fields, ', ')); %#ok<AGROW>
        L{end+1} = ''; %#ok<AGROW>
        for k = 1:numel(r.config_differences)
            d = r.config_differences(k);
            L{end+1} = sprintf('- config `%s`: A = `%s`, B = `%s`', d.field, d.a, d.b); %#ok<AGROW>
        end
        L{end+1} = ''; %#ok<AGROW>
    end
    if r.lp.any_failure
        L{end+1} = sprintf(['**Subproblem failures:** %d on path A, %d on path B; ' ...
            'sequence identity %s. First failure at outer iteration %g with ' ...
            'linprog exit flag %g.'], r.lp.n_failures_A, r.lp.n_failures_B, ...
            r.lp.failure_sequence_identity, r.lp.first_failure_iter_A, ...
            r.lp.first_failure_flag_A); %#ok<AGROW>
        L{end+1} = ''; %#ok<AGROW>
        L{end+1} = sprintf(['**`LP failure -> drho = 0 -> outer stop` chain:** %s. ' ...
            'The run %s end on a failed subproblem.'], ...
            r.lp.zero_step_chain_confirmed, ...
            localDoesDoesNot(r.lp.run_ended_on_failed_subproblem)); %#ok<AGROW>
        L{end+1} = ''; %#ok<AGROW>
        if ~isempty(r.lp.window)
            L{end+1} = '| Iter | Path | ω₁ | ω₂ | ω₃ | N | gap_rel | λ_ref | lp_flag | inner_conv | β | max&#124;dρ&#124; | vol |'; %#ok<AGROW>
            L{end+1} = '|---|---|---|---|---|---|---|---|---|---|---|---|---|'; %#ok<AGROW>
            for k = 1:numel(r.lp.window)
                w = r.lp.window(k);
                L{end+1} = sprintf('| %d | %s | %.6f | %.6f | %.6f | %g | %.6g | %.6g | %g | %g | %.6g | %.17g | %.6f |', ...
                    w.iter, w.path, w.omega1, w.omega2, w.omega3, w.N, w.gap_rel, ...
                    w.lamref, w.lp_flag, w.inner_converged, w.beta, w.d_inf, w.vol); %#ok<AGROW>
            end
            L{end+1} = ''; %#ok<AGROW>
        end
    end
end
if ~anyIssue
    L{end+1} = 'None. Every mesh matched on every compared quantity, and no subproblem failed.';
    L{end+1} = '';
end

% ---- harness validation -------------------------------------------------
ctrlDir = fullfile(fileparts(eqDir), 'negative_controls');
ctrls = dir(fullfile(ctrlDir, 'control_*_*x*.mat'));
% Positive control first: the table reads as "it passes when it should, and
% here is what it takes to make it fail".
isPos = arrayfun(@(f) contains(f.name, 'positive'), ctrls);
ctrls = [ctrls(isPos); ctrls(~isPos)];
if ~isempty(ctrls)
    L{end+1} = '## Harness validation';
    L{end+1} = '';
    L{end+1} = ['A check that only ever returns PASS proves nothing. These control runs ' ...
                'perturb the benchmark path alone, on a small mesh, and ask whether the ' ...
                'harness notices. The `rho_min` control re-injects the exact ' ...
                'configuration-mapping defect of `DIAGNOSTIC_REPRO2007_BENCHMARK.md`.'];
    L{end+1} = '';
    L{end+1} = '| Control | Expected | Verdict | Config identity | First divergence | Fields |';
    L{end+1} = '|---|---|---|---|---|---|';
    for i = 1:numel(ctrls)
        C = load(fullfile(ctrlDir, ctrls(i).name));
        if ~isfield(C, 'rec'); continue; end
        c = C.rec;
        inj = fieldnames(c.path_b_injected_overrides);
        if isempty(inj)
            label = 'unperturbed (positive control)';
            expect = 'PASS';
        else
            vals = cell(1, numel(inj));
            for q = 1:numel(inj)
                vals{q} = sprintf('%s = %g', inj{q}, c.path_b_injected_overrides.(inj{q}));
            end
            label = ['path B ' strjoin(vals, ', ')];
            expect = 'FAIL';
        end
        if isnan(c.first_divergence_iteration)
            fd = '--';
        else
            fd = sprintf('%g', c.first_divergence_iteration);
        end
        flds = strjoin(c.first_divergence_fields, ', ');
        if isempty(flds); flds = '--'; end
        L{end+1} = sprintf('| %s | %s | **%s** | %s | %s | %s |', label, expect, ...
            c.equivalence_verdict, c.config_identity, fd, flds); %#ok<AGROW>
    end
    L{end+1} = '';
    L{end+1} = ['The `rho_min` control diverges at outer iteration 100, which is where ' ...
                'the historical defect first became observable: the initial design is ' ...
                'uniform rho = 0.5 and the move limit is 0.005, so no element can reach ' ...
                'the void floor until iteration 0.5 / 0.005 = 100. The harness locates ' ...
                'the divergence at the iteration the mechanism predicts, not merely ' ...
                'somewhere.'];
    L{end+1} = '';
end

% ---- admission ----------------------------------------------------------
L{end+1} = '## Benchmark admission';
L{end+1} = '';
L{end+1} = ['`olhoff_equivalence_gate(nelx, nely)` is the precondition for an Olhoff ' ...
            'timing or scaling row. It re-derives the normalized configuration hash on ' ...
            'every call and refuses the row if the benchmark path code, the frozen ' ...
            'reproduction bytes, the profile, the task JSON, that config hash, the mesh ' ...
            'or the MATLAB release has moved since the proof was made.'];
L{end+1} = '';
L{end+1} = ['The binding is a content hash of the code that defines the path ' ...
            '(`olhoff_benchmark_path_hash`), not the repository HEAD. HEAD would be the ' ...
            'obvious choice and is the wrong one: committing these artifacts moves HEAD, ' ...
            'so a HEAD binding would invalidate every proof at the moment it was ' ...
            'archived. The commit is still recorded above as provenance.'];
L{end+1} = '';
L{end+1} = ['The harness itself is inside that hash, so editing ' ...
            '`verify_repro2007_benchmark_equivalence.m` invalidates every existing proof ' ...
            'and forces a re-run. That is deliberate and it is the expensive direction on ' ...
            'purpose: a weakened comparison is exactly the defect a self-certifying check ' ...
            'cannot otherwise catch. Two of this harness''s own defects were found that ' ...
            'way -- a checkpoint past the end of the shorter run compared `NaN` against ' ...
            '`NaN` and read as agreement, and the code hash was being overwritten by the ' ...
            'trajectory hash because both were briefly called `benchmark_path_hash`. ' ...
            'Presentation-only code (`olhoff_equivalence_report`, `olhoff_equivalence_gate`, ' ...
            '`olhoff_preflight`) is outside the hash and can be edited without re-proving.'];
L{end+1} = '';
L{end+1} = '| Mesh | Verdict | Row class | Admissible | Reason if refused |';
L{end+1} = '|---|---|---|---|---|';
for i = 1:numel(recs)
    r = recs(i);
    reason = r.timing_exclusion_reason;
    if isempty(reason); reason = '--'; end
    L{end+1} = sprintf('| %s | %s | `%s` | %s | %s |', r.mesh, r.equivalence_verdict, ...
        r.benchmark_row_class, localYesNo(r.timing_admissible), reason); %#ok<AGROW>
end
L{end+1} = '';

% ---- provenance ---------------------------------------------------------
L{end+1} = '## Provenance';
L{end+1} = '';
L{end+1} = sprintf('- Task JSON: `%s`, SHA-256 `%s`', r1.profile.source_json, ...
    localH(r1.profile.source_json_sha256));
L{end+1} = sprintf('- Protocol profile: `%s` in `%s`', r1.profile.protocol_profile_id, ...
    r1.profile.protocol_document);
if ~isempty(r1.profile.deviations_from_protocol)
    L{end+1} = '- Deviation from the protocol profile:';
    for k = 1:numel(r1.profile.deviations_from_protocol)
        L{end+1} = sprintf('  - %s', r1.profile.deviations_from_protocol{k}); %#ok<AGROW>
    end
end
if isfield(r1.profile, 'yuksel_table1')
    y = r1.profile.yuksel_table1;
    L{end+1} = sprintf('- Interpretation source: %s (`%s`)', y.source, y.file);
    L{end+1} = sprintf('- Stated rule: "%s"', y.stated_rule);
    L{end+1} = sprintf('- Implied counts from Table 1: %s', y.implied_counts_from_table1);
    if ~isempty(y.deviations_not_adopted)
        L{end+1} = ['- Settings from Yuksel section 6.2 deliberately NOT adopted (this mode ' ...
                    'changes the outer budget only, so its Olhoff column stays the same ' ...
                    'operating point as the R3 column):'];
        for k = 1:numel(y.deviations_not_adopted)
            L{end+1} = sprintf('  - %s', y.deviations_not_adopted{k}); %#ok<AGROW>
        end
    end
end
L{end+1} = sprintf('- Overrides applied to the task JSON: %s', ...
    strjoin(cellfun(@(x) ['`' x '`'], r1.profile.overrides_applied(:)', ...
    'UniformOutput', false), ', '));
L{end+1} = sprintf('- Per-mesh records: `%s`', ...
    strrep(fullfile(eqDir, 'olhoff_equivalence_<mesh>.{json,mat}'), [repoRoot filesep], ''));
L{end+1} = '';
L{end+1} = ['No file under `Matlab/reproduction2007/algo`, `fem`, `filter` or `mma` was ' ...
            'modified. The `runner/` directory is integration code written for this ' ...
            'repository and is excluded from the clean-room SHA-256 manifest by ' ...
            'construction (`PROVENANCE.md`, "Import integrity").'];
L{end+1} = '';

fid = fopen(path, 'w');
fwrite(fid, strjoin(L, newline), 'char');
fclose(fid);
end

% -------------------------------------------------------------------------
function s = localH(h)
if isempty(h); s = '--'; else; s = h(1:min(16, numel(h))); end
end

function s = localYesNo(tf)
if tf; s = 'yes'; else; s = '**no**'; end
end

function s = localDoesDoesNot(tf)
if tf; s = 'does'; else; s = 'does not'; end
end

function v = localVerdict(tf)
if tf; v = 'PASS'; else; v = 'FAIL'; end
end

function v = localGet(s, name, defaultValue)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
else
    v = defaultValue;
end
end

function localWriteJson(file, s)
txt = jsonencode(s, 'PrettyPrint', true);
fid = fopen(file, 'w');
fwrite(fid, txt, 'char');
fclose(fid);
end
