function report = verify_repro2007_benchmark_equivalence(meshes, opts)
%VERIFY_REPRO2007_BENCHMARK_EQUIVALENCE  Prove the benchmark's Olhoff path is
%   the clean-room Du-Olhoff 2007 reproduction, and not merely similar to it.
%
%   report = VERIFY_REPRO2007_BENCHMARK_EQUIVALENCE()
%   report = VERIFY_REPRO2007_BENCHMARK_EQUIVALENCE(meshes)
%   report = VERIFY_REPRO2007_BENCHMARK_EQUIVALENCE(meshes, opts)
%
%   For every mesh, runs the SAME normalized configuration twice:
%
%     PATH A (oracle)      repro2007_config -> olhoffOpt
%                          The clean-room implementation, called directly.  No
%                          benchmark dispatcher is involved.
%
%     PATH B (benchmark)   run_topopt_from_json -> OlhoffDu2007Repro
%                          -> run_repro2007 -> olhoffOpt
%                          The exact entry point performance_comparison.m uses.
%
%   and requires them to agree BIT FOR BIT on the whole trajectory.
%
%   WHY BIT-EXACT IS THE RIGHT TEST
%   -------------------------------
%   Every source of run-to-run variation in this implementation is pinned: the
%   initial design is uniform rho = 0.5, EIGSOLVE supplies a fixed deterministic
%   ARPACK start vector, linprog runs dual-simplex-highs, and OLHOFFOPT pins
%   BLAS to cfg.threads.  The two paths execute the same frozen files.  There
%   is therefore no mechanism by which they may legitimately differ, and any
%   difference is a finding rather than a tolerance to be widened.  The default
%   tolerance is exactly zero and OPTS.tolerance exists only so that a future
%   caller who needs a non-zero one has to write it down.
%
%   Timings (t_eig, t_grad, t_inner, elapsed_s) are excluded from the identity
%   test: they are wall-clock measurements, not algorithm state.  They are the
%   ONLY excluded columns, and they are listed explicitly below.
%
%   WHAT IS COMPARED, PER OUTER ITERATION
%   -------------------------------------
%   omega (all J+1 modes), omega1..omega3, N, gap_abs, gap_rel, bimodal, multJ,
%   beta, objective, vol, rV, d_inf, move_saturated, n_inner, cum_inner,
%   inner_converged, lp_flag, degen_hits, grayness, d_rms -- plus the
%   normalized configuration at the OLHOFFOPT boundary, the stop
%   classification, and the SHA-256 of the final density field.
%
%   INPUTS
%     meshes   n x 2 [nelx nely].  Default: the four performance meshes
%              160x20, 240x30, 320x40, 400x50.
%     opts     struct, all optional
%       .checkpoints        outer iterations to tabulate.  Default
%                           [1 10 25 50 100 200 400]; the final iteration is
%                           always added.
%       .max_outer_override scalar; shortens both paths equally.  For smoke
%                           tests only -- a shortened run is recorded as
%                           is_diagnostic_prefix = true and must not be used
%                           as an admission gate.
%       .tolerance          struct field -> absolute tolerance, for fields that
%                           cannot be exact.  Default: empty (all exact).
%       .out_dir            artifact directory.  Default
%                           examples/Performance/equivalence.
%       .write_artifacts    default true
%       .profile_mode       'r3' (default) or 'yuksel_table1' -- which
%                           PERFORMANCE_BENCHMARK_PROFILE interpretation to
%                           prove.  Artifacts land in
%                           equivalence/<profile_mode>/.
%       .summary_name       basename of the run-level summary.  Give each
%                           process a distinct one when meshes are run in
%                           parallel, then collect with
%                           OLHOFF_EQUIVALENCE_REPORT.
%       .inject_path_b_overrides
%                           NEGATIVE CONTROL.  optimization.repro2007 fields
%                           applied to the BENCHMARK path only, so that the
%                           harness can be shown to fail when the two paths
%                           really do differ.  Any run with this set is marked
%                           is_negative_control and is never admissible.
%       .verbose            echo each solver's own per-iteration table.
%                           Default false; the identity test does not read
%                           stdout, and 2 x 1600 lines per mesh is noise.
%
%   OUTPUT
%     report.meshes(i)  one record per mesh, schema documented in
%                       OLHOFF_BENCHMARK_EQUIVALENCE_REPORT.md
%     report.verdict    'PASS' only if every mesh passed every check
%
%   See also PERFORMANCE_BENCHMARK_PROFILE, REPRO2007_NORMALIZED_CONFIG,
%            OLHOFF_EQUIVALENCE_GATE, RUN_REPRO2007, OLHOFFOPT.

% ---- paths ---------------------------------------------------------------
here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));
addpath(here);
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'Matlab', 'reproduction2007', 'runner'));

if nargin < 1 || isempty(meshes)
    meshes = [160 20; 240 30; 320 40; 400 50];
end
if nargin < 2 || isempty(opts)
    opts = struct();
end
checkpoints      = localGet(opts, 'checkpoints', [1 10 25 50 100 200 400]);
maxOuterOverride = localGet(opts, 'max_outer_override', []);
tolerance        = localGet(opts, 'tolerance', struct());
outDir           = localGet(opts, 'out_dir', ...
    fullfile(here, 'equivalence', localGet(opts, 'profile_mode', 'r3')));
writeArtifacts   = localGet(opts, 'write_artifacts', true);
verbose          = localGet(opts, 'verbose', false);
summaryName      = localGet(opts, 'summary_name', 'olhoff_equivalence_summary.json');
% Which interpretation of the benchmark is being proved.  Each profile is a
% DIFFERENT normalized configuration and therefore needs its own proof: proving
% the 1600-outer R3 profile says nothing about the 200-outer diagnostic one.
% Artifacts are kept in per-profile subdirectories so the two can never be
% mistaken for each other.
profileMode      = localGet(opts, 'profile_mode', 'r3');
% NEGATIVE CONTROL ONLY.  Fields applied to optimization.repro2007 on the
% BENCHMARK path and not on the oracle path, to prove that this harness detects
% a divergence rather than merely reporting PASS.  A run with this set is
% recorded as is_negative_control and can never be an admission gate.
injectPathB      = localGet(opts, 'inject_path_b_overrides', struct());

if writeArtifacts && exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

% ---- run-level identity --------------------------------------------------
[treeHash, treeEntries] = repro2007_tree_hash();
[pathHash, pathEntries] = olhoff_benchmark_path_hash();
env = localEnvironment(repoRoot);

fprintf('\n');
fprintf('================================================================\n');
fprintf(' Olhoff benchmark-path equivalence\n');
fprintf('   A  repro2007_config -> olhoffOpt                     (oracle)\n');
fprintf('   B  run_topopt_from_json -> run_repro2007 -> olhoffOpt (benchmark)\n');
fprintf('----------------------------------------------------------------\n');
fprintf(' source commit        : %s%s\n', env.source_commit, env.dirty_suffix);
fprintf(' reproduction tree    : %s  (%d frozen files)\n', treeHash, size(treeEntries, 1));
fprintf(' benchmark path code  : %s  (%d files)\n', pathHash, size(pathEntries, 1));
fprintf(' matlab               : %s\n', env.matlab_version);
fprintf(' profile mode         : %s\n', profileMode);
fprintf(' meshes               : %s\n', localMeshList(meshes));
if ~isempty(maxOuterOverride)
    fprintf(' *** DIAGNOSTIC PREFIX: max_outer forced to %d -- NOT an admission gate ***\n', ...
        maxOuterOverride);
end
if ~isempty(fieldnames(injectPathB))
    fprintf(' *** NEGATIVE CONTROL: path B perturbed (%s) -- NOT an admission gate ***\n', ...
        strjoin(fieldnames(injectPathB)', ', '));
end
fprintf('================================================================\n');

report = struct();
report.generated       = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
report.source_commit   = env.source_commit;
report.working_tree_clean = env.clean;
report.environment     = env;
report.reproduction_tree_hash = treeHash;
report.reproduction_tree_files = size(treeEntries, 1);
report.benchmark_path_code_hash = pathHash;
report.benchmark_path_files = pathEntries(:, 1)';
report.profile_mode = profileMode;
report.is_diagnostic_prefix = ~isempty(maxOuterOverride);
report.is_negative_control  = ~isempty(fieldnames(injectPathB));
report.meshes = struct([]);

for i = 1:size(meshes, 1)
    nelx = meshes(i, 1);
    nely = meshes(i, 2);
    fprintf('\n---- mesh %dx%d ------------------------------------------\n', nelx, nely);
    rec = localOneMesh(nelx, nely, checkpoints, maxOuterOverride, tolerance, ...
        verbose, env, treeHash, pathHash, injectPathB, profileMode);
    rec.is_diagnostic_prefix = report.is_diagnostic_prefix;
    rec.is_negative_control  = report.is_negative_control;
    if isempty(report.meshes)
        report.meshes = rec;
    else
        report.meshes(end+1) = rec; %#ok<AGROW>
    end
    if writeArtifacts
        stem = fullfile(outDir, sprintf('olhoff_equivalence_%dx%d', nelx, nely));
        localWriteJson([stem '.json'], rec);
        % The .mat is what OLHOFF_EQUIVALENCE_REPORT reads when it collects
        % meshes that were run as separate MATLAB processes: a JSON round trip
        % would quietly reshape empty arrays and one-element struct arrays,
        % and this record is evidence, not a display format.
        save([stem '.mat'], 'rec', '-v7');
        fprintf('  artifact : %s.{json,mat}\n', stem);
    end
end

verdicts = {report.meshes.equivalence_verdict};
if all(strcmp(verdicts, 'PASS'))
    report.verdict = 'PASS';
else
    report.verdict = 'FAIL';
end

fprintf('\n================================================================\n');
fprintf(' OVERALL: %s\n', report.verdict);
for i = 1:numel(report.meshes)
    m = report.meshes(i);
    fprintf('   %-9s config=%-4s history=%-4s density=%-4s stop=%-4s -> %s\n', ...
        m.mesh, m.config_identity, m.history_identity, m.density_identity, ...
        m.status_identity, m.equivalence_verdict);
end
fprintf('================================================================\n\n');

if writeArtifacts
    localWriteJson(fullfile(outDir, summaryName), report);
end
end

% =========================================================================
function rec = localOneMesh(nelx, nely, checkpoints, maxOuterOverride, ...
        tolerance, verbose, env, treeHash, pathHash, injectPathB, profileMode)

[data, profileId, profileMeta] = performance_benchmark_profile(nelx, nely, profileMode);
if ~isempty(maxOuterOverride)
    data.optimization.repro2007.max_outer = maxOuterOverride;
end

rec = struct();
rec.mesh                    = sprintf('%dx%d', nelx, nely);
rec.nelx                    = nelx;
rec.nely                    = nely;
rec.n_elements              = nelx * nely;
rec.profile_id              = profileId;
rec.profile_mode            = profileMode;
rec.profile                 = profileMeta;
rec.path_b_injected_overrides = injectPathB;
rec.source_commit           = env.source_commit;
rec.reproduction_tree_hash  = treeHash;
% NOT the same thing as rec.benchmark_path_hash below.  This one identifies the
% CODE that defines the benchmark execution path (and the harness that proved
% it); that one is the content hash of path B's TRAJECTORY.  They were briefly
% given the same name, and the second assignment silently won.
rec.benchmark_path_code_hash = pathHash;
rec.environment             = env;

% ---- PATH A: the oracle, called directly ----------------------------------
fprintf('  [A] direct clean-room path ... ');
tA = tic;
[cfgA, resA] = localPathA(data, verbose);
fprintf('%d outer, %.1f s\n', resA.nOuter, toc(tA));
histA = repro2007_history(resA);
stopA = localStoppingFromRes(resA, cfgA);
xA = resA.rho(:);

% ---- PATH B: through the benchmark entry point ---------------------------
fprintf('  [B] benchmark-dispatched path ... ');
tB = tic;
dataB = data;
dataB.optimization.approach = 'OlhoffDu2007Repro';
injected = fieldnames(injectPathB);
for q = 1:numel(injected)
    dataB.optimization.repro2007.(injected{q}) = injectPathB.(injected{q});
end
% Reporting-only.  record_history never reaches cfg: RUN_REPRO2007 consumes it
% after OLHOFFOPT has returned, to decide whether to reshape res.hist into the
% repository schema.  It cannot perturb the trajectory, and the normalized
% configuration below is unaffected by it.
if ~isfield(dataB, 'benchmark'); dataB.benchmark = struct(); end
dataB.benchmark.record_history = true;
if verbose
    dataB.postprocessing.visualize_live = true;   % the only lever on cfg.verbose
end
[xB, omegaB, tIterB, nIterB, ~, ~, telemetryB] = run_topopt_from_json(dataB); %#ok<ASGLU>
fprintf('%d outer, %.1f s\n', nIterB, toc(tB));
histB = telemetryB.history;
cfgB  = telemetryB.solver_config;
xB    = xB(:);

% =====================================================================
% 1. NORMALIZED CONFIGURATION IDENTITY  (WP2)
% =====================================================================
if isempty(fieldnames(cfgB))
    error('verify_repro2007_benchmark_equivalence:NoSolverConfig', ...
        ['The benchmark path returned no telemetry.solver_config.  The ' ...
         'equivalence check cannot compare configurations it cannot see; ' ...
         'run_topopt_from_json must forward info.cfg for this dispatch case.']);
end
[ncA, hashA, textA, hashedFields] = repro2007_normalized_config(cfgA);
[ncB, hashB, textB] = repro2007_normalized_config(cfgB);
allDiff = localStructDiff(ncA, ncB);
% Split the differences by whether they can move a number.  `verbose` legitimately
% differs when the harness silences the direct path, and failing the gate on it
% would train a reader to ignore the gate.  Everything else is numerical and
% fails hard.
isNumericalDiff = arrayfun(@(d) any(strcmp(d.field, hashedFields)), allDiff);
cfgDiff = allDiff(isNumericalDiff);
rec.normalized_config_hash_A = hashA;
rec.normalized_config_hash_B = hashB;
rec.normalized_config_hash   = hashA;
rec.normalized_config        = ncA;
rec.config_differences       = cfgDiff;
rec.config_reporting_differences = allDiff(~isNumericalDiff);
rec.config_identity          = localVerdict(isempty(cfgDiff) && strcmp(hashA, hashB));
fprintf('  config identity  : %s  (%s)\n', rec.config_identity, hashA(1:16));
for k = 1:numel(cfgDiff)
    fprintf('      DIFFERS  %-22s A=%s  B=%s\n', cfgDiff(k).field, ...
        cfgDiff(k).a, cfgDiff(k).b);
end
for k = 1:numel(rec.config_reporting_differences)
    d = rec.config_reporting_differences(k);
    fprintf('      (reporting-only, not hashed)  %-12s A=%s  B=%s\n', ...
        d.field, d.a, d.b);
end
rec.normalized_config_text_sha256 = sha256_hex([textA newline textB]);

% =====================================================================
% 2. TRAJECTORY IDENTITY  (WP3, WP5)
% =====================================================================
cmp = localCompareHistories(histA, histB, tolerance);
rec.n_outer_A = cmp.n_outer_a;
rec.n_outer_B = cmp.n_outer_b;
rec.trajectory_overlap = cmp.overlap;
rec.length_mismatch = cmp.length_mismatch;
rec.compared_fields          = cmp.fields;
rec.excluded_fields          = cmp.excluded;
rec.field_results            = cmp.results;
rec.first_divergence_iteration = cmp.first_divergence_iteration;
rec.first_divergence_fields    = cmp.first_divergence_fields;
rec.history_identity = localVerdict(cmp.identical);
% Trajectory content hashes -- one per path, over every compared field.
rec.direct_path_hash    = cmp.hash_a;
rec.benchmark_path_hash = cmp.hash_b;
fprintf('  history identity : %s  (nA=%d nB=%d, %d fields compared)\n', ...
    rec.history_identity, rec.n_outer_A, rec.n_outer_B, numel(cmp.fields));
if ~strcmp(rec.history_identity, 'PASS')
    fprintf('      first divergence at outer iteration %d, field(s): %s\n', ...
        cmp.first_divergence_iteration, strjoin(cmp.first_divergence_fields, ', '));
end

% ---- initial spectrum, called out separately (WP10 item 2) --------------
rec.initial_spectrum_A = localCol(histA, 'omega', 1);
rec.initial_spectrum_B = localCol(histB, 'omega', 1);
rec.initial_spectrum_identity = localVerdict( ...
    isequaln(rec.initial_spectrum_A, rec.initial_spectrum_B));

% ---- checkpoints (WP5) ---------------------------------------------------
rec.checkpoint_results = localCheckpoints(histA, histB, checkpoints);
rec.checkpoint_identity = localVerdict(all([rec.checkpoint_results.identical]));

% =====================================================================
% 3. FINAL DENSITY IDENTITY  (WP5)
% =====================================================================
rec.density_sha256_A = sha256_hex(xA);
rec.density_sha256_B = sha256_hex(xB);
rec.density_n_A = numel(xA);
rec.density_n_B = numel(xB);
sameSize = numel(xA) == numel(xB);
rec.density_identity = localVerdict(sameSize && isequaln(xA, xB) && ...
    strcmp(rec.density_sha256_A, rec.density_sha256_B));
if sameSize
    rec.density_max_abs_diff = max(abs(xA - xB));
    rec.density_n_differing  = sum(xA ~= xB);
else
    rec.density_max_abs_diff = NaN;
    rec.density_n_differing  = NaN;
end
rec.density_checksum = sprintf('n=%d;sum=%.17g;l2=%.17g;min=%.17g;max=%.17g', ...
    numel(xA), sum(xA), norm(xA), min(xA), max(xA));
fprintf('  density identity : %s  (%s)\n', rec.density_identity, rec.density_sha256_A(1:16));

% =====================================================================
% 4. STOP CLASSIFICATION IDENTITY  (WP7, WP10 item 7)
% =====================================================================
stopB = telemetryB.stopping;
rec.stop_A = localStopRecord(stopA, resA.nOuter);
rec.stop_B = localStopRecord(stopB, nIterB);
stopDiff = localStructDiff(rec.stop_A, rec.stop_B);
rec.stop_differences = stopDiff;
rec.status_identity  = localVerdict(isempty(stopDiff));
fprintf('  stop identity    : %s  (A=%s/%s  B=%s/%s)\n', rec.status_identity, ...
    rec.stop_A.status, rec.stop_A.stop_reason, rec.stop_B.status, rec.stop_B.stop_reason);

% ---- final frequencies, reported but never the pass criterion ----------
rec.omega_final_A = localToVec3(resA.omega(:));
rec.omega_final_B = omegaB(:).';
rec.omega_final_identity = localVerdict(isequaln(rec.omega_final_A, rec.omega_final_B));

% =====================================================================
% 5. SUBPROBLEM / LP FAILURE FORENSICS  (WP6)
% =====================================================================
rec.lp = localLpForensics(histA, histB, cfgA);
if rec.lp.any_failure
    fprintf('  LP FAILURE       : %d failed solve(s), first at outer iteration %d (flag %g)\n', ...
        rec.lp.n_failures_A, rec.lp.first_failure_iter_A, rec.lp.first_failure_flag_A);
    fprintf('  zero-step chain  : LP failure -> drho = 0 -> outer break : %s\n', ...
        rec.lp.zero_step_chain_confirmed);
end

% =====================================================================
% 6. VERDICT  (WP10)
% =====================================================================
checks = {rec.config_identity, rec.history_identity, rec.checkpoint_identity, ...
          rec.initial_spectrum_identity, rec.density_identity, rec.status_identity};
if all(strcmp(checks, 'PASS'))
    rec.equivalence_verdict = 'PASS';
else
    rec.equivalence_verdict = 'FAIL';
end
rec.acceptance = localAcceptance(rec, cmp);
rec.timing_admissible = strcmp(rec.equivalence_verdict, 'PASS') && ...
    ~strcmp(rec.stop_A.status, 'SOLVER_FAILURE') && ...
    isempty(maxOuterOverride) && isempty(fieldnames(injectPathB));
if ~rec.timing_admissible
    if ~isempty(maxOuterOverride) || ~isempty(fieldnames(injectPathB))
        rec.timing_exclusion_reason = ['diagnostic run: max_outer was shortened ' ...
            'and/or path B was deliberately perturbed.  Not an admission gate.'];
        rec.benchmark_row_class = 'DIAGNOSTIC_ONLY';
    elseif strcmp(rec.equivalence_verdict, 'PASS')
        rec.timing_exclusion_reason = ['SOLVER_FAILURE: the two paths agree, but ' ...
            'the run ended on a failed subproblem, so its iteration count and ' ...
            'wall time do not measure the method converging.'];
        rec.benchmark_row_class = 'INVALID_SOLVER_STATUS';
    else
        rec.timing_exclusion_reason = ['equivalence FAILED: the benchmark-dispatched ' ...
            'path did not reproduce the direct clean-room trajectory.'];
        rec.benchmark_row_class = 'INVALID_BENCHMARK_PATH';
    end
else
    rec.timing_exclusion_reason = '';
    rec.benchmark_row_class = 'ADMISSIBLE';
end
fprintf('  VERDICT          : %s   (timing admissible: %d, class %s)\n', ...
    rec.equivalence_verdict, rec.timing_admissible, rec.benchmark_row_class);
end

% =========================================================================
% PATH A -- the oracle
% =========================================================================
function [cfg, res] = localPathA(data, verbose)
%LOCALPATHA  repro2007_config -> olhoffOpt, with no dispatcher in between.
%
%   The configuration comes from REPRO2007_DIRECT_CFG, which reads the task
%   profile itself and never consults run_topopt_from_json's mapping table.  If
%   both paths shared that mapping, the comparison could not detect a mapping
%   defect, which is the exact class of defect it exists to detect.

guard = repro2007_paths(); %#ok<NASGU>   restored when this function returns
repro2007_assert_identity();

% OLHOFFOPT pins BLAS threads and does not restore them; do not leak that.
threadsBefore = maxNumCompThreads();
restoreThreads = onCleanup(@() maxNumCompThreads(threadsBefore)); %#ok<NASGU>

cfg = repro2007_direct_cfg(data, verbose);
res = olhoffOpt(cfg);
end

% =========================================================================
% comparison machinery
% =========================================================================
function cmp = localCompareHistories(hA, hB, tolerance)
%LOCALCOMPAREHISTORIES  Field-by-field, iteration-by-iteration.
%
%   Two kinds of column live in this history and they cannot be compared the
%   same way:
%
%     per-iteration  one value (or one column) per outer iteration.  Compared
%                    over the OVERLAP of the two runs, so that "where did they
%                    part ways" has an answer even when one run is shorter.
%     run-level      a scalar summarising the whole run (n_modes_recorded,
%                    k_mult).  Compared directly; a difference is real but has
%                    no iteration index, and must not be reported as a
%                    divergence at iteration 1.
%
%   A length difference is itself a divergence.  If the overlap is identical
%   but the runs have different lengths, the first divergence is the first
%   iteration one of them did not reach.

% Wall-clock columns are measurements of the machine, not state of the
% algorithm.  They are the only exclusions, and they are named here so that
% the exclusion list is reviewable rather than implicit.
excluded = {'t_eig', 't_grad', 't_inner', 'elapsed_s', 'meta'};

if isempty(fieldnames(hA)) || isempty(fieldnames(hB))
    error('verify_repro2007_benchmark_equivalence:NoHistory', ...
        ['One of the paths returned no per-iteration history.  Set ' ...
         'benchmark.record_history = true for the dispatched path.']);
end

nA = localNOuter(hA);
nB = localNOuter(hB);
nOv = min(nA, nB);

fieldsA = setdiff(fieldnames(hA), excluded, 'stable');
fieldsB = setdiff(fieldnames(hB), excluded, 'stable');
common  = intersect(fieldsA, fieldsB, 'stable');
onlyA   = setdiff(fieldsA, fieldsB);
onlyB   = setdiff(fieldsB, fieldsA);

results = struct('field', {}, 'kind', {}, 'identical', {}, 'max_abs_diff', {}, ...
    'n_differing', {}, 'first_differing_iteration', {}, 'tolerance', {}, ...
    'hash_a', {}, 'hash_b', {}, 'note', {});

firstDiv = Inf;
firstDivFields = {};
allIdentical = isempty(onlyA) && isempty(onlyB) && (nA == nB);

for i = 1:numel(common)
    f = common{i};
    a = hA.(f);
    b = hB.(f);
    tol = localGet(tolerance, f, 0);

    r = struct('field', f, 'kind', '', 'identical', false, 'max_abs_diff', NaN, ...
        'n_differing', NaN, 'first_differing_iteration', NaN, 'tolerance', tol, ...
        'hash_a', '', 'hash_b', '', 'note', '');

    if ischar(a) || ischar(b)
        r.kind = 'char';
        r.identical = isequal(a, b);
        allIdentical = allIdentical && r.identical;
        results(end+1) = r; %#ok<AGROW>
        continue
    end

    [av, kindA] = localTrim(a, nA, nOv);
    [bv, kindB] = localTrim(b, nB, nOv);
    if ~strcmp(kindA, kindB)
        r.kind = 'mixed';
        r.note = sprintf('shape disagrees: A is %s, B is %s', kindA, kindB);
        allIdentical = false;
        results(end+1) = r; %#ok<AGROW>
        continue
    end
    r.kind = kindA;

    % Hash the FULL untrimmed column, so the recorded hash identifies the whole
    % run rather than the part that happened to overlap.
    r.hash_a = sha256_hex(double(a));
    r.hash_b = sha256_hex(double(b));

    if ~isequal(size(av), size(bv))
        r.note = sprintf('size mismatch after trim: %s vs %s', ...
            mat2str(size(av)), mat2str(size(bv)));
        allIdentical = false;
        results(end+1) = r; %#ok<AGROW>
        continue
    end

    av = double(av);
    bv = double(bv);
    d = abs(av - bv);
    bothNaN = isnan(av) & isnan(bv);
    d(bothNaN) = 0;
    differing = ~(d <= tol) & ~bothNaN;
    r.n_differing = sum(differing(:));
    if isempty(d)
        r.max_abs_diff = 0;
    else
        r.max_abs_diff = max(d(:));
    end
    r.identical = (r.n_differing == 0);

    if ~r.identical && strcmp(r.kind, 'per_iteration')
        % Columns are outer iterations for 2-D fields, elements for vectors.
        if isvector(differing)
            idx = find(differing, 1);
        else
            idx = find(any(differing, 1), 1);
        end
        r.first_differing_iteration = idx;
        if idx < firstDiv
            firstDiv = idx;
            firstDivFields = {f};
        elseif idx == firstDiv
            firstDivFields{end+1} = f; %#ok<AGROW>
        end
    elseif ~r.identical
        r.note = 'run-level scalar: differs, but carries no iteration index';
    end

    allIdentical = allIdentical && r.identical;
    results(end+1) = r; %#ok<AGROW>
end

% A length difference is a divergence in its own right.  If the two runs agreed
% everywhere they both reached, they parted at the first iteration only one of
% them took.
lengthMismatch = (nA ~= nB);
if lengthMismatch && ~isfinite(firstDiv)
    firstDiv = nOv + 1;
    firstDivFields = {'n_outer'};
end

cmp = struct();
cmp.fields    = common(:).';
cmp.excluded  = excluded;
cmp.only_in_a = onlyA(:).';
cmp.only_in_b = onlyB(:).';
cmp.results   = results;
cmp.identical = allIdentical;
cmp.n_outer_a = nA;
cmp.n_outer_b = nB;
cmp.overlap   = nOv;
cmp.length_mismatch = lengthMismatch;
if isfinite(firstDiv)
    cmp.first_divergence_iteration = firstDiv;
else
    cmp.first_divergence_iteration = NaN;
end
cmp.first_divergence_fields = firstDivFields;

% One hash per path over the whole compared trajectory, in a fixed field order.
cmp.hash_a = localHistoryHash(hA, common);
cmp.hash_b = localHistoryHash(hB, common);
end

function [v, kind] = localTrim(a, n, nOv)
%LOCALTRIM  Restrict a history column to the overlap, and say what it is.
%
%   A column is per-iteration if it has one entry (or one column) per outer
%   iteration of its own run.  Anything scalar is run-level.  Anything else is
%   left alone and labelled, rather than guessed at.
if isscalar(a)
    v = a;  kind = 'run_level_scalar';
elseif isvector(a) && numel(a) == n
    v = a(1:nOv);  kind = 'per_iteration';
elseif ~isvector(a) && size(a, 2) == n
    v = a(:, 1:nOv);  kind = 'per_iteration';
else
    v = a;  kind = 'other';
end
end

function h = localHistoryHash(H, fields)
parts = cell(numel(fields), 1);
for i = 1:numel(fields)
    v = H.(fields{i});
    if ischar(v)
        parts{i} = sprintf('%s=char:%s', fields{i}, sha256_hex(v));
    else
        parts{i} = sprintf('%s=%s', fields{i}, sha256_hex(double(v)));
    end
end
h = sha256_hex(strjoin(parts, newline));
end

function out = localCheckpoints(hA, hB, checkpoints)
%LOCALCHECKPOINTS  Tabulate the quantities WP3 names, at the requested outer
%   iterations plus the final one.

nA = localNOuter(hA);
nB = localNOuter(hB);
n  = min(nA, nB);
k = unique([checkpoints(:).', nA, nB]);
k = k(k >= 1 & k <= max(nA, nB));

out = struct('iter', {}, 'label', {}, 'in_range', {}, 'identical', {}, ...
    'omega1_A', {}, 'omega1_B', {}, 'omega2_A', {}, 'omega2_B', {}, ...
    'omega3_A', {}, 'omega3_B', {}, 'N_A', {}, 'N_B', {}, ...
    'gap_rel_A', {}, 'gap_rel_B', {}, 'beta_A', {}, 'beta_B', {}, ...
    'objective_A', {}, 'objective_B', {}, 'd_inf_A', {}, 'd_inf_B', {}, ...
    'vol_A', {}, 'vol_B', {}, 'lp_flag_A', {}, 'lp_flag_B', {}, ...
    'inner_converged_A', {}, 'inner_converged_B', {}, ...
    'n_inner_A', {}, 'n_inner_B', {}, 'cum_inner_A', {}, 'cum_inner_B', {}, ...
    'differing_fields', {}, 'note', {});

names = {'omega1','omega2','omega3','N','gap_rel','beta','objective', ...
         'd_inf','vol','lp_flag','inner_converged','n_inner','cum_inner'};

for i = 1:numel(k)
    it = k(i);
    r = struct();
    r.iter = it;
    if it == nA && it == nB
        r.label = sprintf('%d (final, both)', it);
    elseif it == nA
        r.label = sprintf('%d (final A)', it);
    elseif it == nB
        r.label = sprintf('%d (final B)', it);
    else
        r.label = sprintf('%d', it);
    end
    r.in_range = it <= n;
    differing = {};
    for j = 1:numel(names)
        f = names{j};
        va = localAt(hA, f, it);
        vb = localAt(hB, f, it);
        r.([f '_A']) = va;
        r.([f '_B']) = vb;
        if ~isequaln(va, vb)
            differing{end+1} = f; %#ok<AGROW>
        end
    end
    r.differing_fields = differing;
    % A checkpoint past the end of the shorter run reads NaN on BOTH sides, and
    % isequaln(NaN, NaN) is true -- so without the in_range guard an iteration
    % one path never reached would report as agreement.  It is a disagreement:
    % one path stopped and the other did not.
    r.identical = isempty(differing) && r.in_range;
    if ~r.in_range
        if it > nA
            r.note = sprintf('path A ran only %d outer iterations', nA);
        else
            r.note = sprintf('path B ran only %d outer iterations', nB);
        end
    else
        r.note = '';
    end
    out(end+1) = r; %#ok<AGROW>
end
end

function v = localAt(H, f, it)
if ~isfield(H, f)
    v = NaN;
    return
end
a = H.(f);
if it >= 1 && it <= numel(a)
    v = double(a(it));
else
    v = NaN;
end
end

function c = localCol(H, f, k)
if ~isfield(H, f) || size(H.(f), 2) < k
    c = [];
else
    c = H.(f)(:, k).';
end
end

function n = localNOuter(H)
if isempty(fieldnames(H)) || ~isfield(H, 'N')
    n = 0;
else
    n = numel(H.N);
end
end

% =========================================================================
% LP forensics (WP6)
% =========================================================================
function lp = localLpForensics(hA, hB, cfg)
lp = struct();
convA = logical(hA.inner_converged);
convB = logical(hB.inner_converged);
failA = find(~convA);
failB = find(~convB);

lp.n_failures_A = numel(failA);
lp.n_failures_B = numel(failB);
lp.failure_iters_A = failA;
lp.failure_iters_B = failB;
lp.any_failure = ~isempty(failA) || ~isempty(failB);
lp.failure_sequence_identity = localVerdict(isequal(failA, failB));

if isempty(failA)
    lp.first_failure_iter_A = NaN;
    lp.first_failure_flag_A = NaN;
else
    lp.first_failure_iter_A = failA(1);
    lp.first_failure_flag_A = hA.lp_flag(failA(1));
end
if isempty(failB)
    lp.first_failure_iter_B = NaN;
    lp.first_failure_flag_B = NaN;
else
    lp.first_failure_iter_B = failB(1);
    lp.first_failure_flag_B = hB.lp_flag(failB(1));
end

% The causal chain the diagnostic has to settle:
%   linprog fails -> innerLoopLP returns drho = zeros -> max|drho| = 0
%   -> olhoffOpt's `dxOuter < tolOuter` break fires -> run ends.
% Each link is checked against what was recorded, not assumed.
zeroStep = false;
if ~isempty(failA)
    k = failA(1);
    zeroStep = hA.d_inf(k) == 0;
end
lp.zero_step_at_first_failure = zeroStep;
lastIsFailure = ~isempty(convA) && ~convA(end);
lp.run_ended_on_failed_subproblem = lastIsFailure;
if lastIsFailure && hA.d_inf(end) == 0 && hA.d_inf(end) < cfg.tolOuter
    lp.zero_step_chain_confirmed = 'CONFIRMED';
elseif lastIsFailure
    lp.zero_step_chain_confirmed = 'PARTIAL';
else
    lp.zero_step_chain_confirmed = 'NOT_OBSERVED';
end

% Window around the first failure, both paths, with everything WP6 asks for.
lp.window = struct('iter', {}, 'path', {}, 'omega1', {}, 'omega2', {}, ...
    'omega3', {}, 'N', {}, 'gap_rel', {}, 'lamref', {}, 'lp_flag', {}, ...
    'inner_converged', {}, 'beta', {}, 'objective', {}, 'd_inf', {}, ...
    'vol', {}, 'degen_hits', {}, 'multJ', {});
% Anchor the window on the EARLIEST failure seen on either path.  Anchoring on
% path A alone would produce an empty window in the one case that matters most:
% a failure that happens only on the benchmark path.
anchors = [failA(:); failB(:)];
if ~isempty(anchors)
    k = min(anchors);
    lo = max(1, k - 6);
    hi = k + 2;
    for it = lo:hi
        if it <= numel(convA)
            lp.window(end+1) = localWindowRow(hA, it, 'A'); %#ok<AGROW>
        end
        if it <= numel(convB)
            lp.window(end+1) = localWindowRow(hB, it, 'B'); %#ok<AGROW>
        end
    end
end
end

function r = localWindowRow(H, it, tag)
r = struct();
r.iter = it;
r.path = tag;
r.omega1 = localAt(H, 'omega1', it);
r.omega2 = localAt(H, 'omega2', it);
r.omega3 = localAt(H, 'omega3', it);
r.N = localAt(H, 'N', it);
r.gap_rel = localAt(H, 'gap_rel', it);
% lamref is the quantity innerLoopLP divides its whole constraint system by:
% lamref = lam(1) = omega1^2 at that iteration.  A collapsing omega1 is what
% makes the normalized LP unsolvable, so it is recorded explicitly.
r.lamref = localAt(H, 'omega1', it)^2;
r.lp_flag = localAt(H, 'lp_flag', it);
r.inner_converged = localAt(H, 'inner_converged', it);
r.beta = localAt(H, 'beta', it);
r.objective = localAt(H, 'objective', it);
r.d_inf = localAt(H, 'd_inf', it);
r.vol = localAt(H, 'vol', it);
r.degen_hits = localAt(H, 'degen_hits', it);
r.multJ = localAt(H, 'multJ', it);
end

% =========================================================================
% small helpers
% =========================================================================
function s = localStoppingFromRes(res, cfg)
%LOCALSTOPPINGFROMRES  Path A's stop record.
%
%   Deliberately the SAME classifier the dispatched path is classified by
%   (RUN_REPRO2007 calls REPRO2007_STOPPING too).  If the harness reimplemented
%   the precedence rule, a stop-classification difference could be an artifact
%   of the harness rather than a finding about the paths.
s = repro2007_stopping(res, cfg);
end

function d = localStructDiff(a, b)
d = struct('field', {}, 'a', {}, 'b', {});
fa = fieldnames(a);
fb = fieldnames(b);
for i = 1:numel(fa)
    f = fa{i};
    if ~isfield(b, f)
        d(end+1) = struct('field', f, 'a', localShow(a.(f)), 'b', '<absent>'); %#ok<AGROW>
    elseif ~isequaln(a.(f), b.(f))
        d(end+1) = struct('field', f, 'a', localShow(a.(f)), 'b', localShow(b.(f))); %#ok<AGROW>
    end
end
for i = 1:numel(fb)
    f = fb{i};
    if ~isfield(a, f)
        d(end+1) = struct('field', f, 'a', '<absent>', 'b', localShow(b.(f))); %#ok<AGROW>
    end
end
end

function s = localShow(v)
if ischar(v)
    s = v;
elseif isempty(v)
    s = '[]';
elseif islogical(v) && isscalar(v)
    if v, s = 'true'; else, s = 'false'; end
elseif isnumeric(v) && isscalar(v)
    s = sprintf('%.17g', v);
elseif isnumeric(v) || islogical(v)
    s = mat2str(v(:).', 17);
    if numel(s) > 200, s = [s(1:197) '...']; end
else
    s = sprintf('<%s>', class(v));
end
end

function r = localStopRecord(s, nIter)
r = struct();
r.n_outer                = nIter;
r.stop_reason            = localGet(s, 'stop_reason', 'N/A');
r.status                 = localGet(s, 'status', 'N/A');
r.native_stop_reason     = localGet(s, 'native_stop_reason', 'N/A');
r.native_break_taken     = double(localGet(s, 'native_break_taken', NaN));
r.final_max_density_change = localGetNum(s, 'final_max_density_change');
r.final_inner_converged  = double(localGetNum(s, 'final_inner_converged'));
r.final_lp_flag          = localGetNum(s, 'final_lp_flag');
r.subproblem_failed      = double(localGet(s, 'subproblem_failed', NaN));
r.n_subproblem_failures  = localGetNum(s, 'n_subproblem_failures');
r.final_multiplicity     = localGetNum(s, 'final_multiplicity');
r.convergence_tolerance  = localGetNum(s, 'convergence_tolerance');
end

function v = localGetNum(s, f)
if isstruct(s) && isfield(s, f) && ~isempty(s.(f))
    v = double(s.(f));
else
    v = NaN;
end
end

function a = localAcceptance(rec, cmp)
%LOCALACCEPTANCE  The nine WP10 criteria, each answered separately.
a = struct();
a.c1_identical_effective_configuration = rec.config_identity;
a.c2_same_initial_spectrum             = rec.initial_spectrum_identity;
a.c3_same_numerical_trajectory         = rec.history_identity;
a.c4_same_multiplicity_and_eigengap    = localFieldVerdict(cmp, {'N','gap_rel','gap_abs','bimodal','multJ'});
a.c5_same_volume_history               = localFieldVerdict(cmp, {'vol','rV'});
a.c6_same_subproblem_status_sequence   = localFieldVerdict(cmp, {'lp_flag','inner_converged','n_inner','cum_inner'});
a.c7_same_stop_classification          = rec.status_identity;
a.c8_same_final_density_field          = rec.density_identity;
a.c9_no_lp_failure_as_convergence      = localVerdict( ...
    ~(strcmp(rec.stop_A.status, 'CONVERGED') && rec.stop_A.subproblem_failed == 1) && ...
    ~(strcmp(rec.stop_B.status, 'CONVERGED') && rec.stop_B.subproblem_failed == 1));
end

function v = localFieldVerdict(cmp, names)
ok = true;
found = false;
for i = 1:numel(cmp.results)
    if any(strcmp(cmp.results(i).field, names))
        found = true;
        ok = ok && cmp.results(i).identical;
    end
end
v = localVerdict(ok && found);
end

function v = localVerdict(tf)
if tf, v = 'PASS'; else, v = 'FAIL'; end
end

function v = localToVec3(w)
v = NaN(1, 3);
k = min(3, numel(w));
v(1:k) = w(1:k);
end

function v = localGet(s, name, defaultValue)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
else
    v = defaultValue;
end
end

function s = localMeshList(meshes)
parts = cell(1, size(meshes, 1));
for i = 1:size(meshes, 1)
    parts{i} = sprintf('%dx%d', meshes(i,1), meshes(i,2));
end
s = strjoin(parts, ', ');
end

function env = localEnvironment(repoRoot)
env = struct();
env.matlab_version = version();
env.matlab_release = ['R' version('-release')];
env.computer = computer();
env.repo_root = repoRoot;
[st, out] = system(sprintf('cd %s && git rev-parse HEAD', repoRoot));
if st == 0
    env.source_commit = strtrim(out);
else
    env.source_commit = 'UNKNOWN';
end
[st2, out2] = system(sprintf('cd %s && git status --porcelain', repoRoot));
env.clean = (st2 == 0) && isempty(strtrim(out2));
if env.clean
    env.dirty_suffix = '';
else
    env.dirty_suffix = ' (working tree dirty)';
end
[st3, out3] = system(sprintf('cd %s && git rev-parse --abbrev-ref HEAD', repoRoot));
if st3 == 0
    env.branch = strtrim(out3);
else
    env.branch = 'UNKNOWN';
end
end

function localWriteJson(file, s)
txt = jsonencode(s, 'PrettyPrint', true);
fid = fopen(file, 'w');
fwrite(fid, txt, 'char');
fclose(fid);
end
