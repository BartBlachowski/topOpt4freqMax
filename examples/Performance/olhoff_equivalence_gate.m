function gate = olhoff_equivalence_gate(nelx, nely, opts)
%OLHOFF_EQUIVALENCE_GATE  Precondition for admitting an Olhoff timing row.
%
%   gate = OLHOFF_EQUIVALENCE_GATE(nelx, nely)
%   gate = OLHOFF_EQUIVALENCE_GATE(nelx, nely, opts)
%
%   Answers one question: has the benchmark-dispatched Olhoff path been PROVED
%   equivalent to the direct clean-room reproduction, for THIS mesh, at the
%   profile, source commit, frozen implementation and MATLAB release that are
%   in force right now?
%
%   Until that is true, the mesh's timing, scaling and speedup numbers are
%   measurements of an unverified execution path and must not enter a
%   performance table.
%
%   OUTPUT gate
%   NOTE  gate.benchmark_path_code_hash identifies the CODE that defines the
%   path.  It is a different quantity from the per-mesh record's
%   benchmark_path_hash, which is the content hash of path B's trajectory.
%
%     .status      'PASS'    equivalence proved and still valid
%                  'FAIL'    equivalence was run and did not hold
%                  'MISSING' never run for this mesh
%                  'STALE'   run, but something it was tied to has changed
%     .admissible  true only for 'PASS' AND a run that is not itself a
%                  SOLVER_FAILURE.  Two different reasons a row can be
%                  refused, kept separate on purpose.
%     .row_class   'ADMISSIBLE' | 'INVALID_BENCHMARK_PATH' |
%                  'INVALID_SOLVER_STATUS' | 'UNVERIFIED_BENCHMARK_PATH'
%     .reasons     cell array, one line per failed binding
%     .record      the equivalence record, when one was found
%
%   THE PREFLIGHT IS A MANIFEST, NOT A RERUN
%   ----------------------------------------
%   Proving equivalence costs two full solves per mesh, so it is not repeated
%   inside every timing campaign.  Instead the recorded proof is bound to five
%   identities, and ANY of them changing invalidates it:
%
%     1. benchmark path hash    -- the dispatcher, the reproduction's runner/,
%                                  the profile, the task JSON, and the harness
%                                  that made the proof (OLHOFF_BENCHMARK_PATH_HASH)
%     2. reproduction tree hash -- the frozen algo/fem/filter/mma bytes
%     3. profile id + task JSON hash    -- what was configured
%     4. normalized config hash -- what OLHOFFOPT actually received
%     5. mesh, and the MATLAB release   -- linprog and eigs live here
%
%   Binding 4 is recomputed from scratch on every call, so a change to the
%   dispatcher's mapping invalidates the gate without anyone remembering to.
%
%   The repository HEAD is RECORDED but is deliberately not one of the
%   bindings: committing the proof itself moves HEAD, which would invalidate
%   every proof at the moment it was archived.  Binding 1 is both stable under
%   that and tighter -- see OLHOFF_BENCHMARK_PATH_HASH.
%
%   opts.profile_mode      'r3' (default) or 'yuksel_table1'.  A proof is
%                          valid only for the profile it was made against: the
%                          two differ in max_outer, which is part of the
%                          normalized configuration hash.
%   opts.equivalence_dir   default examples/Performance/equivalence/<profile_mode>
%
%   See also VERIFY_REPRO2007_BENCHMARK_EQUIVALENCE, PERFORMANCE_BENCHMARK_PROFILE,
%            OLHOFF_EQUIVALENCE_REPORT.

here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));
addpath(here);
addpath(fullfile(repoRoot, 'Matlab', 'reproduction2007', 'runner'));

if nargin < 3 || isempty(opts)
    opts = struct();
end
profileMode = localGet(opts, 'profile_mode', 'r3');
eqDir = localGet(opts, 'equivalence_dir', ...
    fullfile(here, 'equivalence', profileMode));

gate = struct();
gate.mesh         = sprintf('%dx%d', nelx, nely);
gate.profile_mode = profileMode;
gate.nelx       = nelx;
gate.nely       = nely;
gate.checked    = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
gate.reasons    = {};
gate.record     = struct([]);
gate.artifact   = fullfile(eqDir, sprintf('olhoff_equivalence_%dx%d.mat', nelx, nely));

% ---- what is in force right now -----------------------------------------
[data, profileId, profileMeta] = performance_benchmark_profile(nelx, nely, profileMode);
gate.profile_id            = profileId;
gate.source_json_sha256    = profileMeta.source_json_sha256;
gate.reproduction_tree_hash = repro2007_tree_hash();
gate.benchmark_path_code_hash = olhoff_benchmark_path_hash();
gate.matlab_release        = ['R' version('-release')];
gate.source_commit         = localGitOut(repoRoot, 'rev-parse HEAD');
gate.working_tree_clean    = isempty(localGitOut(repoRoot, 'status --porcelain'));
gate.normalized_config_hash = localExpectedConfigHash(data);

if exist(gate.artifact, 'file') ~= 2
    gate.status = 'MISSING';
    gate.admissible = false;
    gate.row_class = 'UNVERIFIED_BENCHMARK_PATH';
    gate.reasons = {sprintf(['no equivalence proof for mesh %s at profile mode ' ...
        '''%s''.  Run: verify_repro2007_benchmark_equivalence([%d %d], ' ...
        'struct(''profile_mode'', ''%s''))'], gate.mesh, profileMode, nelx, nely, ...
        profileMode)};
    return
end

S = load(gate.artifact);
rec = S.rec;
gate.record = rec;
gate.recorded_verdict = rec.equivalence_verdict;

% ---- bindings ------------------------------------------------------------
stale = {};
localBind = @(name, was, now_) sprintf('%s changed: proof used %s, now %s', ...
    name, localShort(was), localShort(now_));

if ~localHas(rec, 'benchmark_path_code_hash')
    stale{end+1} = ['proof predates benchmark-path code hashing and cannot be ' ...
        'bound to the code that produced it'];
elseif ~strcmp(rec.benchmark_path_code_hash, gate.benchmark_path_code_hash)
    stale{end+1} = localBind('benchmark path code', ...
        rec.benchmark_path_code_hash, gate.benchmark_path_code_hash);
end
if ~strcmp(rec.reproduction_tree_hash, gate.reproduction_tree_hash)
    stale{end+1} = localBind('reproduction tree hash', ...
        rec.reproduction_tree_hash, gate.reproduction_tree_hash);
end
if ~strcmp(rec.profile_id, gate.profile_id)
    stale{end+1} = localBind('profile id', rec.profile_id, gate.profile_id);
end
if ~strcmp(rec.profile.source_json_sha256, gate.source_json_sha256)
    stale{end+1} = localBind('task JSON', ...
        rec.profile.source_json_sha256, gate.source_json_sha256);
end
if ~strcmp(rec.normalized_config_hash, gate.normalized_config_hash)
    stale{end+1} = localBind('normalized config hash', ...
        rec.normalized_config_hash, gate.normalized_config_hash);
end
if ~strcmp(rec.environment.matlab_release, gate.matlab_release)
    stale{end+1} = localBind('MATLAB release', ...
        rec.environment.matlab_release, gate.matlab_release);
end
if rec.nelx ~= nelx || rec.nely ~= nely
    stale{end+1} = sprintf('mesh mismatch: proof is for %s, asked for %s', ...
        rec.mesh, gate.mesh);
end
if localFlag(rec, 'is_diagnostic_prefix')
    stale{end+1} = 'proof was a shortened diagnostic prefix run, not a full run';
end
if localFlag(rec, 'is_negative_control')
    stale{end+1} = 'proof was a negative-control run with path B deliberately perturbed';
end
% Working-tree cleanliness is recorded, not gated: every file that can change
% this execution path is covered by gate.benchmark_path_code_hash and
% gate.reproduction_tree_hash, both computed from the bytes on disk.  An
% unrelated dirty file is not a reason to refuse a proved row; a dirty
% dispatcher already is, through the hash.
gate.source_commit_at_proof = rec.source_commit;
gate.source_commit_drifted  = ~strcmp(rec.source_commit, gate.source_commit);

% ---- verdict -------------------------------------------------------------
if ~isempty(stale)
    gate.status = 'STALE';
    gate.admissible = false;
    gate.row_class = 'UNVERIFIED_BENCHMARK_PATH';
    gate.reasons = stale;
    return
end

if ~strcmp(rec.equivalence_verdict, 'PASS')
    gate.status = 'FAIL';
    gate.admissible = false;
    gate.row_class = 'INVALID_BENCHMARK_PATH';
    gate.reasons = {sprintf(['benchmark path did not reproduce the clean-room ' ...
        'trajectory: config=%s history=%s density=%s stop=%s, first divergence ' ...
        'at outer iteration %g'], rec.config_identity, rec.history_identity, ...
        rec.density_identity, rec.status_identity, rec.first_divergence_iteration)};
    return
end

gate.status = 'PASS';
if localFlag(rec, 'timing_admissible')
    gate.admissible = true;
    gate.row_class = 'ADMISSIBLE';
else
    % The paths agree, so the benchmark path is verified -- but the run they
    % agree on ended on a failed subproblem.  Its iteration count and wall time
    % measure a solver failure, not the method, and are not a timing result.
    gate.admissible = false;
    gate.row_class = 'INVALID_SOLVER_STATUS';
    gate.reasons = {rec.timing_exclusion_reason};
end
end

% -------------------------------------------------------------------------
function h = localExpectedConfigHash(data)
%LOCALEXPECTEDCONFIGHASH  What OLHOFFOPT would receive today, hashed.
%   Builds the configuration but does not run anything.
guard = repro2007_paths(); %#ok<NASGU>
cfg = repro2007_direct_cfg(data, false);
[~, h] = repro2007_normalized_config(cfg);
end

function out = localGitOut(repoRoot, args)
[st, txt] = system(sprintf('cd %s && git %s', repoRoot, args));
if st == 0
    out = strtrim(txt);
else
    out = '';
end
end

function s = localShort(v)
s = char(string(v));
if numel(s) > 20
    s = [s(1:16) '...'];
end
end

function tf = localHas(s, f)
tf = isfield(s, f) && ~isempty(s.(f));
end

function tf = localFlag(s, f)
tf = isfield(s, f) && ~isempty(s.(f)) && logical(s.(f));
end

function v = localGet(s, name, defaultValue)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
else
    v = defaultValue;
end
end
