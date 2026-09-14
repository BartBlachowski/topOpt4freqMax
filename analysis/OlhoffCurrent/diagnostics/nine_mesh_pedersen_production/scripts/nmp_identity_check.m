function S = nmp_identity_check(mode, outJson, lockPath)
%NMP_IDENTITY_CHECK  Parts 1 and 4 of the nine-mesh Pedersen production campaign.
%   RESOLVES, NEVER SOLVES, the nine production configurations.
%
%   mode 'identity'  Part 1: fail-closed campaign identity (HEAD, +impl tree,
%                    SOURCE_MANIFEST, production preset, nine config hashes
%                    against CAMPAIGN_IDENTITY.json AND NINE_MESH_CONFIGS.json,
%                    each mesh resolved independently).
%   mode 'runner'    Part 4: the same checks through the exact resolution calls
%                    performance_comparison.m makes (confbench_method_config with
%                    the campaign outputDir) and the exact call olhoffcurrent_run
%                    makes before solving (olhoffcurrent_config(nelx,nely,'Preset',
%                    olhoffcurrent_preset(mcfg.olhoff_preset).name)), plus the
%                    campaign lock (runner edit pinned, tracked-dirty set pinned).
%
%   The MATLAB path is built the way performance_comparison.m builds it.

here = fileparts(mfilename('fullpath'));
D    = fileparts(here);                                  % .../nine_mesh_pedersen_production
oc   = fileparts(fileparts(D));                          % analysis/OlhoffCurrent
repo = fileparts(fileparts(oc));
gate = fullfile(oc, 'diagnostics', 'postmerge_campaign_gate');
perf = fullfile(repo, 'examples', 'Performance');

restoredefaultpath;
addpath(here);
addpath(perf); addpath(fullfile(perf, 'conference_bench'));
addpath(fullfile(repo, 'tools', 'Matlab'));
addpath(fullfile(repo, 'analysis', 'three_method_parametric_study'));
addpath(oc);
scrub = olhoffcurrent_scrub_forbidden_paths(repo);

NAME = 'duOlhoffPedersenAdaptiveBoxSensitivityFiltered';
M = [160 20; 240 30; 320 40; 400 50; 480 60; 560 70; 640 80; 720 90; 800 100];
EXPECT_HEAD = 'b21483b158f58e05e7b56957f2fbe8e1d2891395';
EXPECT_TREE = '4ba9a3ae10881344a0e60f2b8a8976c5ec9ceccf966bc2a3da4f37fb5aebffbf';
EXPECT_MANI = 'aee44aa9172a8105314bbb9475e28bed8a53d4d154824a8748abf64a208b1836';

idFile  = fullfile(gate, 'CAMPAIGN_IDENTITY.json');
cfgFile = fullfile(gate, 'NINE_MESH_CONFIGS.json');
ID  = jsondecode(fileread(idFile));
NMC = jsondecode(fileread(cfgFile));

S = struct();
S.schema = 'nmp_identity_check/1';
S.mode = mode;
S.when = char(datetime('now', 'TimeZone', 'local', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
S.matlab = version();
S.solved = false;
S.path_scrub_removed = scrub;
S.frozen_sources = struct( ...
    'campaign_identity', relp(idFile, repo), 'campaign_identity_sha256', olhoffcurrent_sha256_file(idFile), ...
    'nine_mesh_configs', relp(cfgFile, repo), 'nine_mesh_configs_sha256', olhoffcurrent_sha256_file(cfgFile));

% ---- repository -----------------------------------------------------------
S.git.head = gitOut(repo, 'rev-parse HEAD');
S.git.branch = gitOut(repo, 'rev-parse --abbrev-ref HEAD');
S.git.tracked_changes = splitLines(gitOut(repo, 'diff --name-only HEAD'));
S.git.status_porcelain = splitLines(gitOut(repo, 'status --porcelain=v1'));

% ---- implementation -------------------------------------------------------
guard = olhoffcurrent_paths(); %#ok<NASGU>
man = olhoffcurrent_source_manifest();
S.impl = struct('tree_sha256', man.treeHash, 'n_files', man.nFiles, 'manifest_ok', man.ok, ...
    'mismatches', {man.mismatches}, 'missing', {man.missing}, 'extra', {man.extra}, ...
    'artifacts_ignored', {man.artifactsIgnored}, ...
    'source_manifest_file_sha256', olhoffcurrent_sha256_file(fullfile(oc, 'SOURCE_MANIFEST.json')));
S.impl.which_olhoffSolve = which('olhoffSolve');
S.impl.which_mmasub = which('mmasub');

prod = olhoffcurrent_production_preset();
S.preset = struct('required', NAME, 'recorded_production', prod.name, ...
    'benchmark_selected', confbench_olhoff_preset(), ...
    'campaign_identity_production', ID.production.preset);

% ---- runner lock (Part 4 only) -------------------------------------------
if strcmp(mode, 'runner')
    L = jsondecode(fileread(lockPath));
    S.lock = struct('path', relp(lockPath, repo), 'sha256', olhoffcurrent_sha256_file(lockPath));
    runner = fullfile(repo, 'examples', 'Performance', 'performance_comparison.m');
    S.runner = struct('path', relp(runner, repo), 'sha256', olhoffcurrent_sha256_file(runner), ...
        'lock_edited_sha256', L.runner.edited_sha256, 'output_dir', L.output_root_abs);
    outDirForConfig = L.output_root_abs;
    allowedDirty = cellstr(L.allowed_tracked_changes);
else
    outDirForConfig = '';
    allowedDirty = {};
end

% ---- nine meshes, each resolved independently -----------------------------
rows = olh.config.schema(); rows = rows(:, 1);
vals = cell(numel(rows), size(M, 1));
meshes = struct([]);
for i = 1:size(M, 1)
    nelx = M(i, 1); nely = M(i, 2); tag = sprintf('%dx%d', nelx, nely);
    fz_id  = ID.nine_config_hashes.(matlab.lang.makeValidName(tag));
    fz_nmc = NMC.configs(strcmp({NMC.configs.mesh}, tag)).config_hash;

    % route R: exactly what performance_comparison.m calls for its methodConfigs
    [mc, profId, prof] = confbench_method_config('olhoff', nelx, nely, outDirForConfig);
    % route S: exactly what olhoffcurrent_run resolves immediately before olhoffSolve
    pr = olhoffcurrent_preset(mc.olhoff_preset);
    cfgS = olhoffcurrent_config(nelx, nely, 'Preset', pr.name);
    % route A: the OlhoffCurrent API with the frozen preset name
    cfgA = olhoffcurrent_config(nelx, nely, 'Preset', NAME);

    hR  = prof.effective_config_hash;
    hRc = olhoffcurrent_config_hash(mc.canonical);
    hS  = olhoffcurrent_config_hash(cfgS);
    hA  = olhoffcurrent_config_hash(cfgA);

    % the hashed content itself, reproduced and re-hashed so the dump is provably it
    lines = hashLines(cfgS, rows);
    hDump = sha256str(strjoin(lines, newline));
    dumpDir = fullfile(D, 'evidence', 'resolved_configs');
    if exist(dumpDir, 'dir') ~= 7; mkdir(dumpDir); end
    if strcmp(mode, 'identity')
        fid = fopen(fullfile(dumpDir, [tag '.txt']), 'w');
        fprintf(fid, '%s\n', strjoin(lines, newline)); fclose(fid);
    end

    for r = 1:numel(rows); vals{r, i} = olh.config.getPath(cfgS, rows{r}); end
    g = @(p) olh.config.getPath(cfgS, p);
    b = g('domain.b');
    m = struct();
    m.mesh = tag; m.nelx = nelx; m.nely = nely; m.NE = nelx*nely;
    m.frozen_hash_campaign_identity = fz_id;
    m.frozen_hash_nine_mesh_configs = fz_nmc;
    m.hash_runner_methodconfig_profile = hR;
    m.hash_runner_methodconfig_canonical = hRc;
    m.hash_solve_route = hS;
    m.hash_api_route = hA;
    m.hash_of_dumped_rows = hDump;
    m.runner_profile_id = profId;
    m.runner_preset = mc.olhoff_preset;
    % cfg.provenance.preset is the UPSTREAM preset olh.config.resolve was given;
    % the OlhoffCurrent name is stamped separately by olhoffcurrent_config.
    m.provenance_preset_upstream = cfgS.provenance.preset;
    m.provenance_preset_olhoffcurrent = cfgS.provenance.olhoffCurrentPreset;
    m.stiffness_model = g('material.stiffness.model');
    m.linearBelow = g('material.stiffness.linearBelow');
    m.p = g('material.stiffness.p');
    m.p_continuation = g('material.stiffness.continuation.enabled');
    m.mass_model = g('material.mass.model');
    m.mass_q = g('material.mass.q');
    m.mass_continuation = g('material.mass.continuation.enabled');
    m.filter_type = g('filter.type');
    m.radiusPhysical = g('filter.radiusPhysical');
    m.radiusElements_row = g('filter.radiusElements');
    m.domain_b = b;
    m.radius_over_b = g('filter.radiusPhysical')/b;
    m.radiusElements_derived = g('filter.radiusPhysical')/(b/nely);
    m.projection = g('projection.enabled');
    m.move_policy = g('move.policy');
    m.move_initial = g('move.initial');
    m.move_minimum = g('move.minimum');
    m.adaptive_grow = g('move.adaptive.grow');
    m.adaptive_shrink = g('move.adaptive.shrink');
    m.continuation_signal = g('move.continuation.signal');
    m.stop_rule = g('stop.rule');
    m.stop_norm = g('stop.norm');
    m.stop_toleranceRule = g('stop.toleranceRule');
    m.stop_tolerance = g('stop.tolerance');
    m.stop_tolerance_rule_value = 0.05*sqrt(nelx*nely/3200);
    m.guard_settledMove = g('stop.guards.settledMove');
    m.guard_ladderExhausted = g('stop.guards.ladderExhausted');
    m.guard_maxDesignChange = g('stop.guards.maxDesignChange');
    m.guard_boxInactiveFraction = g('stop.guards.boxInactiveFraction');
    m.inner_type = g('optimizer.inner.type');
    m.inner_variable = g('optimizer.inner.variable');
    m.inner_variant = g('optimizer.inner.variant');
    m.inner_tolerance = g('optimizer.inner.tolerance');
    m.inner_maxIterations = g('optimizer.inner.maxIterations');
    m.multiplicity_method = g('multiplicity.method');
    m.multiplicity_tolerance = g('multiplicity.tolerance');
    m.maxCluster = g('eigen.maxCluster');
    m.eigen_solver = g('eigen.solver');
    m.design_minimum = g('design.minimum');
    m.volumeFraction = g('design.volumeFraction');
    m.maxOuter = g('runtime.maxOuter');
    m.singleThread = g('runtime.singleThread');
    m.diagnostics = g('runtime.diagnostics');
    m.verbose = g('runtime.verbose');

    m.checks = struct( ...
        'hash_equals_campaign_identity', strcmp(hS, fz_id), ...
        'hash_equals_nine_mesh_configs', strcmp(hS, fz_nmc), ...
        'runner_route_hashes_agree', strcmp(hR, hS) && strcmp(hRc, hS) && strcmp(hA, hS), ...
        'dumped_rows_rehash_to_config_hash', strcmp(hDump, hS), ...
        'preset', strcmp(mc.olhoff_preset, NAME) && strcmp(cfgS.provenance.olhoffCurrentPreset, NAME) && ...
            strcmp(profId, NAME) && strcmp(cfgS.provenance.preset, ID.production.upstream_preset), ...
        'stiffness_pedersen', strcmp(m.stiffness_model, 'pedersen') && m.linearBelow == 0.1 && m.p == 3, ...
        'mass_linear', strcmp(m.mass_model, 'eq2') && m.mass_q == 1, ...
        'no_eq4b', ~strcmp(m.mass_model, 'eq4b'), ...
        'adaptive_box_enabled', strcmp(m.move_policy, 'adaptive') && m.move_initial == 0.1 && ...
            m.move_minimum == 0.002 && m.adaptive_grow == 1.2 && m.adaptive_shrink == 0.7, ...
        'stage_exhaustion_disabled', ~strcmp(m.continuation_signal, 'stageExhaustion') && ~strcmp(m.stop_rule, 'stageExhaustion'), ...
        'natural_design_change_stop', strcmp(m.stop_rule, 'designChange') && strcmp(m.stop_norm, 'l2') && ...
            strcmp(m.stop_toleranceRule, 'meshScaled') && ~m.guard_settledMove && ~m.guard_ladderExhausted && ...
            ~m.guard_maxDesignChange && m.guard_boxInactiveFraction == 0 && ...
            abs(m.stop_tolerance - m.stop_tolerance_rule_value) <= 1e-15*max(1, m.stop_tolerance_rule_value), ...
        'projection_disabled', ~m.projection, ...
        'p_continuation_disabled', ~m.p_continuation && ~m.mass_continuation, ...
        'socp_disabled_nested_mma', strcmp(m.inner_type, 'mma') && strcmp(m.inner_variant, 'published'), ...
        'R_equals_0p06_b', strcmp(m.filter_type, 'sensitivity') && m.radiusPhysical == 0.06 && b == 1 && ...
            (isempty(m.radiusElements_row) || all(isnan(m.radiusElements_row))), ...
        'maxOuter_400_single_thread_no_diag', m.maxOuter == 400 && m.singleThread && ~m.diagnostics && ~m.verbose);
    meshes = [meshes, m]; %#ok<AGROW>
    fprintf('%-8s %s  frozenID %d frozenNMC %d routes %d dump %d\n', tag, hS, ...
        m.checks.hash_equals_campaign_identity, m.checks.hash_equals_nine_mesh_configs, ...
        m.checks.runner_route_hashes_agree, m.checks.dumped_rows_rehash_to_config_hash);
end
S.meshes = meshes;

varying = {};
for r = 1:numel(rows)
    for i = 2:size(M, 1)
        if ~isequaln(vals{r, 1}, vals{r, i}); varying{end+1} = rows{r}; break; end %#ok<AGROW>
    end
end
S.rows_varying_across_meshes = varying;
S.n_schema_rows = numel(rows);

perMesh = arrayfun(@(m) all(struct2array(m.checks)), meshes);
S.checks = struct( ...
    'head', strcmp(S.git.head, EXPECT_HEAD) && strcmp(S.git.head, ID.repository.merged_head), ...
    'branch', strcmp(S.git.branch, 'benchmark-methodology-r2'), ...
    'impl_tree', man.ok && strcmp(man.treeHash, EXPECT_TREE) && strcmp(man.treeHash, ID.implementation.impl_tree_sha256), ...
    'impl_files_79', man.nFiles == 79 && isempty(man.mismatches) && isempty(man.missing) && isempty(man.extra), ...
    'source_manifest', strcmp(S.impl.source_manifest_file_sha256, EXPECT_MANI) && ...
        strcmp(S.impl.source_manifest_file_sha256, ID.implementation.source_manifest_sha256), ...
    'production_preset', strcmp(prod.name, NAME) && strcmp(S.preset.benchmark_selected, NAME) && ...
        strcmp(ID.production.preset, NAME) && strcmp(NMC.preset, NAME), ...
    'nine_meshes_exact', size(M, 1) == 9 && ...
        isequal(fieldnames(ID.nine_config_hashes)', cellfun(@matlab.lang.makeValidName, {meshes.mesh}, 'UniformOutput', false)) && ...
        numel(NMC.configs) == 9 && isequal({NMC.configs.mesh}, {meshes.mesh}), ...
    'all_nine_hashes_match_both_frozen_files', all(arrayfun(@(m) m.checks.hash_equals_campaign_identity && ...
        m.checks.hash_equals_nine_mesh_configs, meshes)), ...
    'all_nine_formulation_checks', all(perMesh), ...
    'distinct_hashes', numel(unique({meshes.hash_solve_route})) == 9, ...
    'only_mesh_bound_rows_vary', all(ismember(varying, {'domain.mesh.nelx', 'domain.mesh.nely', 'stop.tolerance', 'runtime.name'})), ...
    'no_tracked_changes_or_only_locked', isequal(sort(S.git.tracked_changes(:)'), sort(allowedDirty(:)')));
if strcmp(mode, 'runner')
    S.checks.runner_file_is_locked_edit = strcmp(S.runner.sha256, S.runner.lock_edited_sha256);
    S.checks.lock_head_tree_hashes = strcmp(L.head, EXPECT_HEAD) && strcmp(L.impl_tree_sha256, EXPECT_TREE) && ...
        all(arrayfun(@(m) strcmp(L.nine_config_hashes.(matlab.lang.makeValidName(m.mesh)), m.hash_solve_route), meshes));
end
S.pass = all(struct2array(S.checks));
if strcmp(mode, 'identity')
    S.verdict = pick(S.pass, 'NINE_MESH_CAMPAIGN_IDENTITY_PASS', 'NINE_MESH_CAMPAIGN_IDENTITY_FAIL');
else
    S.verdict = pick(S.pass, 'PERFORMANCE_RUNNER_CONFIG_IDENTITY_PASS', 'PERFORMANCE_RUNNER_CONFIG_IDENTITY_FAIL');
end

fid = fopen(outJson, 'w'); fprintf(fid, '%s\n', jsonencode(S, 'PrettyPrint', true)); fclose(fid);
disp(S.checks);
fprintf('varying rows: %s\n%s\n', strjoin(varying, ', '), S.verdict);
end

% =========================================================================
function lines = hashLines(cfg, rows)
% Byte-for-byte the line construction of olhoffcurrent_config_hash.
lines = cell(numel(rows), 1);
for k = 1:numel(rows)
    p = rows{k};
    if strcmp(p, 'runtime.name'); lines{k} = sprintf('%s=<excluded>', p); continue; end
    lines{k} = sprintf('%s=%s', p, show(olh.config.getPath(cfg, p)));
end
end

function s = show(v)
if ischar(v);            s = v;
elseif isstring(v);      s = char(v);
elseif islogical(v);     s = mat2str(v);
elseif isnumeric(v);     s = mat2str(v, 17);
elseif iscell(v);        s = ['{' strjoin(cellfun(@show, v, 'UniformOutput', false), ',') '}'];
elseif isempty(v);       s = '[]';
else,                    s = class(v);
end
end

function h = sha256str(txt)
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(uint8(txt(:)));
d = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(d, 2).', 1, []));
end

function s = gitOut(repo, args)
[st, s] = system(sprintf('git --no-pager -C "%s" %s', repo, args));
if st ~= 0; s = sprintf('GIT_ERROR(%d): %s', st, s); end
s = strtrim(s);
end

function c = splitLines(s)
if isempty(s); c = {}; return; end
c = strsplit(s, newline);
c = c(~cellfun(@isempty, c));
end

function r = relp(p, repo)
r = strrep(p, [repo filesep], '');
end

function s = pick(tf, a, b)
if tf; s = a; else; s = b; end
end
