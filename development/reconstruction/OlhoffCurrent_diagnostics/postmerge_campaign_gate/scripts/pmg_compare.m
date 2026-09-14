function S = pmg_compare(outJson)
%PMG_COMPARE  Steps 5A/5B: the two post-merge 160x20 anchors against their references,
%   at the strongest previously established level -- the migration's mig_compare
%   standard: bitwise leaf comparison (numeric/logical by class, size and raw bytes;
%   char/cell/struct recursively); excluded: tEig tGrad tInner tOuter wallclock and
%   cfg.provenance.resolvedAt.  Comparator functions are copied verbatim from
%   diagnostics/upstream_253069_migration/scripts/mig_compare.m.
here = fileparts(mfilename('fullpath'));
oc = fileparts(fileparts(fileparts(here)));
repo = fileparts(fileparts(oc));
restoredefaultpath; addpath(here); addpath(oc);
guard = olhoffcurrent_paths(); %#ok<NASGU>
EXC = {'tEig', 'tGrad', 'tInner', 'tOuter', 'wallclock', 'resolvedAt'};
EV  = fullfile(oc, 'evidence', 'postmerge_campaign_gate');
MIG = fullfile(oc, 'evidence', 'upstream_253069_migration');
UP  = '/private/tmp/claude-501/-Users-piotrek-Programming-topOpt4freqMax/84bbf569-dc04-4771-be4d-d9d2da2b8566/scratchpad/up253069';
S = struct('standard', ['bitwise (class+size+raw bytes for numeric/logical; recursive for struct/cell); ' ...
    'excluded: tEig tGrad tInner tOuter wallclock provenance.resolvedAt'], 'when', char(datetime('now')));
schemaRows = olh.config.schema(); schemaRows = schemaRows(:, 1);
F = jsondecode(fileread(fullfile(oc, 'tests', 'fixtures', 'schema_rows_pre_253069.json')));
old81 = F.rows;

% ============================ 5A historical ==================================
H = load(fullfile(EV, 'ANCHOR_HISTORICAL_S160.mat'));
PRE = load(fullfile(MIG, 'PRE_BETA.mat')); POST = load(fullfile(MIG, 'POST_BETA.mat')); UPB = load(fullfile(MIG, 'UP_BETA.mat'));
A = struct();
A.meta = H.meta; A.metrics = local_metrics(H.res);
A.vs_migration_post = local_cmp(H.res, POST.res, EXC, schemaRows, H.cfg, POST.cfg);
A.vs_migration_post.cfg_struct_differing = local_diff(H.cfg, POST.cfg, 'cfg', EXC);
A.vs_premigration = local_cmp(H.res, PRE.res, EXC, schemaRows, H.cfg, PRE.cfg);
A.vs_upstream_snapshot_run = local_cmp(H.res, UPB.res, EXC, schemaRows, H.cfg, UPB.cfg);
B = load(fullfile(repo, 'examples', 'Performance', 'conference_benchmark', 'campaign_9mesh_r2', 'benchmark_records.mat'));
rec = B.records(strcmp({B.records.method_key}, 'olhoff'));
ref = rec(find(arrayfun(@(s) isequal(s.mesh(:).', [160 20]), rec), 1));
got = H.res;
A.vs_frozen_campaign_record = struct( ...
    'reference', 'examples/Performance/conference_benchmark/campaign_9mesh_r2/benchmark_records.mat (olhoff, 160x20)', ...
    'rho_bitwise', local_bytesEqual(double(got.rho(:)), double(ref.x(:))), ...
    'omega1_bitwise', local_bytesEqual(double(got.omega(1)), double(ref.omega(1))), ...
    'volume_bitwise', local_bytesEqual(mean(double(got.rho(:))), mean(double(ref.x(:)))), ...
    'outer', [numel(got.hist.N), double(ref.counts.outer_iterations)], ...
    'inner', [sum(got.hist.nInner), double(ref.counts.inner_iterations_total)], ...
    'status', {{got.status, ref.status}}, 'effective_config_hash_recorded', ref.effective_config_hash);
[h81, ~] = local_hashRows(H.cfg, old81);
A.config_hash = struct('new_87row_hash', olhoffcurrent_config_hash(H.cfg), 'n_rows_now', numel(schemaRows), ...
    'old_81row_hash_reconstructed', h81, 'n_old_rows', numel(old81), ...
    'old_81row_hash_recorded_campaign', ref.effective_config_hash, ...
    'old_81row_hash_recorded_provenance_event1', '28756d22aacb59726be9f37583deca89fcfcecc93f9b867d46a49223ed1db697', ...
    'old_values_identical_to_premigration', local_rowsEqual(H.cfg, PRE.cfg, old81));
A.frozen_values = struct('outer', numel(got.hist.N), 'inner', sum(got.hist.nInner), ...
    'omega1_15g', sprintf('%.15g', got.omega(1)));
A.pass = struct( ...
    'bitwise_vs_migration_post', A.vs_migration_post.passStrict && isempty(A.vs_migration_post.cfg_struct_differing), ...
    'vs_premigration', A.vs_premigration.passPrePost, ...
    'bitwise_vs_upstream_snapshot', A.vs_upstream_snapshot_run.passStrict, ...
    'campaign_record', A.vs_frozen_campaign_record.rho_bitwise && A.vs_frozen_campaign_record.omega1_bitwise && ...
        A.vs_frozen_campaign_record.volume_bitwise && isequal(A.vs_frozen_campaign_record.outer, [91 91]) && ...
        isequal(A.vs_frozen_campaign_record.inner, [2241 2241]) && strcmp(got.status, 'CONVERGED'), ...
    'old_81row_hash', numel(old81) == 81 && strcmp(h81, ref.effective_config_hash) && ...
        strcmp(h81, A.config_hash.old_81row_hash_recorded_provenance_event1) && A.config_hash.old_values_identical_to_premigration, ...
    'frozen_values', A.frozen_values.outer == 91 && A.frozen_values.inner == 2241 && ...
        strcmp(A.frozen_values.omega1_15g, '169.495227021538'), ...
    'single_thread_clean_impl', H.meta.threads == 1 && isempty(H.meta.impl_dirty) && ...
        strcmp(H.meta.treeHash, '4ba9a3ae10881344a0e60f2b8a8976c5ec9ceccf966bc2a3da4f37fb5aebffbf'));
A.PASS = all(struct2array(A.pass));
S.historical = A;

% ============================ 5B Pedersen ====================================
P = load(fullfile(EV, 'ANCHOR_PEDERSEN_S160.mat'));
PP = load(fullfile(MIG, 'POST_PED.mat')); UPP = load(fullfile(MIG, 'UP_PED.mat'));
s160 = fullfile(UP, 'repro', 'results', 'S160x20', 'res.mat');
Cm = load(s160, 'res');
B5 = struct();
B5.meta = P.meta; B5.metrics = local_metrics(P.res);
B5.committed_S160 = struct('artifact', 'Olhoff@253069 git archive: repro/results/S160x20/res.mat', 'sha256', local_fileSha(s160), ...
    'sha256_at_6b08708', local_blobSha('/Users/piotrek/Programming/Matlab/Olhoff', '6b0870850d7407a0f2fc31dbf0d0f318c2e5f2f7', 'repro/results/S160x20/res.mat'));
B5.vs_committed_S160 = local_cmp(P.res, Cm.res, EXC, schemaRows, P.cfg, Cm.res.cfg);
B5.vs_migration_post = local_cmp(P.res, PP.res, EXC, schemaRows, P.cfg, PP.cfg);
B5.vs_migration_post.cfg_struct_differing = local_diff(P.cfg, PP.cfg, 'cfg', EXC);
B5.vs_upstream_snapshot_run = local_cmp(P.res, UPP.res, EXC, schemaRows, P.cfg, UPP.cfg);
h = P.res.hist; a = P.res.aux;
B5.box = struct('move_max_first5', h.move(1:5), 'move_max_last', h.move(end), ...
    'moveMean_first5', a.moveMean(1:5), 'moveMean_last', a.moveMean(end), ...
    'move_equals_committed', local_leafEqual(h.move, Cm.res.hist.move), ...
    'moveMean_equals_committed', local_leafEqual(a.moveMean, Cm.res.aux.moveMean), ...
    'Mnd_equals_committed', local_leafEqual(a.Mnd, Cm.res.aux.Mnd), ...
    'multiplicity_N_equals_committed', local_leafEqual(h.N, Cm.res.hist.N), ...
    'nInner_equals_committed', local_leafEqual(h.nInner, Cm.res.hist.nInner));
B5.stop = struct('rule', olh.config.getPath(P.cfg, 'stop.rule'), 'status', P.res.status, ...
    'status_committed', Cm.res.status, 'log_equals_committed', isequal(P.res.log, Cm.res.log));
B5.frozen_values = struct('outer', numel(h.N), 'inner', sum(h.nInner), 'omega1_15g', sprintf('%.15g', P.res.omega(1)));
B5.pass = struct( ...
    'vs_committed_S160', B5.vs_committed_S160.passCommitted, ...
    'bitwise_vs_migration_post', B5.vs_migration_post.passStrict && isempty(B5.vs_migration_post.cfg_struct_differing), ...
    'bitwise_vs_upstream_snapshot', B5.vs_upstream_snapshot_run.passStrict, ...
    'committed_artifact_is_6b08708_blob', strcmp(B5.committed_S160.sha256, B5.committed_S160.sha256_at_6b08708), ...
    'box_and_multiplicity', B5.box.move_equals_committed && B5.box.moveMean_equals_committed && ...
        B5.box.Mnd_equals_committed && B5.box.multiplicity_N_equals_committed && B5.box.nInner_equals_committed, ...
    'stop', strcmp(B5.stop.rule, 'designChange') && strcmp(B5.stop.status, B5.stop.status_committed) && B5.stop.log_equals_committed, ...
    'frozen_values', B5.frozen_values.outer == 121 && B5.frozen_values.inner == 2369 && ...
        strcmp(B5.frozen_values.omega1_15g, '169.210576386275'), ...
    'single_thread_clean_impl', P.meta.threads == 1 && isempty(P.meta.impl_dirty) && ...
        strcmp(P.meta.treeHash, '4ba9a3ae10881344a0e60f2b8a8976c5ec9ceccf966bc2a3da4f37fb5aebffbf'));
B5.PASS = all(struct2array(B5.pass));
S.pedersen = B5;
fid = fopen(outJson, 'w'); fwrite(fid, jsonencode(S, 'PrettyPrint', true)); fclose(fid);
disp(S.historical.pass); disp(S.pedersen.pass);
fprintf('PMGCOMPARE historical PASS=%d pedersen PASS=%d\n', S.historical.PASS, S.pedersen.PASS);
end

% ---------------------------------------------------------------- helpers
function [h, lines] = local_hashRows(cfg, rows)
% character-identical to mig_hash_rows / olhoffcurrent_config_hash, explicit row list
lines = cell(numel(rows), 1);
for k = 1:numel(rows)
    p = rows{k};
    if strcmp(p, 'runtime.name'); lines{k} = sprintf('%s=<excluded>', p); continue; end
    lines{k} = sprintf('%s=%s', p, local_show(olh.config.getPath(cfg, p)));
end
h = local_sha(uint8(strjoin(lines, newline)));
end

function s = local_show(v)
if ischar(v);            s = v;
elseif isstring(v);      s = char(v);
elseif islogical(v);     s = mat2str(v);
elseif isnumeric(v);     s = mat2str(v, 17);
elseif iscell(v);        s = ['{' strjoin(cellfun(@local_show, v, 'UniformOutput', false), ',') '}'];
elseif isempty(v);       s = '[]';
else,                    s = class(v);
end
end

function tf = local_rowsEqual(a, b, rows)
tf = true;
for r = 1:numel(rows)
    if ~local_leafEqual(olh.config.getPath(a, rows{r}), olh.config.getPath(b, rows{r})); tf = false; return; end
end
end

function m = local_metrics(res)
r = double(res.rho(:)); w = double(res.omega(:));
m = struct('status', res.status, 'nOuter', numel(res.hist.N), 'innerTotal', sum(res.hist.nInner), ...
    'innerMax', max(res.hist.nInner), 'omega1', w(1), 'omega2', w(2), 'gap12_pct', 100*(w(2)-w(1))/w(1), ...
    'volume', mean(r), 'Mnd_pct', 100*mean(4*r.*(1-r)), 'gray_fraction', mean(r > 0.1 & r < 0.9), ...
    'mid_fraction', mean(r > 0.4 & r < 0.6), 'rho_sha256_bytes', local_sha(typecast(r, 'uint8')), ...
    'final_move_max', res.hist.move(end), 'final_N', res.hist.N(end), 'log_tail', {res.log(max(1,end-2):end)});
end

% ---- verbatim from upstream_253069_migration/scripts/mig_compare.m -------------
function c = local_cmp(a, b, EXC, rows, cfgA, cfgB)
d = local_diff(a, b, 'res', [EXC, {'cfg'}]);
onlyA = d(endsWith(d, '(only in candidate)'));
onlyB = d(endsWith(d, '(only in reference)'));
vals  = setdiff(d, [onlyA, onlyB], 'stable');
c = struct('differing', {vals}, 'onlyInCandidate', {onlyA}, 'onlyInReference', {onlyB});
if ~isempty(rows)
    cd = {};
    for r = 1:numel(rows)
        try, va = olh.config.getPath(cfgA, rows{r}); catch, va = '<absent>'; end
        try, vb = olh.config.getPath(cfgB, rows{r}); catch, vb = '<absent>'; end
        if ~local_leafEqual(va, vb); cd{end+1} = rows{r}; end %#ok<AGROW>
    end
    c.cfgRowsDiffering = cd;
else
    c.cfgRowsDiffering = {};
end
c.passDiffs  = isempty(vals);
c.passStrict = isempty(vals) && isempty(onlyA) && isempty(onlyB) && isempty(c.cfgRowsDiffering);
exNames = regexp(onlyB, '^res\.hist\.(ex[A-Za-z]+) \(only in reference\)$', 'tokens', 'once');
exOnly = all(~cellfun(@isempty, exNames));
exEmpty = exOnly && all(cellfun(@(t) isempty(b.hist.(t{1})), exNames));
c.droppedFieldsAllEmptyExTrace = exEmpty;
addedRows = {'material.stiffness.linearBelow', 'optimizer.inner.asymptoteHistory', 'move.adaptive.grow', ...
             'move.adaptive.shrink', 'stop.guards.settledWindow', 'stop.guards.boxInactiveFraction'};
c.cfgRowsDifferingOnlyAddedRows = all(ismember(c.cfgRowsDiffering, addedRows));
c.passPrePost = isempty(vals) && all(ismember(onlyA, {'res.aux (only in candidate)'})) && ...
    (isempty(onlyB) || exEmpty) && c.cfgRowsDifferingOnlyAddedRows;
c.passCommitted = isempty(vals) && isempty(onlyB) && all(ismember(c.cfgRowsDiffering, {'stop.rule', 'runtime.verbose', 'runtime.name'}));
end

function d = local_diff(a, b, path, EXC)
d = {};
if isstruct(a) && isstruct(b) && isscalar(a) && isscalar(b)
    f = union(fieldnames(a), fieldnames(b), 'stable');
    for k = 1:numel(f)
        if any(strcmp(f{k}, EXC)); continue; end
        p = [path '.' f{k}];
        if ~isfield(a, f{k}),     d{end+1} = [p ' (only in reference)']; %#ok<AGROW>
        elseif ~isfield(b, f{k}), d{end+1} = [p ' (only in candidate)']; %#ok<AGROW>
        else, d = [d, local_diff(a.(f{k}), b.(f{k}), p, EXC)]; %#ok<AGROW>
        end
    end
elseif iscell(a) && iscell(b) && isequal(size(a), size(b))
    for k = 1:numel(a)
        d = [d, local_diff(a{k}, b{k}, sprintf('%s{%d}', path, k), EXC)]; %#ok<AGROW>
    end
elseif ~local_leafEqual(a, b)
    d{end+1} = path;
end
end

function tf = local_leafEqual(a, b)
if (isnumeric(a) || islogical(a)) && (isnumeric(b) || islogical(b))
    tf = strcmp(class(a), class(b)) && isequal(size(a), size(b)) && ...
         (isempty(a) || isequal(typecast(local_bytes(a), 'uint8'), typecast(local_bytes(b), 'uint8')));
else
    tf = isequaln(a, b);
end
end

function u = local_bytes(v)
if islogical(v), u = uint8(v(:)); elseif ~isreal(v), u = typecast([real(v(:)); imag(v(:))], 'uint8');
else, u = typecast(v(:), 'uint8'); end
end

function tf = local_bytesEqual(a, b)
tf = isequal(size(a), size(b)) && isequal(typecast(a(:), 'uint8'), typecast(b(:), 'uint8'));
end

function h = local_fileSha(p)
fid = fopen(p, 'r'); b = fread(fid, Inf, '*uint8'); fclose(fid);
h = local_sha(b);
end

function h = local_blobSha(repo, commit, path)
tmp = [tempname() '.blob'];
if system(sprintf('git --no-pager -C "%s" cat-file -e "%s:%s"', repo, commit, path)) ~= 0; h = ''; return; end
if system(sprintf('git --no-pager -C "%s" cat-file blob "%s:%s" > "%s"', repo, commit, path, tmp)) ~= 0; h = ''; return; end
h = local_fileSha(tmp); delete(tmp);
end

function h = local_sha(bytes)
md = java.security.MessageDigest.getInstance('SHA-256');
if ~isempty(bytes), md.update(uint8(bytes(:))); end
x = typecast(md.digest(), 'uint8'); h = lower(reshape(dec2hex(x, 2).', 1, []));
end
