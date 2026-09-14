function mig_config_analysis(outJson)
%MIG_CONFIG_ANALYSIS  CONFIG_SCHEMA_MIGRATION evidence (MIGRATION_PREREGISTRATION §6).
%   Reads cfg_pre.mat / cfg_post.mat / cfg_up.mat (mig_config_dump) and checks, for
%   every recorded historical configuration:
%     (i)   every pre-migration schema leaf equal by value (isequal) pre vs post
%     (ii)  the pre-migration 81-row hash recomputed FROM THE POST configuration
%           equals the recorded hash
%     (iii) every added row at the value that selects the pre-migration code path
%   and, for every configuration, post vs upstream on every 87-row leaf.
P = mig_paths();
mig_use_snapshot(P.up);                      % olh.config.getPath/schema (upstream = migrated +impl)
A = load(fullfile(P.ev, 'cfg_pre.mat'));
B = load(fullfile(P.ev, 'cfg_post.mat'));
C = load(fullfile(P.ev, 'cfg_up.mat'));
oldRows = A.rows; newRows = B.rows;
assert(isequal(newRows, C.rows), 'mig:rows', 'post and upstream schema rows differ');
added   = setdiff(newRows, oldRows, 'stable');
removed = setdiff(oldRows, newRows, 'stable');
% the value of each added row that reproduces the pre-migration code path
neutral = struct( ...
    'material_stiffness_model', 'simp', ...
    'optimizer_inner_asymptoteHistory', 'inner', ...
    'stop_guards_settledWindow', 1, ...
    'stop_guards_boxInactiveFraction', 0);
key = @(p) strrep(p, '.', '_');

S = struct();
S.schema = struct('oldRows', numel(oldRows), 'newRows', numel(newRows), ...
    'added', {added(:).'}, 'removed', {removed(:).'}, ...
    'oldOrderPreserved', isequal(newRows(ismember(newRows, oldRows)), oldRows));
S.addedRowNeutralValues = neutral;
S.historical = struct('case', {}, 'mesh', {}, 'recordedHash', {}, 'recordedSource', {}, ...
    'preHash', {}, 'postHash', {}, 'postOldSchemaHash', {}, 'preHashEqualsRecorded', {}, ...
    'oldSchemaHashFromPostEqualsRecorded', {}, 'oldLeavesEqual', {}, 'differingOldLeaves', {}, ...
    'addedLeaves', {}, 'addedLeavesNeutral', {}, 'postEqualsUpstreamAllLeaves', {}, 'upHash', {});
for i = 1:numel(A.D)
    a = A.D(i);
    j = find(arrayfun(@(d) strcmp(d.case, a.case) && isequal(d.mesh, a.mesh), B.D), 1);
    k = find(arrayfun(@(d) strcmp(d.case, a.case) && isequal(d.mesh, a.mesh), C.D), 1);
    b = B.D(j); c = C.D(k);
    diffs = {};
    for r = 1:numel(oldRows)
        if ~isequal(olh.config.getPath(a.cfg, oldRows{r}), olh.config.getPath(b.cfg, oldRows{r}))
            diffs{end+1} = oldRows{r}; %#ok<AGROW>
        end
    end
    [hOld, ~] = mig_hash_rows(b.cfg, oldRows);
    addedVals = struct(); neutralOk = true;
    for r = 1:numel(added)
        v = olh.config.getPath(b.cfg, added{r});
        addedVals.(key(added{r})) = v;
        if isfield(neutral, key(added{r}))
            neutralOk = neutralOk && isequal(v, neutral.(key(added{r})));
        end
    end
    S.historical(end+1) = struct('case', a.case, 'mesh', a.mesh, 'recordedHash', a.recordedHash, ...
        'recordedSource', a.recordedSource, 'preHash', a.sideHash, 'postHash', b.sideHash, ...
        'postOldSchemaHash', hOld, 'preHashEqualsRecorded', strcmp(a.sideHash, a.recordedHash), ...
        'oldSchemaHashFromPostEqualsRecorded', strcmp(hOld, a.recordedHash), ...
        'oldLeavesEqual', isempty(diffs), 'differingOldLeaves', {diffs}, ...
        'addedLeaves', addedVals, 'addedLeavesNeutral', neutralOk, ...
        'postEqualsUpstreamAllLeaves', local_allEqual(b.cfg, c.cfg, newRows), 'upHash', c.sideHash); %#ok<AGROW>
end

% Pedersen preset: post vs upstream at every sweep mesh, and vs the committed sweep cfg
S.pedersen = struct('mesh', {}, 'postHash', {}, 'upHash', {}, 'postEqualsUpstreamAllLeaves', {}, ...
    'committedSweepRun', {}, 'differingLeavesVsCommitted', {});
for j = 1:numel(B.D)
    b = B.D(j);
    if ~strcmp(b.case, 'PED'); continue; end
    k = find(arrayfun(@(d) strcmp(d.case, 'PED') && isequal(d.mesh, b.mesh), C.D), 1);
    c = C.D(k);
    run = sprintf('S%dx%d', b.mesh(1), b.mesh(2));
    R = load(fullfile(P.up, 'repro', 'results', run, 'res.mat'), 'res');
    dv = {};
    for r = 1:numel(newRows)
        try
            vc = olh.config.getPath(R.res.cfg, newRows{r});
        catch
            vc = '<absent in committed cfg>';
        end
        if ~isequal(vc, olh.config.getPath(b.cfg, newRows{r})); dv{end+1} = newRows{r}; end %#ok<AGROW>
    end
    S.pedersen(end+1) = struct('mesh', b.mesh, 'postHash', b.sideHash, 'upHash', c.sideHash, ...
        'postEqualsUpstreamAllLeaves', local_allEqual(b.cfg, c.cfg, newRows), ...
        'committedSweepRun', run, 'differingLeavesVsCommitted', {dv}); %#ok<AGROW>
end

H = S.historical;
S.checks = struct( ...
    'allPreHashesReproduceRecorded', all([H.preHashEqualsRecorded]), ...
    'allOldLeavesEqual', all([H.oldLeavesEqual]), ...
    'allOldSchemaHashesFromPostReproduceRecorded', all([H.oldSchemaHashFromPostEqualsRecorded]), ...
    'allAddedLeavesNeutral', all([H.addedLeavesNeutral]), ...
    'allHistoricalPostEqualsUpstream', all([H.postEqualsUpstreamAllLeaves]), ...
    'allPedersenPostEqualsUpstream', all([S.pedersen.postEqualsUpstreamAllLeaves]), ...
    'schemaOnlyGrew', isempty(removed) && S.schema.oldOrderPreserved);
S.pass = all(struct2array(S.checks));
fid = fopen(outJson, 'w'); fwrite(fid, jsonencode(S, 'PrettyPrint', true)); fclose(fid);
fprintf('CONFIG ANALYSIS pass=%d rows %d->%d added=%s\n', S.pass, numel(oldRows), numel(newRows), strjoin(added, ','));
disp(S.checks);
for j = 1:numel(S.pedersen)
    fprintf('  PED %s differs from committed sweep cfg in: %s\n', S.pedersen(j).committedSweepRun, ...
        strjoin(S.pedersen(j).differingLeavesVsCommitted, ', '));
end
end

function ok = local_allEqual(x, y, rows)
ok = true;
for r = 1:numel(rows)
    if ~isequal(olh.config.getPath(x, rows{r}), olh.config.getPath(y, rows{r})); ok = false; return; end
end
end
