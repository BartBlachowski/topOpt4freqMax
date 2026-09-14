function s = nmp_identity_snapshot(ctx)
%NMP_IDENTITY_SNAPSHOT  Repository scientific identity at this instant, against the lock.
%   HEAD, branch, the set of tracked files that differ from HEAD, the runner file
%   (must be the locked run-selection edit), olhoffcurrent_run.m (must be the
%   committed file the hooks are bound to), the +impl tree re-hashed from disk,
%   the SOURCE_MANIFEST.json bytes, and the lock file itself.
repo = ctx.repo_abs;
oc = fullfile(repo, 'analysis', 'OlhoffCurrent');
s = struct();
s.when = nmp_now();
s.head = gitOut(repo, 'rev-parse HEAD');
s.branch = gitOut(repo, 'rev-parse --abbrev-ref HEAD');
tc = gitOut(repo, 'diff --name-only HEAD');
if isempty(tc); s.tracked_changes = {}; else; s.tracked_changes = sort(strsplit(tc, newline)); end
s.runner_sha256 = olhoffcurrent_sha256_file(fullfile(repo, ctx.runner.path));
s.olhoffcurrent_run_sha256 = olhoffcurrent_sha256_file(fullfile(repo, ctx.olhoffcurrent_run.path));
man = olhoffcurrent_source_manifest();
s.impl_tree_sha256 = man.treeHash;
s.impl_n_files = man.nFiles;
s.impl_manifest_ok = man.ok;
s.source_manifest_sha256 = olhoffcurrent_sha256_file(fullfile(oc, 'SOURCE_MANIFEST.json'));
s.lock_sha256 = olhoffcurrent_sha256_file(ctx.lock_path);

allowed = sort(cellstr(ctx.allowed_tracked_changes));
s.checks = struct( ...
    'head', strcmp(s.head, ctx.head), ...
    'branch', strcmp(s.branch, ctx.branch), ...
    'impl_tree', man.ok && strcmp(s.impl_tree_sha256, ctx.impl_tree_sha256), ...
    'source_manifest', strcmp(s.source_manifest_sha256, ctx.source_manifest_sha256), ...
    'runner_is_locked_edit', strcmp(s.runner_sha256, ctx.runner.edited_sha256), ...
    'olhoffcurrent_run_committed', strcmp(s.olhoffcurrent_run_sha256, ctx.olhoffcurrent_run.sha256), ...
    'tracked_changes_only_locked', isequal(s.tracked_changes(:)', allowed(:)'), ...
    'lock_unchanged', strcmp(s.lock_sha256, ctx.lock_sha256_expected));
s.pass = all(struct2array(s.checks));
end

function out = gitOut(repo, args)
[st, out] = system(sprintf('git --no-pager -C "%s" %s', repo, args));
out = strtrim(out);
if st ~= 0; out = sprintf('GIT_ERROR(%d): %s', st, out); end
end
