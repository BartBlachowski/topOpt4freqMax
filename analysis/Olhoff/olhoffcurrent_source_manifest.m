function man = olhoffcurrent_source_manifest(varargin)
%OLHOFFCURRENT_SOURCE_MANIFEST  Integrity manifest over the promoted source.
%
%   man = OLHOFFCURRENT_SOURCE_MANIFEST() hashes every SOURCE file under +impl/
%   and returns the manifest plus a single tree hash that stands for the whole
%   implementation.
%
%   Filesystem and editor artifacts (.DS_Store, .asv, .m~, ...) are stepped
%   over rather than hashed, because they are not executable source and macOS
%   and MATLAB create them unbidden.  What counts as an artifact is decided in
%   ONE place, olhoffcurrent_is_artifact, which never classifies anything MATLAB
%   can execute as one -- so an unexpected .m file is still a hard block.  The
%   artifacts that were stepped over are returned in man.artifactsIgnored, so
%   they are visible rather than silently dropped.  The tree hash is the SHA-256 of the sorted
%   "<relative path>  <sha256>" lines, so it changes if any file changes, is
%   added or is removed, and does not depend on filesystem ordering.
%
%   Options:
%     'Verify'  (default true when SOURCE_MANIFEST.json exists) compare against
%               the recorded manifest and populate man.mismatches / man.missing
%               / man.extra
%     'Write'   (default false) (re)write SOURCE_MANIFEST.json.  Use ONLY when
%               deliberately promoting a new upstream state -- rewriting it to
%               make a check pass destroys the thing the check is for.
%
%   man.ok is true only when the tree hashes to exactly what was recorded.
%   man.artifactsIgnored lists the non-source files that were stepped over.
%
%   See also OLHOFFCURRENT_CURRENTNESS, OLHOFFCURRENT_PROVENANCE.

root     = olhoffcurrent_root();
core     = fullfile(root, '+impl');
manFile  = fullfile(root, 'SOURCE_MANIFEST.json');

p = inputParser();
p.addParameter('Verify', exist(manFile,'file') == 2, @(v) islogical(v) && isscalar(v));
p.addParameter('Write',  false, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
opt = p.Results;

[L, skippedAbs] = local_listFiles(core);
rel = cell(numel(L),1); hsh = cell(numel(L),1);
for k = 1:numel(L)
    rel{k} = strrep(L{k}(numel(core)+2:end), filesep, '/');
    hsh{k} = olhoffcurrent_sha256_file(L{k});
end
[rel, ix] = sort(rel); hsh = hsh(ix);

lines = cell(numel(rel),1);
for k = 1:numel(rel); lines{k} = sprintf('%s  %s', rel{k}, hsh{k}); end
joined = strjoin(lines, newline);

md = java.security.MessageDigest.getInstance('SHA-256');
md.update(uint8(joined(:)));
d = typecast(md.digest(), 'uint8');
treeHash = lower(reshape(dec2hex(d, 2).', 1, []));

skipped = cell(numel(skippedAbs),1);
for k = 1:numel(skippedAbs)
    skipped{k} = strrep(skippedAbs{k}(numel(core)+2:end), filesep, '/');
end
skipped = sort(skipped);

man = struct('root', core, 'nFiles', numel(rel), 'treeHash', treeHash, ...
             'files', {rel}, 'hashes', {hsh}, 'ok', true, ...
             'mismatches', {{}}, 'missing', {{}}, 'extra', {{}}, ...
             'artifactsIgnored', {skipped});

if opt.Write
    S = struct('manifest_schema','olhoff_current_source_manifest/1', ...
               'generated', char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ssXXX','TimeZone','local')), ...
               'root','analysis/OlhoffCurrent/+impl', ...
               'n_files', numel(rel), 'tree_sha256', treeHash, ...
               'files', struct('path', rel, 'sha256', hsh));
    fid = fopen(manFile,'w'); c = onCleanup(@() fclose(fid));
    fprintf(fid, '%s\n', jsonencode(S, 'PrettyPrint', true));
    return
end

if opt.Verify
    if exist(manFile,'file') ~= 2
        man.ok = false; man.missing = {manFile}; return
    end
    R = jsondecode(fileread(manFile));
    recPath = {R.files.path}; recHash = {R.files.sha256};
    for k = 1:numel(rel)
        j = find(strcmp(rel{k}, recPath), 1);
        if isempty(j); man.extra{end+1} = rel{k};
        elseif ~strcmp(hsh{k}, recHash{j}); man.mismatches{end+1} = rel{k}; end
    end
    for j = 1:numel(recPath)
        if ~any(strcmp(recPath{j}, rel)); man.missing{end+1} = recPath{j}; end
    end
    man.recordedTreeHash = R.tree_sha256;
    man.ok = isempty(man.mismatches) && isempty(man.missing) && ...
             isempty(man.extra) && strcmp(treeHash, R.tree_sha256);
end
end

function [out, skipped] = local_listFiles(d)
%LOCAL_LISTFILES  Every SOURCE file under d, plus the artifacts it stepped over.
%   Classification is delegated to olhoffcurrent_is_artifact -- the single
%   artifact policy -- so this function contains no ignore list of its own.
out = {}; skipped = {};
L = dir(d);
for k = 1:numel(L)
    if strcmp(L(k).name,'.') || strcmp(L(k).name,'..'); continue; end
    p = fullfile(d, L(k).name);
    if L(k).isdir
        [o, sk] = local_listFiles(p);
        out = [out, o]; skipped = [skipped, sk]; %#ok<AGROW>
    elseif olhoffcurrent_is_artifact(L(k).name)
        skipped{end+1} = p; %#ok<AGROW>
    else
        out{end+1} = p; %#ok<AGROW>
    end
end
end
