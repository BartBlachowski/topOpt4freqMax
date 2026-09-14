function sd_target_identity()
%SD_TARGET_IDENTITY  Part 2: target implementation identity, read-only.
%   Uses the target's own integrity tools (manifest verify, currentness,
%   config hash).  Writes evaluations/target_identity.json.
[P, guard] = sd_use_target(); %#ok<ASGLU>
out = struct();
man = olhoffcurrent_source_manifest();          % Verify (default), never Write
out.manifest_ok = man.ok;
out.manifest_tree = man.treeHash; out.manifest_recorded_tree = man.recordedTreeHash;
out.manifest_nfiles = man.nFiles;
fn = {'mismatches','missing','extra','artifactsIgnored'};
for i = 1:numel(fn), if isfield(man, fn{i}), out.(['manifest_' fn{i}]) = man.(fn{i}); end, end
cur = olhoffcurrent_currentness();
out.currentness_state = cur.state;
cur = rmfield(cur, intersect(fieldnames(cur), {'manifest'})); out.currentness = cur;
% canonical production at 480x60
[cfgProd, info] = olhoffcurrent_config(480, 60);
out.production_preset = info.name;
out.production_upstream_preset = info.upstreamPreset;
out.production_cfgHash_480 = olhoffcurrent_config_hash(cfgProd);
out.production_levels = cfgProd.move.levels;
out.production_signal = cfgProd.move.continuation.signal;
out.production_stop_rule = cfgProd.stop.rule;
% retained three-rung canary C480
L = load(P.c480traj, 'cfg', 'meta');
out.c480_cfgHash = olhoffcurrent_config_hash(L.cfg);
out.c480_cfgHash_recorded = '03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e';
out.c480_cfgHash_match = strcmp(out.c480_cfgHash, out.c480_cfgHash_recorded);
D = sd_cfgdiff(cfgProd, L.cfg);
out.prod_vs_c480_diff = arrayfun(@(d) struct('path', d.path, 'production', sd_s(d.a), 'c480', sd_s(d.b)), D);
% frozen historical preset file
out.frozenM4_sha256 = sd_filehash(fullfile(P.impl, 'architecture', '+olh', '+presets', 'duOlhoffFrozenM4.m'));
ff = {'olhoffcurrent_preset.m','olhoffcurrent_config.m','olhoffcurrent_run.m','PROVENANCE.json', ...
      'PROVENANCE.md','SOURCE_MANIFEST.json','README.md'};
for i = 1:numel(ff)
    out.files.(matlab.lang.makeValidName(ff{i})) = sd_filehash(fullfile(P.oc, ff{i}));
end
out.c480_trajectory_sha256 = sd_filehash(P.c480traj);
out.c480_state_sha256 = sd_filehash(P.c480state);
fid = fopen(fullfile(P.eval, 'target_identity.json'), 'w');
fprintf(fid, '%s\n', jsonencode(out, 'PrettyPrint', true)); fclose(fid);
fprintf('manifest ok=%d tree=%s n=%d currentness=%s prodHash=%s c480 match=%d\n', ...
    out.manifest_ok, out.manifest_tree, out.manifest_nfiles, out.currentness_state, ...
    out.production_cfgHash_480, out.c480_cfgHash_match);
for i = 1:numel(D), fprintf('  prod vs C480: %-40s %s -> %s\n', D(i).path, sd_s(D(i).a), sd_s(D(i).b)); end
end

function s = sd_s(v)
if ischar(v) || isstring(v), s = char(v);
elseif isnumeric(v) || islogical(v), s = mat2str(v, 17);
elseif iscell(v), s = strjoin(cellfun(@sd_s, v, 'UniformOutput', false), ' | ');
else, s = class(v);
end
end
