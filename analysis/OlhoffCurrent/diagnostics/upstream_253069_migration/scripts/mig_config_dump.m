function mig_config_dump(side, outMat)
%MIG_CONFIG_DUMP  Resolve every recorded historical configuration on one side (no solve).
%   side 'pre' | 'post' | 'up'.  Resolution only: no mesh is solved here.
%   Records, per configuration: resolved cfg, schema row list, the side's own
%   config hash (pre/post: olhoffcurrent_config_hash), and for post the
%   pre-migration 81-row hash recomputed from the migrated configuration.
P = mig_paths();
switch side
    case {'pre', 'post'}
        restoredefaultpath; addpath(P.scripts); addpath(P.oc);
        guard = olhoffcurrent_paths(); %#ok<NASGU>
    case 'up'
        mig_use_snapshot(P.up);
end
S = olh.config.schema();
rows = S(:,1);

% recorded historical hashes (pre-migration 81-row schema)
H = {};
J = jsondecode(fileread(P.ref.benchmarkResults));
for i = 1:numel(J.runs)
    r = J.runs(i);
    if iscell(r), r = r{1}; end
    if isfield(r, 'method_key') && strcmp(r.method_key, 'olhoff')
        H(end+1,:) = {'BETA', double(r.mesh(:).'), r.effective_config_hash, ...
            'examples/Performance/conference_benchmark/campaign_9mesh_r2/benchmark_results.json'}; %#ok<AGROW>
    end
end
H = [H; ...
  {'EX3', [320 40],  'afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab', 'diagnostics/three_rung_promotion_validation_retry1/METRICS.json (TR3_C_320x40)'}; ...
  {'EX3', [480 60],  '03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e', 'diagnostics/three_rung_canary_preflight/EFFECTIVE_CONFIG.json (CAN3_480x60)'}; ...
  {'EX3', [800 100], '7724af5ed26786e16132c30fbae840f5d8679d5ea798fc75d5c48eefd0e50ffe', 'diagnostics/three_rung_canary_preflight/EFFECTIVE_CONFIG.json (CAN3_800x100)'}; ...
  {'EX4', [160 20],  '31d2ef382746a942a4036d07dd6a1012742432cba0497d2de5ec24b51d2b5904', 'diagnostics/two_branch_controller_validation/runs/C160x20_record.json'}; ...
  {'EX4', [320 40],  '2359a1112fcec9edd0971aae9a89508cd703ea3899770e7e85c850138c2550f4', 'diagnostics/two_branch_controller_validation/runs/C320x40_record.json'}];
if ~strcmp(side, 'pre')
    meshes = [160 20; 240 30; 320 40; 400 50; 480 60; 560 70; 640 80; 720 90; 800 100];
    for i = 1:size(meshes,1)
        H(end+1,:) = {'PED', meshes(i,:), '', '(new preset: no pre-migration hash)'}; %#ok<AGROW>
    end
end

D = struct('case', {}, 'mesh', {}, 'recordedHash', {}, 'recordedSource', {}, 'cfg', {}, ...
           'sideHash', {}, 'lines', {});
for i = 1:size(H,1)
    cfg = mig_case_cfg(side, H{i,1}, H{i,2}(1), H{i,2}(2));
    if strcmp(side, 'up')
        [h, L] = mig_hash_rows(cfg, rows);
    else
        h = olhoffcurrent_config_hash(cfg);
        [h2, L] = mig_hash_rows(cfg, rows);
        assert(strcmp(h, h2), 'mig:hash', 'mig_hash_rows diverges from olhoffcurrent_config_hash');
    end
    D(end+1) = struct('case', H{i,1}, 'mesh', H{i,2}, 'recordedHash', H{i,3}, ...
        'recordedSource', H{i,4}, 'cfg', cfg, 'sideHash', h, 'lines', {L}); %#ok<AGROW>
    fprintf('%-4s %-5s %4dx%-4d side=%s recorded=%s match=%d\n', side, H{i,1}, H{i,2}(1), H{i,2}(2), ...
        h(1:12), H{i,3}(1:min(12,end)), strcmp(h, H{i,3}));
end

meta = struct('side', side, 'nRows', numel(rows), 'matlab', version, 'when', char(datetime('now')));
if ~strcmp(side, 'up')
    meta.treeHash = olhoffcurrent_source_manifest('Verify', false).treeHash;
end
if strcmp(side, 'post')
    % preset registry as the wrapper reports it
    meta.registry = olhoffcurrent_presets();
end
save(outMat, 'D', 'rows', 'meta', '-v7.3');
end
