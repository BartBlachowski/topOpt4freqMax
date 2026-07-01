%TEST_TOPOLOGY_QUALITY Verify morphology guard rejects unusable topologies.
%
% Output: scripts/revision_v1/topology_quality_results.json

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));
resultPath = fullfile(scriptDir, 'topology_quality_results.json');

addpath(fullfile(repoRoot, 'tools', 'Matlab'));

nelx = 80;
nely = 20;

% Clean support-spanning solid field: should pass.
clean = ones(nely, nelx);
cleanCheck = checkTopologyQuality(clean, nelx, nely, ...
    'max_solid_components', 20, ...
    'min_largest_solid_fraction', 0.95, ...
    'require_left_right_spanning', true);
localAssert(cleanCheck.pass, 'clean support-spanning topology should pass');

% Speckled field: one support-spanning frame plus many isolated solid islands.
% This catches image-like salt-and-pepper material fragments.
speckled = zeros(nely, nelx);
speckled([1 2 nely-1 nely], :) = 1;
speckled(:, [1 2 nelx-1 nelx]) = 1;
for r = 5:3:(nely-4)
    for c = 6:4:(nelx-5)
        speckled(r, c) = 1;
    end
end
speckledCheck = checkTopologyQuality(speckled, nelx, nely, ...
    'max_solid_components', 20, ...
    'min_largest_solid_fraction', 0.80, ...
    'require_left_right_spanning', true);
localAssert(~speckledCheck.pass, 'speckled topology should fail');
localAssert(speckledCheck.metrics.solid_component_count > ...
    speckledCheck.thresholds.max_solid_components, ...
    'speckled topology should fail by solid component count');

% Split support blocks: component count is low, but no component spans both supports.
split = zeros(nely, nelx);
split(:, 1:12) = 1;
split(:, nelx-11:nelx) = 1;
splitCheck = checkTopologyQuality(split, nelx, nely, ...
    'max_solid_components', 20, ...
    'min_largest_solid_fraction', 0.40, ...
    'require_left_right_spanning', true);
localAssert(~splitCheck.pass, 'split support topology should fail');
localAssert(~splitCheck.metrics.largest_component_touches_both_supports, ...
    'split topology should fail by support spanning');

result = struct();
result.gate = 'TOPOLOGY-QUALITY';
result.status = 'passed';
result.clean = cleanCheck;
result.speckled = speckledCheck;
result.split = splitCheck;
result.timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
result.matlab_version = version;

fid = fopen(resultPath, 'w');
if fid < 0
    error('TopologyQuality:ResultWrite', 'Unable to write %s.', resultPath);
end
cleanupFid = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(result, PrettyPrint=true));

fprintf('\nTOPOLOGY QUALITY TEST PASSED\n');
fprintf('  clean components: %d\n', cleanCheck.metrics.solid_component_count);
fprintf('  speckled components: %d (limit %d)\n', ...
    speckledCheck.metrics.solid_component_count, ...
    speckledCheck.thresholds.max_solid_components);
fprintf('  split spans supports: %d\n', splitCheck.metrics.largest_component_touches_both_supports);
fprintf('  Saved: %s\n', resultPath);

function localAssert(cond, msg)
if ~cond
    error('TopologyQuality:AssertionFailed', '%s', msg);
end
end
