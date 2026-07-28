function baseline = a4_frozen_baseline_reference(outDir)
%A4_FROZEN_BASELINE_REFERENCE  Locate and validate the immutable V-P2-2 fixture.

expectedSha256 = ...
    '9c3d961bcdf731cf413f0be7d4999b121acffea31d9e11356cb67d4b3f269806';
thisDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(thisDir));
referencePath = fullfile(repoRoot, 'examples', 'Revision_v1', ...
    'reference', 'a4', 'a4_topology_Ninf.csv');
producedTopologyPath = fullfile(outDir, 'a4_topology_inf.csv');

baseline = a4_validate_frozen_baseline( ...
    referencePath, expectedSha256, outDir, producedTopologyPath);
end
