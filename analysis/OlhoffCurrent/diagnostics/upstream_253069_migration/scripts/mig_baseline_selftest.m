function mig_baseline_selftest(repoRoot, outFile)
%MIG_BASELINE_SELFTEST  confbench_selftest on a PRISTINE pre-migration tree (git archive 013cc48).
%   Classifies harness self-test failures as pre-existing or migration-caused:
%   same driver path setup as performance_comparison.m, rooted at repoRoot.
restoredefaultpath;
scriptDir = fullfile(repoRoot, 'examples', 'Performance');
addpath(scriptDir);
addpath(fullfile(scriptDir, 'conference_bench'));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'analysis', 'three_method_parametric_study'));
addpath(fullfile(repoRoot, 'analysis', 'OlhoffCurrent'));
olhoffcurrent_scrub_forbidden_paths(repoRoot);
maxNumCompThreads(1);
assert(strncmp(which('confbench_selftest'), repoRoot, numel(repoRoot)), 'mig:baseline', 'selftest not from baseline');
st = confbench_selftest(outFile);
fprintf('BASELINE failed ids: %s\n', strjoin({st.tests(~[st.tests.pass]).id}, ', '));
end
