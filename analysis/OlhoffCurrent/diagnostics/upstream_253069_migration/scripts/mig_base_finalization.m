function mig_base_finalization(root, outMat)
restoredefaultpath; addpath(fullfile(root,'analysis','OlhoffCurrent')); addpath(fullfile(root,'analysis','OlhoffCurrent','tests'));
assert(strncmp(which('test_finalization_gate'), root, numel(root)));
n1 = test_evidence_retention();
restoredefaultpath; addpath(fullfile(root,'analysis','OlhoffCurrent')); addpath(fullfile(root,'analysis','OlhoffCurrent','tests'));
n2 = test_finalization_gate();
save(outMat, 'n1', 'n2');
fprintf('BASELINE evidence_retention=%d finalization_gate=%d\n', n1, n2);
end
