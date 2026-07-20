function a4_persist_phase2_checkpoint(path, checkpoint)
%A4_PERSIST_PHASE2_CHECKPOINT  Persist accumulated telemetry before propagation.
% The file is overwritten atomically through a sibling temporary file so a
% failure cannot erase the last valid checkpoint (Phase-2 specification §4.5).
if nargin < 1 || isempty(path), return; end
[parent, ~, ~] = fileparts(path);
if ~isempty(parent) && ~exist(parent, 'dir'), mkdir(parent); end
tmp = [path '.new'];
save(tmp, 'checkpoint', '-v7.3');
movefile(tmp, path, 'f');
end
