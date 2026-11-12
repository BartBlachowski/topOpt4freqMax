function d = results_dir(name)
%RESULTS_DIR  results/<name> under the repository root, created if needed.
d = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results', name);
if ~exist(d, 'dir'), mkdir(d); end
end
