function mt_one(arm, nelx, nely)
%MT_ONE  Execute exactly one arm at one mesh and archive it.
%   Each invocation forces maxNumCompThreads(1) inside mt_run, so several
%   invocations may run concurrently without affecting any trajectory.
repo = fileparts(fileparts(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))))));
addpath(fullfile(repo,'analysis','OlhoffCurrent'));
addpath(fullfile(repo,'analysis','OlhoffCurrent','diagnostics','move_transition','code'));
olhoffcurrent_scrub_forbidden_paths(repo);
runsDir = fullfile(repo,'analysis','OlhoffCurrent','diagnostics','move_transition','runs');
out = mt_run(arm, nelx, nely, runsDir); %#ok<NASGU>
end
