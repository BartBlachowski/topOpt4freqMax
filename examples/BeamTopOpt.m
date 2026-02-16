clear; clc;
close all;

thisFile = mfilename('fullpath');
if isempty(thisFile)
    thisDir = pwd;
else
    thisDir = fileparts(thisFile);
end
repoRoot = fileparts(thisDir);
toolsDir = fullfile(repoRoot, 'tools');
if exist(toolsDir, 'dir') == 7
    addpath(toolsDir);
end

% Compatibility for MATLAB installations that do not expose
% matlab.internal.math.checkInputName used by newer built-ins.
compatDir = fullfile(toolsDir, 'compat');
if exist('matlab.internal.math.checkInputName', 'file') ~= 2 && exist(compatDir, 'dir') == 7
    addpath(compatDir);
end

jsonPath = fullfile(thisDir, 'BeamTopOptFreq.json');
data = jsondecode(fileread(jsonPath));

data.optimisation.approach = 'Olhoff';
[x1, omega1, tIter1, nIter1] = run_topopt_from_json(data);

data.optimisation.approach = 'Yuksel';
[x2, omega2, tIter2, nIter2] = run_topopt_from_json(data);
