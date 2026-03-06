clear; clc;
close all;

scriptDir = fileparts(mfilename('fullpath'));
prevDir = pwd;
cleanupObj = onCleanup(@() cd(prevDir)); %#ok<NASGU>
cd(scriptDir);

jsonPath = fullfile(scriptDir, 'ClampedHingedBeamTopOptFreq.json');
data = jsondecode(fileread(jsonPath));
res_str="400x50";
forms_str = "_forms_12_";
basename="clamped_hinged_beam";

results = weightedTopologyResultsHelper(data,basename,res_str,forms_str);
