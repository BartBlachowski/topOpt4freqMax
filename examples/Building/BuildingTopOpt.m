clear; clc;
close all;

scriptDir = fileparts(mfilename('fullpath'));
prevDir = pwd;
cleanupObj = onCleanup(@() cd(prevDir)); %#ok<NASGU>
cd(scriptDir);

jsonPath = fullfile(scriptDir, 'BuildingTopOptFreq.json');
data = jsondecode(fileread(jsonPath));

res_str="80x240";
forms_str = "_forms_12_";
basename="building";

results = weightedTopologyResultsHelper(data,basename,res_str,forms_str);
