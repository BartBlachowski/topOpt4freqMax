clear; clc;
close all;

jsonPath = 'BeamTopOptFreq.json';
data = jsondecode(fileread(jsonPath));
res_str="400x50";
forms_str = "_forms_12_";
basename = "clamped_beam";

results = weightedTopologyResultsHelper(data,basename,res_str,forms_str);

