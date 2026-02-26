clear; clc;
close all;

jsonPath = 'BuildingTopOptFreq.json';
data = jsondecode(fileread(jsonPath));

res_str="50x150";
forms_str = "_forms_12_";

% data.optimisation.approach = 'Olhoff';
% [x1, omega1, tIter1, nIter1, mem1] = run_topopt_from_json(data);
% 
% data.optimisation.approach = 'Yuksel';
% [x2, omega2, tIter2, nIter2, mem2] = run_topopt_from_json(data);

data.domain.load_cases(1).factor=1.0;
data.domain.load_cases(2).factor=0.0;

data.optimisation.approach = 'OurApproach';
[x3, omega3, tIter3, nIter3, mem3] = run_topopt_from_json(data);

[status,msg] = movefile("OurApproach_"+res_str+".png","building_1_"+res_str+forms_str+".png");
[status,msg] = movefile("OurApproach_"+res_str+".fig","building_1_"+res_str+forms_str+".fig");

[status,msg] = movefile("topopt_config_topology_mode_1.png","building_1_topopt_config_topology_mode_1.png");
[status,msg] = movefile("topopt_config_topology_mode_2.png","building_1_topopt_config_topology_mode_2.png");


data.domain.load_cases(1).factor=0.75;
data.domain.load_cases(2).factor=0.25;

data.optimisation.approach = 'OurApproach';
[x3, omega3, tIter3, nIter3, mem3] = run_topopt_from_json(data);

[status,msg] = movefile("OurApproach_"+res_str+".png","building_75_"+res_str+forms_str+".png");
[status,msg] = movefile("OurApproach_"+res_str+".fig","building_75_"+res_str+forms_str+".fig");
[status,msg] = movefile("topopt_config_topology_mode_1.png","building_75_topopt_config_topology_mode_1.png");
[status,msg] = movefile("topopt_config_topology_mode_2.png","building_75_topopt_config_topology_mode_2.png");


data.domain.load_cases(1).factor=0.5;
data.domain.load_cases(2).factor=0.5;

data.optimisation.approach = 'OurApproach';
[x3, omega3, tIter3, nIter3, mem3] = run_topopt_from_json(data);

[status,msg] = movefile("OurApproach_"+res_str+".png","building_5_"+res_str+forms_str+".png");
[status,msg] = movefile("OurApproach_"+res_str+".fig","building_5_"+res_str+forms_str+".fig");
[status,msg] = movefile("topopt_config_topology_mode_1.png","building_5_topopt_config_topology_mode_1.png");
[status,msg] = movefile("topopt_config_topology_mode_2.png","building_5_topopt_config_topology_mode_2.png");


data.domain.load_cases(1).factor=0.25;
data.domain.load_cases(2).factor=0.75;

data.optimisation.approach = 'OurApproach';
[x3, omega3, tIter3, nIter3, mem3] = run_topopt_from_json(data);

[status,msg] = movefile("OurApproach_"+res_str+".png","building_25_"+res_str+forms_str+".png");
[status,msg] = movefile("OurApproach_"+res_str+".fig","building_25_"+res_str+forms_str+".fig");
[status,msg] = movefile("topopt_config_topology_mode_1.png","building_25_topopt_config_topology_mode_1.png");
[status,msg] = movefile("topopt_config_topology_mode_2.png","building_25_topopt_config_topology_mode_2.png");


data.domain.load_cases(1).factor=0.0;
data.domain.load_cases(2).factor=1.0;

data.optimisation.approach = 'OurApproach';
[x3, omega3, tIter3, nIter3, mem3] = run_topopt_from_json(data);

[status,msg] = movefile("OurApproach_"+res_str+".png","building_0_"+res_str+forms_str+".png");
[status,msg] = movefile("OurApproach_"+res_str+".fig","building_0_"+res_str+forms_str+".fig");
[status,msg] = movefile("topopt_config_topology_mode_1.png","building_0_topopt_config_topology_mode_1.png");
[status,msg] = movefile("topopt_config_topology_mode_2.png","building_0_topopt_config_topology_mode_2.png");




% data.domain.load_cases(2).loads.mode=4;
% forms_str = "forms_13_";
% 
% 
% data.domain.load_cases(1).factor=1.0;
% data.domain.load_cases(2).factor=0.0;
% 
% data.optimisation.approach = 'OurApproach';
% [x3, omega3, tIter3, nIter3, mem3] = run_topopt_from_json(data);
% 
% [status,msg] = movefile("OurApproach_"+res_str+".png","building_1_"+res_str+forms_str+".png");
% [status,msg] = movefile("OurApproach_"+res_str+".fig","building_1_"+res_str+forms_str+".fig");
% 
% 
% data.domain.load_cases(1).factor=0.75;
% data.domain.load_cases(2).factor=0.25;
% 
% data.optimisation.approach = 'OurApproach';
% [x3, omega3, tIter3, nIter3, mem3] = run_topopt_from_json(data);
% 
% [status,msg] = movefile("OurApproach_"+res_str+".png","building_75_"+res_str+forms_str+".png");
% [status,msg] = movefile("OurApproach_"+res_str+".fig","building_75_"+res_str+forms_str+".fig");
% 
% 
% 
% data.domain.load_cases(1).factor=0.5;
% data.domain.load_cases(2).factor=0.5;
% 
% data.optimisation.approach = 'OurApproach';
% [x3, omega3, tIter3, nIter3, mem3] = run_topopt_from_json(data);
% 
% [status,msg] = movefile("OurApproach_"+res_str+".png","building_5_"+res_str+forms_str+".png");
% [status,msg] = movefile("OurApproach_"+res_str+".fig","building_5_"+res_str+forms_str+".fig");
% 
% 
% 
% data.domain.load_cases(1).factor=0.25;
% data.domain.load_cases(2).factor=0.75;
% 
% data.optimisation.approach = 'OurApproach';
% [x3, omega3, tIter3, nIter3, mem3] = run_topopt_from_json(data);
% 
% [status,msg] = movefile("OurApproach_"+res_str+".png","building_25_"+res_str+forms_str+".png");
% [status,msg] = movefile("OurApproach_"+res_str+".fig","building_25_"+res_str+forms_str+".fig");
% 
% 
% 
% data.domain.load_cases(1).factor=0.0;
% data.domain.load_cases(2).factor=1.0;
% 
% data.optimisation.approach = 'OurApproach';
% [x3, omega3, tIter3, nIter3, mem3] = run_topopt_from_json(data);
% 
% [status,msg] = movefile("OurApproach_"+res_str+".png","building_0_"+res_str+forms_str+".png");
% [status,msg] = movefile("OurApproach_"+res_str+".fig","building_0_"+res_str+forms_str+".fig");