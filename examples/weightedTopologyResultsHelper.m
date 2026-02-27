function results = weightedTopologyResultsHelper(data,basename,res_str,forms_str)

    data.domain.load_cases(1).factor=1.0;
    data.domain.load_cases(2).factor=0.0;

    
    data.optimisation.approach = 'OurApproach';
    [x3, omega3, tIter3, nIter3, mem3] = run_topopt_from_json(data);
    results = packResultRow(x3, omega3, tIter3, nIter3, mem3);

    
    [status,msg] = movefile("OurApproach_"+res_str+".png",basename+"_1_"+res_str+forms_str+".png");
    [status,msg] = movefile("OurApproach_"+res_str+".fig",basename+"_1_"+res_str+forms_str+".fig");
    
    [status,msg] = movefile("topopt_config_topology_mode_1.png",basename+"_1_topopt_config_topology_mode_1.png");
    [status,msg] = movefile("topopt_config_topology_mode_2.png",basename+"_1_topopt_config_topology_mode_2.png");
    [status,msg] = movefile("topopt_config_correlation.csv",basename+"_1_correlation.csv");
    
    
    
    data.domain.load_cases(1).factor=0.75;
    data.domain.load_cases(2).factor=0.25;
    
    data.optimisation.approach = 'OurApproach';
    [x3, omega3, tIter3, nIter3, mem3] = run_topopt_from_json(data);
    results = [results; packResultRow(x3, omega3, tIter3, nIter3, mem3)];
    
    [status,msg] = movefile("OurApproach_"+res_str+".png",basename+"_75_"+res_str+forms_str+".png");
    [status,msg] = movefile("OurApproach_"+res_str+".fig",basename+"_75_"+res_str+forms_str+".fig");
    [status,msg] = movefile("topopt_config_topology_mode_1.png",basename+"_75_topopt_config_topology_mode_1.png");
    [status,msg] = movefile("topopt_config_topology_mode_2.png",basename+"_75_topopt_config_topology_mode_2.png");
    [status,msg] = movefile("topopt_config_correlation.csv",basename+"_75_correlation.csv");
    
    
    data.domain.load_cases(1).factor=0.5;
    data.domain.load_cases(2).factor=0.5;
    
    data.optimisation.approach = 'OurApproach';
    [x3, omega3, tIter3, nIter3, mem3] = run_topopt_from_json(data);
    results = [results; packResultRow(x3, omega3, tIter3, nIter3, mem3)];
    
    [status,msg] = movefile("OurApproach_"+res_str+".png",basename+"_5_"+res_str+forms_str+".png");
    [status,msg] = movefile("OurApproach_"+res_str+".fig",basename+"_5_"+res_str+forms_str+".fig");
    [status,msg] = movefile("topopt_config_topology_mode_1.png",basename+"_5_topopt_config_topology_mode_1.png");
    [status,msg] = movefile("topopt_config_topology_mode_2.png",basename+"_5_topopt_config_topology_mode_2.png");
    [status,msg] = movefile("topopt_config_correlation.csv",basename+"_5_correlation.csv");
    
    
    data.domain.load_cases(1).factor=0.25;
    data.domain.load_cases(2).factor=0.75;
    
    data.optimisation.approach = 'OurApproach';
    [x3, omega3, tIter3, nIter3, mem3] = run_topopt_from_json(data);
    results = [results; packResultRow(x3, omega3, tIter3, nIter3, mem3)];
    
    [status,msg] = movefile("OurApproach_"+res_str+".png",basename+"_25_"+res_str+forms_str+".png");
    [status,msg] = movefile("OurApproach_"+res_str+".fig",basename+"_25_"+res_str+forms_str+".fig");
    [status,msg] = movefile("topopt_config_topology_mode_1.png",basename+"_25_topopt_config_topology_mode_1.png");
    [status,msg] = movefile("topopt_config_topology_mode_2.png",basename+"_25_topopt_config_topology_mode_2.png");
    [status,msg] = movefile("topopt_config_correlation.csv",basename+"_25_correlation.csv");
    
    
    data.domain.load_cases(1).factor=0.0;
    data.domain.load_cases(2).factor=1.0;
    
    data.optimisation.approach = 'OurApproach';
    [x3, omega3, tIter3, nIter3, mem3] = run_topopt_from_json(data);
    results = [results; packResultRow(x3, omega3, tIter3, nIter3, mem3)];
    
    [status,msg] = movefile("OurApproach_"+res_str+".png",basename+"_0_"+res_str+forms_str+".png");
    [status,msg] = movefile("OurApproach_"+res_str+".fig",basename+"_0_"+res_str+forms_str+".fig");
    [status,msg] = movefile("topopt_config_topology_mode_1.png",basename+"_0_topopt_config_topology_mode_1.png");
    [status,msg] = movefile("topopt_config_topology_mode_2.png",basename+"_0_topopt_config_topology_mode_2.png");
    [status,msg] = movefile("topopt_config_correlation.csv",basename+"_0_correlation.csv");
end

function row = packResultRow(x, omega, tIter, nIter, mem)
    row = [reshape(x, 1, []), ...
           reshape(omega, 1, []), ...
           reshape(tIter, 1, []), ...
           reshape(nIter, 1, []), ...
           reshape(mem, 1, [])];
end
