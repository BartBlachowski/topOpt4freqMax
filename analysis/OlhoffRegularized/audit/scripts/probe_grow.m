function probe_grow()
addpath('/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffRegularized/Matlab');
cases={ {16,2,'simply',1}, {16,2,'fixedPinned',1}, {8,4,'cantilever',1}, ...
        {16,2,'simply',.05}, {16,2,'fixedPinned',.05}, {24,4,'simply',1}, {24,4,'fixedPinned',1} };
for i=1:numel(cases)
    c=cases{i};
    rc=struct('verbose',false,'formulation','olhoff','optimizer','lp', ...
        'max_outer_iterations',3,'max_trial_steps',2,'certificate_radius',c{4}, ...
        'move_max',max(c{4},.005),'certificate_mult_tol',1e-12);
    try
        [~,~,in]=topopt_olhoff_regularized(c{1},c{2},.5,3,1.3,.005,c{3},rc);
        h=in.history;
        fprintf('%2dx%-2d %-12s r=%-6g certN=%s grown=%s gain=%s nextGap=%s\n', ...
            c{1},c{2},c{3},c{4},mat2str(h.certificateN),mat2str(h.certificateGrown), ...
            mat2str(h.certificateRelativeGain,3),mat2str(h.certificateNextGap,3));
    catch ME
        fprintf('%2dx%-2d %-12s r=%-6g ERROR %s\n',c{1},c{2},c{3},c{4},ME.message);
    end
end
end
