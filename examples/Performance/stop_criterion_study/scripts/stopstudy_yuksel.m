function stopstudy_yuksel(outDir, meshes, stage2Cap, stage1Cap, e1Every)
repo = '/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(repo,'tools','Matlab'));
addpath(fullfile(repo,'examples','Performance','conference_bench'));
addpath(fullfile(repo,'examples','Performance','benchmark_profile'));
addpath(fullfile(repo,'examples','bimodality'));
maxNumCompThreads(1);
for i = 1:size(meshes,1)
    nelx = meshes(i,1); nely = meshes(i,2);
    [cfg, profileId] = confbench_method_config('yuksel', nelx, nely);
    cfg.benchmark.record_history = true;
    cfg.benchmark.extend_beyond_native_stop = true;   % stage 2 only; stage 1 handoff stays native
    cfg.optimization.yuksel.stage1_tol = 0.01;        % production
    cfg.optimization.yuksel.stage2_tol = 0.01;        % production
    cfg.optimization.convergence_tol = 0.01;
    cfg.optimization.max_iters = stage2Cap;
    cfg.optimization.yuksel.stage1_max_iters = stage1Cap;
    cfg.optimization.yuksel.stage1_budget_independent = true;   % stage-1 budget = stage1Cap, not max_iters
    cfg.postprocessing.record_design_history = true;
    t = tic; [x,~,~,nIter,~,~,telemetry] = run_topopt_from_json(cfg); wall = toc(t);
    X = telemetry.design_history;
    assert(size(X,2) == nIter && isequal(X(:,end), x(:)));
    x0 = cfg.optimization.volume_fraction*ones(size(X,1),1);
    M = stopstudy_metrics(X, x0);
    H = telemetry.history;
    M.obj = H.objective(1:nIter); M.stage = H.stage(1:nIter); M.elapsed = H.elapsed_s(1:nIter);
    M.moveActive = H.move_active_frac(1:nIter); M.dinfDesign = H.d_inf_design(1:nIter);
    nStage1 = telemetry.stopping.iter_stage1;
    idx = unique([1:e1Every:nIter, nStage1, nStage1+1, nIter]); idx = idx(idx>=1 & idx<=nIter);
    t2 = tic; om = e1_structural_omegas(X(:,idx), nelx, nely, 3); tE1 = toc(t2);
    omegaE1 = nan(nIter,3); omegaE1(idx,:) = om;
    stopping = telemetry.stopping; extension = telemetry.extension;
    save(fullfile(outDir, sprintf('yuksel_%dx%d.mat', nelx, nely)), ...
        'M','omegaE1','nIter','nStage1','wall','tE1','profileId','stopping','extension','nelx','nely','-v7');
    T = struct2table(M); T.w1 = omegaE1(:,1); T.w2 = omegaE1(:,2); T.w3 = omegaE1(:,3);
    writetable(T, fullfile(outDir, sprintf('yuksel_%dx%d.csv', nelx, nely)));
    fprintf('DONE yuksel %dx%d: %d it (stage1 %d), wall %.1f s, E1 %.1f s, w1 %.3f Mnd %.4f\n', ...
        nelx, nely, nIter, nStage1, wall, tE1, om(end,1), M.Mnd(end));
end
end
