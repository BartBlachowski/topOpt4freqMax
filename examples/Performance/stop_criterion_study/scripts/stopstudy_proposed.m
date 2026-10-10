function stopstudy_proposed(outDir, meshes, maxIter, e1Every)
if nargin < 4, e1Every = 1; end
repo = '/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(repo,'tools','Matlab'));
addpath(fullfile(repo,'examples','Performance','conference_bench'));
addpath(fullfile(repo,'examples','Performance','benchmark_profile'));
addpath(fullfile(repo,'examples','bimodality'));
maxNumCompThreads(1);
for i = 1:size(meshes,1)
    nelx = meshes(i,1); nely = meshes(i,2);
    [cfg, profileId] = confbench_method_config('proposed', nelx, nely);
    cfg.benchmark.record_history = true;
    cfg.benchmark.extend_beyond_native_stop = true;
    cfg.optimization.max_iters = maxIter;
    cfg.optimization.convergence_tol = 0.01;      % production rule, recorded as native_stop_iter
    cfg.postprocessing.record_design_history = true;
    t = tic; [x,~,~,nIter,~,~,telemetry] = run_topopt_from_json(cfg); wall = toc(t);
    X = telemetry.design_history;
    fprintf('CHECK size(X,2)=%d nIter=%d numel(x)=%d maxdiff(X(:,end),x(:))=%g\n', size(X,2), nIter, numel(x), max(abs(X(:,end)-x(:))));
    nIter = size(X,2);
    x0 = cfg.optimization.volume_fraction*ones(size(X,1),1);
    M = stopstudy_metrics(X, x0);
    M.obj = telemetry.history.objective(1:nIter);
    M.elapsed = telemetry.history.elapsed_s(1:nIter);
    M.moveActive = telemetry.history.move_active_frac(1:nIter);
    M.dinfDesign = telemetry.history.d_inf_design(1:nIter);
    idx = unique([1:e1Every:nIter, nIter]);
    t2 = tic; om = e1_structural_omegas(X(:,idx), nelx, nely, 3); tE1 = toc(t2);
    omegaE1 = nan(nIter,3); omegaE1(idx,:) = om;
    stopping = telemetry.stopping; extension = telemetry.extension;
    nativeStopIter = extension.native_stop_iter;
    save(fullfile(outDir, sprintf('proposed_%dx%d.mat', nelx, nely)), ...
        'M','omegaE1','nIter','wall','tE1','profileId','stopping','extension','nativeStopIter','nelx','nely','-v7');
    T = struct2table(rmfield(M, {})); T.w1 = omegaE1(:,1); T.w2 = omegaE1(:,2); T.w3 = omegaE1(:,3);
    writetable(T, fullfile(outDir, sprintf('proposed_%dx%d.csv', nelx, nely)));
    fprintf('DONE proposed %dx%d: %d it (native stop %g), wall %.1f s, E1 %.1f s, w1 %.3f Mnd %.4f\n', ...
        nelx, nely, nIter, nativeStopIter, wall, tE1, omegaE1(end,1), M.Mnd(end));
end
end
