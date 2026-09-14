function samples = run_timing_replays(plan, meshes, outputDir, authorizationToken)
%RUN_TIMING_REPLAYS Execute reviewed serial fixed-horizon timing replays.
% Endpoints/horizons must already be frozen offline. No gate/evaluator/render
% work occurs inside the timed method invocation.
arguments
    plan table
    meshes table
    outputDir (1,:) char
    authorizationToken (1,:) char
end
assert(strcmp(authorizationToken,'AUTHORIZE_FROZEN_TIMING_REPLAYS_AFTER_ENDPOINT_REVIEW'), ...
    'ie2a:TimingAuthorization','Timing replays are locked pending endpoint review.');
outputDir=ie2a.assert_output_isolated(outputDir,'production');if ~isfolder(outputDir),mkdir(outputDir);end
required={'method','horizon','repetition','discarded_warmup'};assert(all(ismember(required,plan.Properties.VariableNames)),'ie2a:TimingPlan','Timing plan schema is incomplete.');
samples=table();maxNumCompThreads(1);
for i=1:height(plan)
    method=char(plan.method(i));mr=meshes(string(meshes.method)==string(method),:);assert(height(mr)==1,'ie2a:TimingMesh','Exactly one mesh row is required per plan method.');
    s=localTimed(method,mr.nelx,mr.nely,plan.horizon(i));
    components=fieldnames(s);
    for j=1:numel(components)
        row=table(string(method),string(components{j}),double(s.(components{j})),plan.repetition(i),logical(plan.discarded_warmup(i)), ...
            'VariableNames',{'method','component','seconds','repetition','discarded_warmup'});samples=[samples;row]; %#ok<AGROW>
    end
end
writetable(samples,fullfile(outputDir,'timing_replay_samples.csv'));
writetable(ie2a.timing_summary(samples),fullfile(outputDir,'timing_replay_summary.csv'));
end
function s=localTimed(method,nx,ny,horizon)
p=ie2a.paths();addpath(fullfile(p.repo,'tools','Matlab'),fullfile(p.repo,'analysis','three_method_parametric_study'), ...
    fullfile(p.repo,'analysis','olhoff_stabilization_audit'),fullfile(p.repo,'Matlab','reproduction2007','runner'));
switch method
    case {'Proposed','Yuksel'}
        prm=struct('record_history',false,'extend_beyond_native_stop',true,'max_iters',horizon);
        if strcmp(method,'Proposed'),prm.move=.2;prm.tol=.01;prm.rmin_element=2;
        else,prm.move=.1;prm.stage1_tol=.01;prm.stage2_tol=.01;prm.rmin_element=2.5;prm.stage1_max_iters=2000;end
        cfg=study_base_config(lower(method),nx,ny,prm);wall=tic;[~,~,~,~,~,~,tel]=run_topopt_from_json(cfg);total=toc(wall);
        s=struct('T_init',tel.timing.initialization_time,'T_loop_fixed_horizon',tel.timing.optimization_loop_time, ...
            'T_native_finalize',tel.timing.postprocessing_time,'T_result_fixed_horizon',total);
    case 'Olhoff'
        [cfg,~]=repro2007_config('fig3a_best');cfg.nelx=nx;cfg.nely=ny;cfg.maxOuter=horizon;cfg.threads=1;cfg.verbose=false;
        policy=struct('id','S1','move_sequence',[.005 .0025],'gap_threshold',.01,'persistence',100);
        wall=tic;r=olhoffOptStabilized(cfg,policy);total=toc(wall);
        s=struct('T_init',NaN,'T_loop_fixed_horizon',sum(r.hist.tEig+r.hist.tGrad+r.hist.tInner), ...
            'T_native_finalize',NaN,'T_result_fixed_horizon',total);
end
end
