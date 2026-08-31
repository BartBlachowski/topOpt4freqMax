function [rows,report]=run_timing_firewall(rows,outDir,cfg,cells)
%RUN_TIMING_FIREWALL Separate clean fixed-horizon native replays.
% CELLS carries the authoritative native execution evidence produced by the
% measurement trajectories. Every replay must reproduce exactly the native work
% that evidence records; the replay is not permitted to time a different
% computation from the one whose endpoints are reported.
arguments
    rows struct
    outDir (1,:) char
    cfg struct
    cells struct
end
primary=rows([rows.P]==cfg.P_primary);cases=struct([]);n=0;
for i=1:numel(primary)
    for kind={'enter','cert'}
        h=primary(i).(['k_' kind{1}]);if ~isfinite(h),continue;end
        key=sprintf('%s|%s|%s|%d',primary(i).method,primary(i).method_variant,primary(i).mesh,h);
        if any(arrayfun(@(x)strcmp(x.key,key),cases)),continue;end
        n=n+1;rec=struct('key',key,'method',primary(i).method,'variant',primary(i).method_variant, ...
            'nelx',primary(i).nelx,'nely',primary(i).nely,'horizon',h, ...
            'expected',localExpectedWork(primary(i),cells,h));
        if n==1,cases=rec;else,cases(n)=rec;end %#ok<AGROW>
    end
end
samples=struct([]);s=0;
for i=1:numel(cases)
    for rep=0:cfg.timing_repetitions
        x=localOnce(cases(i));localAssertWorkIdentity(cases(i),x);
        s=s+1;rec=struct('key',cases(i).key,'method',cases(i).method, ...
            'method_variant',cases(i).variant,'mesh',sprintf('%dx%d',cases(i).nelx,cases(i).nely), ...
            'horizon',cases(i).horizon,'repetition',rep,'discarded_warmup',rep==0, ...
            'total_seconds',x.total_seconds,'stage1_seconds',x.stage1_seconds,'stage2_seconds',x.stage2_seconds, ...
            'native_iterations',x.native_iterations, ...
            'replay_stage1_iterations',x.stage1_iterations,'replay_stage2_iterations',x.stage2_iterations, ...
            'expected_stage1_iterations',cases(i).expected.stage1,'expected_stage2_iterations',cases(i).expected.stage2, ...
            'work_identity_verified',true);
        if s==1,samples=rec;else,samples(s)=rec;end %#ok<AGROW>
    end
end
if isempty(samples),report=struct('pass',true,'sample_count',0,'reason','no certified endpoints', ...
        'work_identity_verified',true,'work_identity_checks',0);return;end
S=struct2table(samples);writetable(S,fullfile(outDir,'timing_replay_samples.csv'));
for i=1:numel(rows)
    if ~isfinite(rows(i).k_enter),continue;end
    enterKey=sprintf('%s|%s|%s|%d',rows(i).method,rows(i).method_variant,rows(i).mesh,rows(i).k_enter);
    certKey=sprintf('%s|%s|%s|%d',rows(i).method,rows(i).method_variant,rows(i).mesh,rows(i).k_cert);
    te=median(S.total_seconds(strcmp(S.key,enterKey)&~S.discarded_warmup));
    tc=median(S.total_seconds(strcmp(S.key,certKey)&~S.discarded_warmup));
    rows(i).native_total_time=te;rows(i).native_total_time_to_enter=te;rows(i).native_total_time_to_cert=tc;
    rows(i).mean_native_iteration_time=te/max(1,rows(i).native_iterations);
end
report=struct('pass',true,'sample_count',height(S),'warmups_per_case',1,'replays_per_case',cfg.timing_repetitions, ...
    'threads',1,'fixed_horizon',true,'trajectory_capture',false, ...
    'work_identity_verified',true,'work_identity_checks',height(S), ...
    'excluded',{{'common evaluator','topology','persistence','rendering','figure export','trajectory I/O'}});
end

function e=localExpectedWork(row,cells,horizon)
%LOCALEXPECTEDWORK Native work the replay must reproduce, from recorded evidence.
% Sourced from the measurement trajectory, never from a hard-coded compensation.
% Stage 2 (or a single-loop method) runs exactly the requested horizon under
% extension mode; Stage 1 is a native-convergence quantity and is whatever the
% recorded run actually needed, independent of that horizon.
e=struct('stage1',NaN,'stage2',double(horizon),'stage1_budget',NaN);
ix=[];
for i=1:numel(cells)
    tr=cells(i).trajectory;
    if strcmp(tr.method,row.method)&&strcmp(tr.method_variant,row.method_variant)&& ...
            tr.nelx==row.nelx&&tr.nely==row.nely,ix=i;break;end
end
assert(~isempty(ix),'iefinal:TimingEvidenceMissing', ...
    'No authoritative native evidence for %s/%s at %s; refusing to time an unverifiable replay.', ...
    row.method,row.method_variant,row.mesh);
if strcmp(row.method,'Yuksel')
    tr=cells(ix).trajectory;native=tr.native;
    assert(isfield(native,'stage1_updates')&&isfinite(native.stage1_updates), ...
        'iefinal:TimingEvidenceMissing','Yuksel Stage-1 evidence is missing for %s.',row.mesh);
    e.stage1=double(native.stage1_updates);
    % Same Stage-1 budget the recorded run used, so Stage 1 takes the same course.
    assert(isfield(tr,'stage1_budget')&&isfinite(tr.stage1_budget), ...
        'iefinal:TimingEvidenceMissing','Yuksel Stage-1 budget evidence is missing for %s.',row.mesh);
    e.stage1_budget=double(tr.stage1_budget);
end
end

function localAssertWorkIdentity(c,x)
%LOCALASSERTWORKIDENTITY Fail closed if the replay did not do the intended work.
switch c.method
    case 'Yuksel'
        assert(isfinite(x.stage1_iterations)&&x.stage1_iterations==c.expected.stage1, ...
            'iefinal:TimingWorkMismatch', ...
            'Yuksel timing replay Stage-1 work %g does not match the recorded native Stage-1 work %g (%s).', ...
            x.stage1_iterations,c.expected.stage1,c.key);
        assert(isfinite(x.stage2_iterations)&&x.stage2_iterations==c.horizon, ...
            'iefinal:TimingWorkMismatch', ...
            'Yuksel timing replay Stage-2 work %g does not match horizon %g (%s).', ...
            x.stage2_iterations,c.horizon,c.key);
        assert(x.native_iterations==x.stage1_iterations+x.stage2_iterations, ...
            'iefinal:TimingWorkMismatch', ...
            'Yuksel total %g ~= Stage-1 %g + Stage-2 %g (%s).', ...
            x.native_iterations,x.stage1_iterations,x.stage2_iterations,c.key);
    otherwise
        assert(isfinite(x.native_iterations)&&x.native_iterations==c.horizon, ...
            'iefinal:TimingWorkMismatch', ...
            '%s timing replay executed %g native updates, expected horizon %g (%s).', ...
            c.method,x.native_iterations,c.horizon,c.key);
end
end

function out=localOnce(c)
p=iefinal.paths();addpath(fullfile(p.repo,'tools','Matlab'),fullfile(p.repo,'analysis','three_method_parametric_study'), ...
    fullfile(p.repo,'analysis','olhoff_stabilization_audit'),fullfile(p.repo,'Matlab','reproduction2007','runner'));
maxNumCompThreads(1);t=tic;s1=NaN;s2=NaN;nit=NaN;n1=NaN;n2=NaN;
switch c.method
    case {'Proposed','Yuksel'}
        prm=struct('record_history',false,'extend_beyond_native_stop',true,'max_iters',c.horizon);
        if strcmp(c.method,'Proposed'),prm.move=.2;prm.tol=.01;prm.rmin_element=2;
        else
            prm.move=.1;prm.stage1_tol=.01;prm.stage2_tol=.01;prm.rmin_element=2.5;
            % max_iters is the Stage-2 horizon here, so the Stage-1 budget must
            % not be clamped by it; Stage 1 has to run its own native course.
            prm.stage1_max_iters=c.expected.stage1_budget;
            prm.stage1_budget_independent=true;
        end
        cfg0=study_base_config(lower(c.method),c.nelx,c.nely,prm);
        [~,~,~,nit,~,nStage,tel]=run_topopt_from_json(cfg0);
        if isfield(tel,'timing')
            if isfield(tel.timing,'stage1_loop_time'),s1=tel.timing.stage1_loop_time;end
            if isfield(tel.timing,'stage2_loop_time'),s2=tel.timing.stage2_loop_time;end
        end
        if strcmp(c.method,'Yuksel')
            n1=localField(nStage,'stage1',NaN);n2=localField(nStage,'stage2',NaN);
        end
    case 'Olhoff'
        guard=repro2007_paths(); %#ok<NASGU>
        [cfg0,~]=repro2007_config('fig3a_best');cfg0.nelx=c.nelx;cfg0.nely=c.nely;cfg0.maxOuter=c.horizon;cfg0.verbose=false;cfg0.threads=1;cfg0.captureTrajectory=false;
        if strcmp(c.variant,'lp')
            policy=struct('id','S1','move_sequence',[.005 .0025],'gap_threshold',.01,'persistence',100);r=olhoffOptStabilized(cfg0,policy);
        else
            cfg0.rminEl=3;cfg0.rminPhys=.06;cfg0.move=.01;cfg0.tolMult=.05;cfg0.innerSolver='mma';cfg0.offDiag=true;cfg0.filterMode='diag';
            cfg0.maxInner=300;cfg0.tolInner=.01;cfg0.minInner=5;cfg0.captureInnerHistories=false;cfg0.extendBeyondNativeStop=true;r=olhoffOpt(cfg0);
        end
        nit=r.nOuter;
end
out=struct('total_seconds',toc(t),'stage1_seconds',s1,'stage2_seconds',s2, ...
    'native_iterations',nit,'stage1_iterations',n1,'stage2_iterations',n2);
end

function x=localField(s,n,d)
if isstruct(s)&&isfield(s,n)&&~isempty(s.(n)),x=double(s.(n));else,x=d;end
end
