function run_stage_a_v2(method)
%RUN_STAGE_A_V2 Stage A re-run for Yuksel and Proposed at authoritative budgets.
%
%   The first Stage A pass (raw/stage_a) capped both Yuksel stages and the
%   Proposed loop at 300 iterations.  That cap is BELOW the R3 authoritative
%   safety budgets (Yuksel 10000 per stage, Proposed 2000), and it censored
%   the very quantity this study must measure: the iteration at which each
%   method's own native stopping rule first fires.  Those runs are retained in
%   the ledger as CENSORED_BY_ITERATION_BUDGET and are not eligible for
%   profile selection.  This pass re-runs them with budgets large enough that
%   the native rule, not the cap, ends the observation.
%
%   Extension mode stays on so every native fire is followed by look-ahead
%   iterations (WP11).  Olhoff is not re-run here: its Stage A trajectories
%   are already observer-only over 1200 iterations with per-iteration density
%   snapshots, which is sufficient for offline detector analysis.

if nargin < 1, method = 'all'; end
method = lower(char(method));
repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
study = fullfile(repo,'analysis','three_method_parametric_study');
outDir = fullfile(study,'raw','stage_a_v2');
if ~exist(outDir,'dir'), mkdir(outDir); end
addpath(fullfile(repo,'tools','Matlab'));
addpath(study);
maxNumCompThreads(1);

% Authoritative safety budgets (benchmark_protocol_r3.json).  Reduced from the
% protocol's 10000/2000 only where a smaller number still leaves the native
% rule uncensored with a wide look-ahead margin; every run records whether the
% budget or the native rule ended it.
YUKSEL_BUDGET   = 1000;   % per stage
PROPOSED_BUDGET = 2000;   % R3 value verbatim

switch method
    case 'all', methods = {'yuksel','proposed'};
    case {'yuksel','proposed'}, methods = {method};
    otherwise, error('run_stage_a_v2:UnknownMethod','Unknown method %s.',method);
end
for m = 1:numel(methods)
    switch methods{m}
        case 'yuksel',   run_yuksel(outDir, YUKSEL_BUDGET);
        case 'proposed', run_proposed(outDir, PROPOSED_BUDGET);
    end
end
end

function run_yuksel(outDir, budget)
% Preregistered one-factor levels around the R3 published-case baseline
% (move 0.2, stage tolerances 0.01/0.01), plus the stage-specific diagnostic
% cells that separate which stage a tolerance change actually acts on.
cases = { ...
    'yuksel_base',            0.2,  0.01,  0.01; ...
    'yuksel_move_010',        0.1,  0.01,  0.01; ...
    'yuksel_move_030',        0.3,  0.01,  0.01; ...
    'yuksel_tol_both_0005',   0.2,  0.005, 0.005; ...
    'yuksel_tol_both_0020',   0.2,  0.02,  0.02; ...
    'yuksel_tol_s1_0005',     0.2,  0.005, 0.01; ...
    'yuksel_tol_s1_0020',     0.2,  0.02,  0.01; ...
    'yuksel_tol_s2_0005',     0.2,  0.01,  0.005; ...
    'yuksel_tol_s2_0020',     0.2,  0.01,  0.02};
for i = 1:size(cases,1)
    id = cases{i,1};
    path = fullfile(outDir,[id '.mat']);
    if isfile(path), fprintf('SKIP existing %s\n',id); continue; end
    mv = cases{i,2}; t1 = cases{i,3}; t2 = cases{i,4};
    fprintf('START %s %s\n',id,char(datetime('now')));
    started = datetime('now','TimeZone','local');
    try
        p = struct('move',mv,'stage1_tol',t1,'stage2_tol',t2, ...
            'stage1_max_iters',budget,'max_iters',budget,'tol',t2, ...
            'record_history',true,'extend_beyond_native_stop',true);
        cfg = study_base_config('yuksel',240,30,p);
        [x,w,tIter,nIter,mem,nStage,tel] = run_topopt_from_json(cfg);
        record = base_record(id,'Yuksel',started,x,w,tIter,nIter,mem,nStage,tel,cfg);
        record.move = mv; record.stage1_tol = t1; record.stage2_tol = t2;
        record.stage_budget = budget;
    catch ME
        record = failure_record(id,'Yuksel',started,ME);
    end
    save(path,'record','-v7.3');
    fprintf('DONE %s status=%s\n',id,record.status);
end
end

function run_proposed(outDir, budget)
% The first pass showed the Proposed native tolerance is inert at a 300-cap:
% 0.0005/0.001 never fired and 0.002 fired at 298.  The tolerance levels are
% therefore widened here so the sweep spans firing and non-firing regimes,
% and the move levels are refined to match Olhoff's and Yuksel's arm sizes.
cases = { ...
    'proposed_base',        0.2,  0.001; ...
    'proposed_move_010',    0.1,  0.001; ...
    'proposed_move_015',    0.15, 0.001; ...
    'proposed_move_030',    0.3,  0.001; ...
    'proposed_tol_00005',   0.2,  0.0005; ...
    'proposed_tol_00020',   0.2,  0.002; ...
    'proposed_tol_00050',   0.2,  0.005; ...
    'proposed_tol_00100',   0.2,  0.01};
for i = 1:size(cases,1)
    id = cases{i,1};
    path = fullfile(outDir,[id '.mat']);
    if isfile(path), fprintf('SKIP existing %s\n',id); continue; end
    mv = cases{i,2}; tol = cases{i,3};
    fprintf('START %s %s\n',id,char(datetime('now')));
    started = datetime('now','TimeZone','local');
    try
        p = struct('move',mv,'tol',tol,'max_iters',budget, ...
            'record_history',true,'extend_beyond_native_stop',true);
        cfg = study_base_config('proposed',240,30,p);
        [x,w,tIter,nIter,mem,nStage,tel] = run_topopt_from_json(cfg);
        record = base_record(id,'Proposed',started,x,w,tIter,nIter,mem,nStage,tel,cfg);
        record.move = mv; record.tol = tol; record.budget = budget;
    catch ME
        record = failure_record(id,'Proposed',started,ME);
    end
    save(path,'record','-v7.3');
    fprintf('DONE %s status=%s\n',id,record.status);
end
end

function r = base_record(id,method,started,x,w,tIter,nIter,mem,nStage,tel,cfg)
r = struct('run_id',id,'method',method,'status','COMPLETED_OBSERVER', ...
    'started',char(started),'finished',char(datetime('now','TimeZone','local')), ...
    'nelx',240,'nely',30,'wall_time',tel.timing.total_wall_time, ...
    'loop_time',tel.timing.optimization_loop_time,'t_iter',tIter,'n_iter',nIter, ...
    'omega_native',w(1:3),'peak_ram_mb',mem,'n_stage',nStage,'telemetry',tel, ...
    'cfg',cfg,'x_late',x,'native_stop_iter',tel.extension.native_stop_iter);
if isfield(tel.extension,'xPhys_at_native_stop') && ~isempty(tel.extension.xPhys_at_native_stop)
    r.x_native = tel.extension.xPhys_at_native_stop;
else
    r.x_native = [];
end
end

function r = failure_record(id,method,started,ME)
r = struct('run_id',id,'method',method,'status','SOLVER_FAILURE', ...
    'started',char(started),'finished',char(datetime('now','TimeZone','local')), ...
    'error_id',ME.identifier,'error_message',ME.message,'error_report',getReport(ME,'extended'));
end
