function run_stage_a(method)
%RUN_STAGE_A Execute the preregistered calibration-mesh observer trajectories.

if nargin < 1, method = 'all'; end
method = lower(char(method));
repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
study = fullfile(repo,'analysis','three_method_parametric_study');
outDir = fullfile(study,'raw','stage_a');
if ~exist(outDir,'dir'), mkdir(outDir); end
addpath(fullfile(repo,'tools','Matlab'));
addpath(study);
addpath(fullfile(repo,'Matlab','reproduction2007','runner'));
addpath(fullfile(repo,'analysis','olhoff_native_convergence'));
pathGuard = repro2007_paths(); %#ok<NASGU>
maxNumCompThreads(1);

switch method
    case 'all', methods = {'olhoff','yuksel','proposed'};
    case {'olhoff','yuksel','proposed'}, methods = {method};
    otherwise, error('Unknown method %s.',method);
end
for m=1:numel(methods)
    switch methods{m}
        case 'olhoff', run_olhoff(outDir);
        case 'yuksel', run_yuksel(outDir);
        case 'proposed', run_proposed(outDir);
    end
end
end

function run_olhoff(outDir)
moves=[0.005 0.0075 0.010 0.015 0.020 0.025 0.030];
for i=1:numel(moves)
    mv=moves(i); id=sprintf('olhoff_move_%05d',round(mv*10000));
    path=fullfile(outDir,[id '.mat']);
    if isfile(path), fprintf('SKIP existing %s\n',id); continue; end
    fprintf('START %s %s\n',id,char(datetime('now')));
    overrides=struct('move',mv,'maxOuter',1200,'tolOuter',0.001,'verbose',false,'threads',1);
    [cfg,meta]=repro2007_config('fig3a_best',overrides);
    opts=struct('run_label',id,'store_density_every',1, ...
        'detector_enabled',false,'detector_active_stop',false,'suppress_native_stop',true);
    started=datetime('now','TimeZone','local');
    try
        res=olhoffOptTelemetry(cfg,opts);
        record=struct('run_id',id,'method','Olhoff','status','COMPLETED_OBSERVER', ...
            'started',char(started),'finished',char(datetime('now','TimeZone','local')), ...
            'nelx',240,'nely',30,'move',mv,'tol_outer_observed',0.001, ...
            'wall_time',res.wallclock,'n_iter',res.nOuter,'omega_native',res.omega(1:3), ...
            'native_tol_first_fire',first_true(res.hist.dxOuter<0.001), ...
            'meta',meta,'cfg',res.cfg,'hist',res.hist,'telemetry',res.telemetry, ...
            'x_late',res.rho,'log',{res.log},'mode_table',res.modeTable);
    catch ME
        record=failure_record(id,'Olhoff',started,ME);
    end
    save(path,'record','-v7.3');
    fprintf('DONE %s status=%s\n',id,record.status);
end
end

function run_yuksel(outDir)
cases={ ...
    'yuksel_base',0.2,0.01,0.01; ...
    'yuksel_move_010',0.1,0.01,0.01; ...
    'yuksel_move_030',0.3,0.01,0.01; ...
    'yuksel_tol_both_0005',0.2,0.005,0.005; ...
    'yuksel_tol_both_0020',0.2,0.02,0.02; ...
    'yuksel_tol_s1_0005',0.2,0.005,0.01; ...
    'yuksel_tol_s1_0020',0.2,0.02,0.01; ...
    'yuksel_tol_s2_0005',0.2,0.01,0.005; ...
    'yuksel_tol_s2_0020',0.2,0.01,0.02};
for i=1:size(cases,1)
    id=cases{i,1}; path=fullfile(outDir,[id '.mat']);
    if isfile(path), fprintf('SKIP existing %s\n',id); continue; end
    mv=cases{i,2}; t1=cases{i,3}; t2=cases{i,4};
    fprintf('START %s %s\n',id,char(datetime('now'))); started=datetime('now','TimeZone','local');
    try
        p=struct('move',mv,'stage1_tol',t1,'stage2_tol',t2, ...
            'stage1_max_iters',1000,'max_iters',300,'tol',t2, ...
            'record_history',true,'extend_beyond_native_stop',true);
        cfg=study_base_config('yuksel',240,30,p);
        [x,w,tIter,nIter,mem,nStage,tel]=run_topopt_from_json(cfg);
        record=base_record(id,'Yuksel',started,x,w,tIter,nIter,mem,nStage,tel,cfg);
        record.move=mv; record.stage1_tol=t1; record.stage2_tol=t2;
    catch ME
        record=failure_record(id,'Yuksel',started,ME);
    end
    save(path,'record','-v7.3'); fprintf('DONE %s status=%s\n',id,record.status);
end
end

function run_proposed(outDir)
cases={ ...
    'proposed_base',0.2,0.001; ...
    'proposed_move_010',0.1,0.001; ...
    'proposed_move_030',0.3,0.001; ...
    'proposed_tol_00005',0.2,0.0005; ...
    'proposed_tol_00020',0.2,0.002};
for i=1:size(cases,1)
    id=cases{i,1}; path=fullfile(outDir,[id '.mat']);
    if isfile(path), fprintf('SKIP existing %s\n',id); continue; end
    mv=cases{i,2}; tol=cases{i,3};
    fprintf('START %s %s\n',id,char(datetime('now'))); started=datetime('now','TimeZone','local');
    try
        p=struct('move',mv,'tol',tol,'max_iters',300, ...
            'record_history',true,'extend_beyond_native_stop',true);
        cfg=study_base_config('proposed',240,30,p);
        [x,w,tIter,nIter,mem,nStage,tel]=run_topopt_from_json(cfg);
        record=base_record(id,'Proposed',started,x,w,tIter,nIter,mem,nStage,tel,cfg);
        record.move=mv; record.tol=tol;
    catch ME
        record=failure_record(id,'Proposed',started,ME);
    end
    save(path,'record','-v7.3'); fprintf('DONE %s status=%s\n',id,record.status);
end
end

function r=base_record(id,method,started,x,w,tIter,nIter,mem,nStage,tel,cfg)
r=struct('run_id',id,'method',method,'status','COMPLETED_OBSERVER', ...
    'started',char(started),'finished',char(datetime('now','TimeZone','local')), ...
    'nelx',240,'nely',30,'wall_time',tel.timing.total_wall_time, ...
    'loop_time',tel.timing.optimization_loop_time,'t_iter',tIter,'n_iter',nIter, ...
    'omega_native',w(1:3),'peak_ram_mb',mem,'n_stage',nStage,'telemetry',tel, ...
    'cfg',cfg,'x_late',x,'native_stop_iter',tel.extension.native_stop_iter);
if isfield(tel.extension,'xPhys_at_native_stop') && ~isempty(tel.extension.xPhys_at_native_stop)
    r.x_native=tel.extension.xPhys_at_native_stop;
else
    r.x_native=[];
end
end

function r=failure_record(id,method,started,ME)
r=struct('run_id',id,'method',method,'status','SOLVER_FAILURE', ...
    'started',char(started),'finished',char(datetime('now','TimeZone','local')), ...
    'error_id',ME.identifier,'error_message',ME.message,'error_report',getReport(ME,'extended'));
end

function k=first_true(v)
k=find(v,1,'first'); if isempty(k), k=NaN; end
end
