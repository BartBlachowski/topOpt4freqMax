function run_stage_a2_ours()
%RUN_STAGE_A2_OURS  WP10 Stage A2 boundary mapping for Yuksel and Proposed.
%
%   Stage A1 selected the LOOSEST tolerance tested for both methods, which is
%   a sweep endpoint rather than a knee: it says only that the study did not
%   look far enough out.  These four runs extend the tolerance axis until
%   quality actually degrades, so the knee is located by evidence rather than
%   by where the grid happened to stop.  Effort matches the four-run Olhoff
%   Stage A2 refinement.

repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
study = fullfile(repo,'analysis','three_method_parametric_study');
outDir = fullfile(study,'raw','stage_a_v2');
addpath(fullfile(repo,'tools','Matlab')); addpath(study);
maxNumCompThreads(1);

yuk = {'yuksel_tol_s2_0050', 0.2, 0.01, 0.05; ...
       'yuksel_tol_both_0050',0.2, 0.05, 0.05};
for i = 1:size(yuk,1)
    id = yuk{i,1}; p = fullfile(outDir,[id '.mat']);
    if isfile(p), fprintf('SKIP %s\n',id); continue; end
    fprintf('START %s\n',id); started = datetime('now','TimeZone','local');
    try
        prm = struct('move',yuk{i,2},'stage1_tol',yuk{i,3},'stage2_tol',yuk{i,4}, ...
            'stage1_max_iters',1000,'max_iters',1000,'tol',yuk{i,4}, ...
            'record_history',true,'extend_beyond_native_stop',true);
        cfg = study_base_config('yuksel',240,30,prm);
        [x,w,tIter,nIter,mem,nStage,tel] = run_topopt_from_json(cfg);
        record = mk(id,'Yuksel',started,x,w,tIter,nIter,mem,nStage,tel,cfg);
        record.move = yuk{i,2}; record.stage1_tol = yuk{i,3}; record.stage2_tol = yuk{i,4};
    catch ME, record = fail(id,'Yuksel',started,ME); end
    save(p,'record','-v7.3'); fprintf('DONE %s %s\n',id,record.status);
end

pro = {'proposed_tol_00200', 0.2, 0.02; 'proposed_tol_00500', 0.2, 0.05};
for i = 1:size(pro,1)
    id = pro{i,1}; p = fullfile(outDir,[id '.mat']);
    if isfile(p), fprintf('SKIP %s\n',id); continue; end
    fprintf('START %s\n',id); started = datetime('now','TimeZone','local');
    try
        prm = struct('move',pro{i,2},'tol',pro{i,3},'max_iters',2000, ...
            'record_history',true,'extend_beyond_native_stop',true);
        cfg = study_base_config('proposed',240,30,prm);
        [x,w,tIter,nIter,mem,nStage,tel] = run_topopt_from_json(cfg);
        record = mk(id,'Proposed',started,x,w,tIter,nIter,mem,nStage,tel,cfg);
        record.move = pro{i,2}; record.tol = pro{i,3}; record.budget = 2000;
    catch ME, record = fail(id,'Proposed',started,ME); end
    save(p,'record','-v7.3'); fprintf('DONE %s %s\n',id,record.status);
end
end

function r = mk(id,method,started,x,w,tIter,nIter,mem,nStage,tel,cfg)
r = struct('run_id',id,'method',method,'status','COMPLETED_OBSERVER', ...
    'started',char(started),'finished',char(datetime('now','TimeZone','local')), ...
    'nelx',240,'nely',30,'wall_time',tel.timing.total_wall_time, ...
    'loop_time',tel.timing.optimization_loop_time,'t_iter',tIter,'n_iter',nIter, ...
    'omega_native',w(1:3),'peak_ram_mb',mem,'n_stage',nStage,'telemetry',tel, ...
    'cfg',cfg,'x_late',x,'native_stop_iter',tel.extension.native_stop_iter);
if isfield(tel.extension,'xPhys_at_native_stop') && ~isempty(tel.extension.xPhys_at_native_stop)
    r.x_native = tel.extension.xPhys_at_native_stop;
else, r.x_native = []; end
end

function r = fail(id,method,started,ME)
r = struct('run_id',id,'method',method,'status','SOLVER_FAILURE', ...
    'started',char(started),'finished',char(datetime('now','TimeZone','local')), ...
    'error_id',ME.identifier,'error_message',ME.message,'error_report',getReport(ME,'extended'));
end
