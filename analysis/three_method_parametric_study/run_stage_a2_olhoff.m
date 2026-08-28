function run_stage_a2_olhoff()
%RUN_STAGE_A2_OLHOFF  WP10 Stage A2 local refinement of the Olhoff move sweep.
%
%   Stage A1 found the scientifically interesting region to be move in
%   [0.005, 0.015]: 0.005/0.010/0.015 keep the reproduced bimodal pair
%   (eigengap <= 1%) while 0.0075 and everything at or above 0.020 do not.
%   The 0.0075 anomaly sits between two valid neighbours, so this refinement
%   maps the validity boundary rather than chasing the fastest configuration.
%   Nothing here refines toward the shortest runtime: 0.0125 and 0.0175 are
%   added on the slow side of the observed knee for exactly that reason.

repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
study = fullfile(repo,'analysis','three_method_parametric_study');
outDir = fullfile(study,'raw','stage_a');
addpath(fullfile(repo,'Matlab','reproduction2007','runner'));
addpath(fullfile(repo,'analysis','olhoff_native_convergence'));
pathGuard = repro2007_paths(); %#ok<NASGU>
maxNumCompThreads(1);

moves = [0.00625 0.00875 0.0125 0.0175];
for i = 1:numel(moves)
    mv = moves(i);
    id = sprintf('olhoff_move_%05d',round(mv*10000));
    path = fullfile(outDir,[id '.mat']);
    if isfile(path), fprintf('SKIP existing %s\n',id); continue; end
    fprintf('START %s %s\n',id,char(datetime('now')));
    overrides = struct('move',mv,'maxOuter',1200,'tolOuter',0.001,'verbose',false,'threads',1);
    [cfg,meta] = repro2007_config('fig3a_best',overrides);
    opts = struct('run_label',id,'store_density_every',1, ...
        'detector_enabled',false,'detector_active_stop',false,'suppress_native_stop',true);
    started = datetime('now','TimeZone','local');
    try
        res = olhoffOptTelemetry(cfg,opts);
        record = struct('run_id',id,'method','Olhoff','status','COMPLETED_OBSERVER', ...
            'started',char(started),'finished',char(datetime('now','TimeZone','local')), ...
            'nelx',240,'nely',30,'move',mv,'tol_outer_observed',0.001, ...
            'wall_time',res.wallclock,'n_iter',res.nOuter,'omega_native',res.omega(1:3), ...
            'native_tol_first_fire',firstTrue(res.hist.dxOuter<0.001), ...
            'meta',meta,'cfg',res.cfg,'hist',res.hist,'telemetry',res.telemetry, ...
            'x_late',res.rho,'log',{res.log},'mode_table',res.modeTable);
    catch ME
        record = struct('run_id',id,'method','Olhoff','status','SOLVER_FAILURE', ...
            'started',char(started),'finished',char(datetime('now','TimeZone','local')), ...
            'error_id',ME.identifier,'error_message',ME.message, ...
            'error_report',getReport(ME,'extended'));
    end
    save(path,'record','-v7.3');
    fprintf('DONE %s status=%s\n',id,record.status);
end
end

function k=firstTrue(v), k=find(v,1,'first'); if isempty(k), k=NaN; end, end
