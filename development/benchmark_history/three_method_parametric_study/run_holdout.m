function run_holdout(mode)
%RUN_HOLDOUT  WP17/WP19 cross-resolution validation of the frozen profiles.
%
%   run_holdout('observer')     -- hold-out meshes, stopping suppressed, so the
%                                  predicted stop iteration and the continued
%                                  trajectory are both observable.
%   run_holdout('prospective')  -- calibration mesh, stopping ENABLED, to check
%                                  k_predicted == k_actual (WP19).
%
%   No parameter is retuned here.  Every setting is read from
%   results/profile_freeze_manifest.json, which is written before this runs.

if nargin < 1, mode = 'observer'; end
repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
study = fullfile(repo,'analysis','three_method_parametric_study');
addpath(fullfile(repo,'tools','Matlab')); addpath(study);
addpath(fullfile(repo,'Matlab','reproduction2007','runner'));
addpath(fullfile(repo,'analysis','olhoff_native_convergence'));
pathGuard = repro2007_paths(); %#ok<NASGU>
maxNumCompThreads(1);

man = jsondecode(fileread(fullfile(study,'results','profile_freeze_manifest.json')));
outDir = fullfile(study,'raw',mode);
if ~exist(outDir,'dir'), mkdir(outDir); end

if strcmp(mode,'prospective')
    meshes = [240 30];
else
    meshes = [160 20; 320 40; 400 50];
end

P = man.profiles;
names = fieldnames(P);
for mi = 1:size(meshes,1)
    nelx = meshes(mi,1); nely = meshes(mi,2);
    for pi = 1:numel(names)
        p = P.(names{pi});
        id = sprintf('%s_%dx%d',names{pi},nelx,nely);
        path = fullfile(outDir,[id '.mat']);
        if isfile(path), fprintf('SKIP %s\n',id); continue; end
        fprintf('START %s %s\n',id,char(datetime('now')));
        started = datetime('now','TimeZone','local');
        try
            switch lower(p.method)
                case 'olhoff',   record = runOlhoff(p,id,nelx,nely,mode,started);
                case 'yuksel',   record = runOurs(p,id,nelx,nely,mode,started,'yuksel');
                case 'proposed', record = runOurs(p,id,nelx,nely,mode,started,'proposed');
            end
        catch ME
            record = struct('run_id',id,'method',p.method,'status','SOLVER_FAILURE', ...
                'profile',names{pi},'nelx',nelx,'nely',nely,'mode',mode, ...
                'started',char(started),'finished',char(datetime('now','TimeZone','local')), ...
                'error_id',ME.identifier,'error_message',ME.message, ...
                'error_report',getReport(ME,'extended'));
        end
        save(path,'record','-v7.3');
        fprintf('DONE %s status=%s\n',id,record.status);
    end
end
end

function r = runOlhoff(p,id,nelx,nely,mode,started)
overrides = struct('nelx',nelx,'nely',nely,'move',p.move, ...
    'rminEl',p.rmin_element,'rminPhys',[],'tolMult',p.tol_mult, ...
    'maxOuter',p.max_outer,'tolOuter',p.tol_outer, ...
    'innerSolver',p.inner_solver,'filterMode',p.filter_mode, ...
    'offDiag',false,'verbose',false,'threads',1);
cfg = repro2007_config('fig3a_best',overrides);
d = detectorStruct(p.detector);
opts = struct('run_label',id,'store_density_every',1,'detector',d, ...
    'detector_enabled',true, ...
    'detector_active_stop',strcmp(mode,'prospective'), ...
    'suppress_native_stop',true);
res = olhoffOptTelemetry(cfg,opts);
n = res.nOuter;
loopCum = cumsum(res.hist.tEig(1:n)+res.hist.tGrad(1:n)+res.hist.tInner(1:n));
fire = NaN;
for k = 61:n
    if nativeConvergenceDetector(res.hist,res.telemetry,k,res.cfg,d), fire = k; break; end
end
r = struct('run_id',id,'method','Olhoff','profile',p.profile_id,'status','COMPLETED', ...
    'mode',mode,'nelx',nelx,'nely',nely, ...
    'started',char(started),'finished',char(datetime('now','TimeZone','local')), ...
    'wall_time',res.wallclock,'loop_time',loopCum(n),'n_iter',n, ...
    'practical_stop_iter',fire, ...
    'loop_time_practical',iff(isnan(fire),NaN,loopCum(max(fire,1))), ...
    'omega_native',res.omega(1:3),'cfg',res.cfg,'hist',res.hist, ...
    'telemetry',res.telemetry,'x_late',res.rho,'break_reason',res.telemetry.break_reason);
if isnan(fire)
    r.x_practical = [];
else
    r.x_practical = double(res.telemetry.rho_snapshots(:,fire+1));
end
end

function r = runOurs(p,id,nelx,nely,mode,started,which)
prm = struct('move',p.move,'record_history',true, ...
    'extend_beyond_native_stop',~strcmp(mode,'prospective'), ...
    'rmin_element',p.rmin_element,'max_iters',p.max_iters);
if strcmp(which,'yuksel')
    prm.stage1_tol = p.stage1_tol; prm.stage2_tol = p.stage2_tol;
    prm.stage1_max_iters = p.max_iters; prm.tol = p.stage2_tol;
else
    prm.tol = p.tol;
end
cfg = study_base_config(which,nelx,nely,prm);
[x,w,tIter,nIter,mem,nStage,tel] = run_topopt_from_json(cfg); %#ok<ASGLU>
k = tel.extension.native_stop_iter;
r = struct('run_id',id,'method',p.method,'profile',p.profile_id,'status','COMPLETED', ...
    'mode',mode,'nelx',nelx,'nely',nely, ...
    'started',char(started),'finished',char(datetime('now','TimeZone','local')), ...
    'wall_time',tel.timing.total_wall_time,'loop_time',tel.timing.optimization_loop_time, ...
    'n_iter',nIter,'practical_stop_iter',k, ...
    'loop_time_practical',elapsedAt(tel,k,tel.timing.optimization_loop_time,nIter), ...
    'omega_native',w(1:3),'peak_ram_mb',mem,'n_stage',nStage, ...
    'telemetry',tel,'cfg',cfg,'x_late',x);
if strcmp(mode,'prospective')
    r.x_practical = double(x(:));           % stopping was live: final state IS the stop
elseif ~isnan(k) && ~isempty(tel.extension.xPhys_at_native_stop)
    r.x_practical = double(tel.extension.xPhys_at_native_stop(:));
else
    r.x_practical = [];
end
end

function d = detectorStruct(s)
d = struct('objective_block',s.objective_block,'window',s.window, ...
    'persistence',s.persistence, ...
    'objective_block_drift_tol',s.objective_block_drift_tol, ...
    'objective_phase_recurrence_tol',s.objective_phase_recurrence_tol, ...
    'rho_phase_rms_tol',s.rho_phase_rms_tol, ...
    'topology_phase_turnover_tol',s.topology_phase_turnover_tol, ...
    'modal_window',s.modal_window,'gap_tol',s.gap_tol, ...
    'volume_tol_rel',s.volume_tol_rel,'required_N',s.required_N);
end

function t = elapsedAt(tel,k,loopTime,n)
t = NaN;
if isnan(k), return; end
t = loopTime*k/max(n,1);
if isfield(tel,'history') && isstruct(tel.history) && isfield(tel.history,'iter')
    j = find(tel.history.iter(:) == k, 1, 'first');
    if ~isempty(j) && isfinite(tel.history.elapsed_s(j)), t = tel.history.elapsed_s(j); end
end
end

function v = iff(c,a,b), if c, v=a; else, v=b; end, end
