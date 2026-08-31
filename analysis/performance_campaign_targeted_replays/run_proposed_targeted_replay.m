function outFile=run_proposed_targeted_replay(runId)
%RUN_PROPOSED_TARGETED_REPLAY Execute one of two identical Proposed replays.
if nargin<1,runId=1;end
assert(ismember(runId,[1 2]),'Only determinism runs 1 and 2 are authorized.');
here=fileparts(mfilename('fullpath'));repo=fileparts(fileparts(here));
addpath(here);addpath(fullfile(repo,'tools'));addpath(fullfile(repo,'tools','Matlab'));
addpath(fullfile(repo,'analysis','ourApproach','Matlab'));
addpath(fullfile(repo,'analysis','three_method_parametric_study'));
maxNumCompThreads(1);gate=targeted_replay_config_gate('Proposed');cfg=gate.replay;

nx=cfg.domain.mesh.nelx;ny=cfg.domain.mesh.nely;L=cfg.domain.size.length;H=cfg.domain.size.height;
extraFixedDofs=supportsToFixedDofs(cfg.bc.supports,nx,ny,L,H);
runCfg=struct('E0',cfg.material.E, ...
    'Emin',cfg.material.E*cfg.void_material.E_min_ratio,'nu',cfg.material.nu, ...
    'rho0',cfg.material.rho,'rho_min',cfg.void_material.rho_min, ...
    'move',cfg.optimization.move_limit,'conv_tol',cfg.optimization.convergence_tol, ...
    'max_iters',cfg.optimization.max_iters,'supportType','NONE', ...
    'approach_name',cfg.optimization.approach,'save_frq_iterations',true, ...
    'visualization_quality','regular','visualize_live',false, ...
    'extraFixedDofs',extraFixedDofs,'pasS',[],'pasV',[], ...
    'record_history',true,'extend_beyond_native_stop',false, ...
    'harmonic_normalize',cfg.optimization.harmonic_normalize, ...
    'semi_harmonic_baseline',cfg.optimization.semi_harmonic_baseline, ...
    'semi_harmonic_load_sensitivity',cfg.optimization.semi_harmonic_load_sensitivity, ...
    'load_cases',cfg.domain.load_cases,'optimizer',cfg.optimization.optimizer);
args=struct('nelx',nx,'nely',ny,'volfrac',cfg.optimization.volume_fraction, ...
    'penal',cfg.optimization.penalization, ...
    'rmin_physical',cfg.optimization.filter.radius*(L/nx), ...
    'filter_code',0,'L',L,'H',H);
assert(runCfg.move==0.2&&runCfg.conv_tol==0.01&&runCfg.max_iters==2000);
assert(args.rmin_physical==0.1&&strcmp(runCfg.optimizer,'OC')&&~runCfg.harmonic_normalize);
assert(~runCfg.semi_harmonic_load_sensitivity&&strcmp(runCfg.semi_harmonic_baseline,'solid'));

fprintf('TARGETED_REPLAY_START Proposed 160x20 run=%d\n',runId);
t=tic;[xFinal,fHz,tIter,nIter,info]=topopt_freq(args.nelx,args.nely,args.volfrac, ...
    args.penal,args.rmin_physical,args.filter_code,args.L,args.H,runCfg);
diagnosticWall=toc(t);omega=2*pi*double(fHz(:));
if strcmp(info.stopping.stop_reason,'density_change_tolerance'),status='NATIVE_CONVERGED'; ...
else,status='CAP_HIT';end
original=find_original_run(repo,'Proposed',nx,ny);
comparison=compare_original(original,xFinal,omega,info,nIter,status);
offlineTic=tic;common=study_evaluate_design(xFinal,nx,ny,args.volfrac);offlineTime=toc(offlineTic);
result=struct('method','Proposed','mesh','160x20','run_id',runId,'xFinal',xFinal(:), ...
    'omega',omega,'fHz',fHz(:),'tIter_diagnostic',tIter,'nIter',nIter,'info',info, ...
    'status',status,'diagnostic_wall_time_s',diagnosticWall,'timing_role','DIAGNOSTIC ONLY', ...
    'offline_common_evaluator_time_s',offlineTime,'common_evaluators',common, ...
    'effective_solver_args',args,'runCfg',runCfg,'gate',gate,'original_comparison',comparison);
outFile=fullfile(here,'raw','proposed',sprintf('proposed_160x20_diagnostic_run%d.mat',runId));
save(outFile,'result','-v7.3');
fprintf('TARGETED_REPLAY_DONE Proposed run=%d status=%s n=%d omega1=%.12g endpoint_match=%d\n', ...
    runId,status,nIter,omega(1),comparison.endpoint_matches_original);
end

function r=find_original_run(repo,method,nx,ny)
j=jsondecode(fileread(fullfile(repo,'examples','Performance','final_campaign','benchmark_results.json')));
r=struct();
for i=1:numel(j.runs)
    q=j.runs(i);
    if strcmp(q.method,method)&&q.mesh.nelx==nx&&q.mesh.nely==ny,r=q;return;end
end
error('Original run not found.');
end

function c=compare_original(o,x,w,info,nIter,status)
c=struct();c.iterations_match=o.iterations.iter_total==nIter;c.status_match=strcmp(o.stopping.status,status);
c.omega_max_abs_difference=max(abs(double(o.results.final_frequencies_rad_s(:))-w(:)));
c.density_checksum_replay=numeric_fingerprint(x);c.density_checksum_original=char(o.results.topology_checksum);
c.density_checksum_match=strcmp(c.density_checksum_replay,c.density_checksum_original);
c.objective_checksum_replay=numeric_fingerprint(info.objective_history);
c.objective_checksum_original=char(o.results.objective_history_checksum);
c.objective_checksum_match=strcmp(c.objective_checksum_replay,c.objective_checksum_original);
c.final_max_dx_difference=abs(o.stopping.final_max_density_change-info.stopping.final_max_density_change);
c.final_rms_dx_difference=abs(o.stopping.final_rms_density_change-info.stopping.final_rms_density_change);
c.endpoint_matches_original=c.iterations_match&&c.status_match&&c.omega_max_abs_difference==0&& ...
    c.density_checksum_match&&c.objective_checksum_match&&c.final_max_dx_difference==0&&c.final_rms_dx_difference==0;
end

function s=numeric_fingerprint(x)
x=double(x(:));w=(1:numel(x))';s=sprintf('n=%d;sum=%.17g;weighted=%.17g;l2=%.17g', ...
    numel(x),sum(x),sum(w.*x),norm(x));
end
