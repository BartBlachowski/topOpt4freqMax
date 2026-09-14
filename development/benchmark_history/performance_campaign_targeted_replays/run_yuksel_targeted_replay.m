function outFile=run_yuksel_targeted_replay()
%RUN_YUKSEL_TARGETED_REPLAY Execute the single authorized Yuksel 800x100 replay.
here=fileparts(mfilename('fullpath'));repo=fileparts(fileparts(here));
addpath(here);addpath(fullfile(repo,'tools'));addpath(fullfile(repo,'tools','Matlab'));
addpath(fullfile(repo,'analysis','YukselApproach','Matlab'));
maxNumCompThreads(1);gate=targeted_replay_config_gate('Yuksel');cfg=gate.replay;

nx=cfg.domain.mesh.nelx;ny=cfg.domain.mesh.nely;L=cfg.domain.size.length;H=cfg.domain.size.height;
extraFixedDofs=supportsToFixedDofs(cfg.bc.supports,nx,ny,L,H);
runCfg=struct('E0',cfg.material.E, ...
    'Emin',cfg.material.E*cfg.void_material.E_min_ratio,'nu',cfg.material.nu, ...
    'rho0',cfg.material.rho,'rho_min',cfg.void_material.rho_min, ...
    'beamL',L,'beamH',H,'conv_tol',cfg.optimization.convergence_tol, ...
    'approach_name',cfg.optimization.approach,'save_frq_iterations',false, ...
    'visualization_quality','regular','visualize_live',false, ...
    'extraFixedDofs',extraFixedDofs,'pasS',[],'pasV',[], ...
    'record_history',true,'extend_beyond_native_stop',false, ...
    'final_modes',3,'stage1_tol',cfg.optimization.yuksel.stage1_tol, ...
    'stage2_tol',cfg.optimization.yuksel.stage2_tol, ...
    'audit_collect',true,'audit_snapshot_every',10);
args=struct('nelx',nx,'nely',ny,'volfrac',cfg.optimization.volume_fraction, ...
    'penal',cfg.optimization.penalization,'rmin_element',cfg.optimization.filter.radius, ...
    'filter_code',1,'filter_boundary','N','eta',0.5,'beta',1.0, ...
    'move',cfg.optimization.move_limit,'stage2_cap',cfg.optimization.max_iters, ...
    'stage1_cap',cfg.optimization.yuksel.stage1_max_iters,'bc_type','none', ...
    'mode_history_modes',0);
assert(args.move==0.1&&args.rmin_element==2.5&&args.stage1_cap==1000&&args.stage2_cap==1000);
assert(runCfg.stage1_tol==0.01&&runCfg.stage2_tol==0.01&&~runCfg.extend_beyond_native_stop);

fprintf('TARGETED_REPLAY_START Yuksel 800x100\n');
t=tic;[xFinal,uFinal,info]=top99neo_inertial_freq(args.nelx,args.nely,args.volfrac, ...
    args.penal,args.rmin_element,args.filter_code,args.filter_boundary,args.eta,args.beta, ...
    args.move,args.stage2_cap,args.stage1_cap,args.bc_type,args.mode_history_modes,runCfg);
diagnosticWall=toc(t);
omega=reshape(double(info.stage2.omegaFinal),[],1);omega(end+1:3,1)=NaN;
status=classify_status(info,args);
original=find_original_run(repo,'Yuksel',nx,ny);
comparison=compare_original(original,xFinal,omega,info,status);
result=struct('method','Yuksel','mesh','800x100','xFinal',xFinal(:), ...
    'uFinal',uFinal,'omega',omega,'info',info,'status',status, ...
    'diagnostic_wall_time_s',diagnosticWall,'timing_role','DIAGNOSTIC ONLY', ...
    'effective_solver_args',args,'runCfg',runCfg,'gate',gate, ...
    'original_comparison',comparison);
outFile=fullfile(here,'raw','yuksel','yuksel_800x100_diagnostic.mat');
save(outFile,'result','-v7.3');
fprintf('TARGETED_REPLAY_DONE Yuksel status=%s stage1=%d stage2=%d endpoint_match=%d\n', ...
    status,info.stage1.iterations,info.stage2.iterations,comparison.endpoint_matches_original);
end

function status=classify_status(info,args)
if contains(lower(info.stage1.stop_reason),'max_iter')||contains(lower(info.stage2.stop_reason),'max_iter')
    status='CAP_HIT';
elseif contains(lower(info.stage2.stop_reason),'tolerance')
    status='NATIVE_CONVERGED';
else,status='UNRECOGNIZED_STOP';end
assert(info.stage1.iterations<=args.stage1_cap&&info.stage2.iterations<=args.stage2_cap);
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

function c=compare_original(o,x,w,info,status)
c=struct();c.iterations_match=o.iterations.iter_total==info.timing.total_iterations;
c.stage1_iterations_match=o.iterations.iter_stage1==info.stage1.iterations;
c.stage2_iterations_match=o.iterations.iter_stage2==info.stage2.iterations;
c.status_match=strcmp(o.stopping.status,status);
c.omega_max_abs_difference=max(abs(double(o.results.final_frequencies_rad_s(:))-w(:)));
c.density_checksum_replay=numeric_fingerprint(x);
c.density_checksum_original=char(o.results.topology_checksum);
c.density_checksum_match=strcmp(c.density_checksum_replay,c.density_checksum_original);
c.objective_checksum_replay=numeric_fingerprint(info.stage2.c);
c.objective_checksum_original=char(o.results.objective_history_checksum);
c.objective_checksum_match=strcmp(c.objective_checksum_replay,c.objective_checksum_original);
c.final_max_dx_difference=abs(o.stopping.final_max_density_change-info.stopping.final_max_density_change);
c.final_rms_dx_difference=abs(o.stopping.final_rms_density_change-info.stopping.final_rms_density_change);
c.endpoint_matches_original=c.iterations_match&&c.stage1_iterations_match&&c.stage2_iterations_match&& ...
    c.status_match&&c.omega_max_abs_difference==0&&c.density_checksum_match&&c.objective_checksum_match&& ...
    c.final_max_dx_difference==0&&c.final_rms_dx_difference==0;
end

function s=numeric_fingerprint(x)
x=double(x(:));w=(1:numel(x))';s=sprintf('n=%d;sum=%.17g;weighted=%.17g;l2=%.17g', ...
    numel(x),sum(x),sum(w.*x),norm(x));
end
