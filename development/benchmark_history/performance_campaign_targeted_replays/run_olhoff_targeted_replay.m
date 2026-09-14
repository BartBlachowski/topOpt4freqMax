function outFile=run_olhoff_targeted_replay()
%RUN_OLHOFF_TARGETED_REPLAY Execute the single authorized Olhoff 640x80 replay.
here=fileparts(mfilename('fullpath'));repo=fileparts(fileparts(here));
addpath(here);addpath(fullfile(repo,'Matlab','reproduction2007','runner'));
addpath(fullfile(repo,'analysis','olhoff_stabilization_audit'));
guard=repro2007_paths(); %#ok<NASGU>
maxNumCompThreads(1);gate=targeted_replay_config_gate('Olhoff');st=gate.original.optimization.stabilization;
[cfg,~]=repro2007_config('fig3a_best');cfg.nelx=640;cfg.nely=80;cfg.maxOuter=1600;cfg.verbose=false;cfg.threads=1;
assert(cfg.rminEl==st.rmin_element&&cfg.move==st.move_initial&&cfg.maxOuter==st.max_iters_expected);
assert(cfg.rhomin==gate.original.optimization.rho_min&&cfg.tolMult==gate.original.optimization.tol_mult);
assert(strcmpi(cfg.innerSolver,gate.original.optimization.inner_solver)&&cfg.offDiag==gate.original.optimization.off_diagonal);
policy=struct('id','S1','move_sequence',[st.move_initial st.move_stabilized], ...
    'gap_threshold',st.gap_threshold,'persistence',st.persistence);
fprintf('TARGETED_REPLAY_START Olhoff 640x80\n');
res=olhoffOptStabilizedDiagnostic(cfg,policy);
originalFile=fullfile(repo,'examples','Performance','final_campaign','raw','olhoff','s1_640x80.mat');
o=load(originalFile,'res');comparison=compare_results(o.res,res);
outFile=fullfile(here,'raw','olhoff','s1_640x80_diagnostic.mat');
save(outFile,'res','comparison','gate','originalFile','-v7.3');
fprintf('TARGETED_REPLAY_DONE Olhoff status=%s n=%d failure=%g verdict=%s\n', ...
    res.status,res.nOuter,res.failure_iteration,comparison.reproduction_verdict);
end

function c=compare_results(a,b)
fields={'omega','N','beta','nInner','dxOuter','vol','degen','multJ','innerConv', ...
    'cumInner','moveLimit','policyStage','trigger','gap12','dRms','moveBoundFraction', ...
    'stronglyMovingFraction','lpFlag','finiteOk','volumeResidual'};
c=struct('original_completed',a.nOuter,'replay_completed',b.nOuter, ...
    'original_failure_attempt',a.failure_iteration,'replay_failure_attempt',b.failure_iteration, ...
    'original_trigger_iterations',a.trigger_iterations,'replay_trigger_iterations',b.trigger_iterations, ...
    'field_comparisons',struct(),'all_history_bit_identical',true);
for i=1:numel(fields)
    f=fields{i};x=double(a.hist.(f));y=double(b.hist.(f));same=isequaln(x,y);
    if isequal(size(x),size(y))
        d=max(abs(x(:)-y(:)),[],'omitnan');if isempty(d),d=0;end
    else,d=Inf;end
    c.field_comparisons.(f)=struct('same_size',isequal(size(x),size(y)), ...
        'bit_identical',same,'max_abs_difference',d);
    c.all_history_bit_identical=c.all_history_bit_identical&&same;
end
c.final_density_bit_identical=isequaln(a.rho,b.rho);
c.final_density_max_abs_difference=max(abs(double(a.rho(:))-double(b.rho(:))));
c.snapshots_bit_identical=isequaln(a.rho_snapshots,b.rho_snapshots);
c.snapshots_max_abs_difference=max(abs(double(a.rho_snapshots(:))-double(b.rho_snapshots(:))));
c.policy_identical=isequaln(a.policy,b.policy);c.config_identical=isequaln(a.cfg,b.cfg);
c.same_status=strcmp(a.status,b.status);c.same_failure_location=a.failure_iteration==b.failure_iteration;
if c.all_history_bit_identical&&c.final_density_bit_identical&&c.snapshots_bit_identical&& ...
        c.policy_identical&&c.config_identical&&c.same_status&&c.same_failure_location
    c.reproduction_verdict='FAILURE_REPRODUCED';
else
    c.reproduction_verdict='REPLAY_NONDETERMINISM';
end
end
