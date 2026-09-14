function cellResult=analyze_cell(tr,B0,cfg)
%ANALYZE_CELL Authoritative reference, measurement, persistence and rows.
% The supplied trajectory is a reference trajectory. Production callers rerun
% the measurement prefix separately and pass that trajectory to BUILD_ROWS.
assert(size(tr.x_post,2)>=cfg.reference_horizon,'iefinal:MissingReferenceTrajectory', ...
    'Required reference trajectory has %d states; expected %d.',size(tr.x_post,2),cfg.reference_horizon);
refTr=tr;refTr.xPhys=tr.x_post(:,1:cfg.reference_horizon);
refA=ie2a.analyze_trajectory(refTr);
ref=ie2a.reference_phase(refA.Q,refA.H0,P=cfg.P_primary,BRef=cfg.B_ref, ...
    SolverTerminated=tr.solver_terminated,EvaluatorValid=refA.evaluator_valid);
if ~strcmp(ref.status,'PASS')
    cellResult=struct('reference',ref,'reference_analysis',refA,'budget',struct(), ...
        'status',ref.status,'rows',struct([]));return
end
budget=ie2a.measurement_budget(B0,ref.b_ref,cfg.P_primary,cfg.B_ref);
cellResult=struct('reference',ref,'reference_analysis',refA,'budget',budget,'status','REFERENCE_PASS','rows',struct([]));
end
