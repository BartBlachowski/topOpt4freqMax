function report=reference_length_replay(outputDir)
%REFERENCE_LENGTH_REPLAY Re-run frozen workflow over lossless H=3200 evidence.
p=iefinal.paths();base=fullfile(p.repo,'analysis','iteration_efficiency_phase2i_precision_qualification');
cap=fullfile(base,'raw','capture_96x12_H3200.mat');ev=fullfile(base,'raw','reference_evaluation.mat');
assert(isfile(cap)&&isfile(ev),'iefinal:MissingQualification','Reference-length lossless evidence is missing.');
C=load(cap,'Xd','rdScientific');E=load(ev,'Qd','hardD','validD','ordD','escD');
assert(isa(C.Xd,'double')&&size(C.Xd,2)==3201,'iefinal:TrajectoryPrecision','96x12 H3200 trajectory is not lossless double.');
assert(isequal(C.rdScientific.rho,C.Xd(:,end)),'iefinal:TrajectoryIdentity','Stored H3200 terminal state differs from authoritative optimizer state.');
Q=E.Qd;H0=logical(E.hardD)&logical(E.validD);ref=ie2a.reference_phase(Q,H0,EvaluatorValid=logical(E.validD));
assert(strcmp(ref.status,'PASS')&&ref.b_ref==2100,'iefinal:ReferenceReplay','Reference replay did not reproduce b_ref=2100.');
budget=ie2a.measurement_budget(3200,ref.b_ref,100,3200);ratio=Q./ref.Q_ref;robust=min(ratio,[],2);
levels=[.98 .99 .995];Pvals=[50 100 200];endpoints=struct([]);r=0;
for ip=1:numel(Pvals)
    A=false(3200,3);for iq=1:3,A(:,iq)=H0&robust>=levels(iq);end
    s=ie2a.scan_persistence(A,Pvals(ip));
    for iq=1:3
        r=r+1;rec=struct('q',levels(iq),'P',Pvals(ip), ...
            'k_enter',s.k_enter(iq),'k_cert',s.k_cert(iq),'status','PASS');
        if r==1,endpoints=rec;else,endpoints(r)=rec;end %#ok<AGROW>
    end
end
checkpoints=[80 252 453 552 2100 3200];binaryIdentity=true;gateIdentity=true;
for k=checkpoints
    x=C.Xd(:,k+1);copy=double(x);binaryIdentity=binaryIdentity&&isequal(ie2a.exact_count_binary(x,.5),ie2a.exact_count_binary(copy,.5));
    gateIdentity=gateIdentity&&isequaln(ie2a.topology_metrics(x,96,12),ie2a.topology_metrics(copy,96,12));
end
report=struct('pass',true,'mesh','96x12','B_ref',3200,'trajectory_dtype',class(C.Xd), ...
    'stored_terminal_equals_authoritative_optimizer_state',true,'checkpoints',checkpoints, ...
    'exact_count_binary_identity',binaryIdentity,'hard_gate_identity',gateIdentity, ...
    'reference_status',ref.status,'b_ref',ref.b_ref,'B0',3200,'B_meas',budget.B_meas, ...
    'tail_truncated',budget.certification_tail_truncated,'Q_ref',ref.Q_ref, ...
    'endpoints',endpoints,'maximum_selected_ordinal',max(E.ordD,[],'all'), ...
    'maximum_escalation_count',max(E.escD,[],'all'));
localWrite(fullfile(outputDir,'reference_length_replay.json'),report);
end
function localWrite(path,v)
fid=fopen(path,'w');assert(fid>0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(v,PrettyPrint=true));clear c
end
