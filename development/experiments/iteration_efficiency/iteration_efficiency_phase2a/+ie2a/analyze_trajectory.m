function a = analyze_trajectory(tr, qRef, levels)
%ANALYZE_TRAJECTORY Offline common evaluators and structural gates.
arguments
    tr struct
    qRef (1,3) double = [NaN NaN NaN]
    levels (1,:) double = [.98 .99 .995]
end
n=size(tr.xPhys,2);Q=nan(n,3);Htop=false(n,1);Hvol=false(n,1);top=cell(n,1);
evaluatorValid=false(n,1);evaluatorStatus=strings(n,1);selectedOrdinal=nan(n,3);modal=cell(n,1);
for k=1:n
    x=tr.xPhys(:,k);t=ie2a.topology_metrics(x,tr.nelx,tr.nely);e=ie2a.evaluate_common(x,tr.nelx,tr.nely,.5);
    Q(k,:)=e.Q;Htop(k)=t.topology_pass;Hvol(k)=t.volume_pass;top{k}=rmfield(t,'binary');
    evaluatorValid(k)=strcmp(e.status,'PASS');evaluatorStatus(k)=string(e.status);
    selectedOrdinal(k,:)=e.selected_ordinal;modal{k}=e.modal;
end
Hhealth=all(isfinite(tr.xPhys),1).'&evaluatorValid&all(isfinite(Q),2);Hmethod=logical(tr.method_gate(:));
H0=Hhealth&Hvol&Htop&Hmethod;
a=struct('Q',Q,'H_health',Hhealth,'H_volume',Hvol,'H_topology',Htop,'H_method',Hmethod, ...
    'H0',H0,'topology',{top},'evaluator_valid',evaluatorValid, ...
    'evaluator_status',evaluatorStatus,'selected_ordinal',selectedOrdinal,'modal',{modal});
if all(isfinite(qRef))
    ratio=Q./qRef;robust=min(ratio,[],2);pass=false(n,numel(levels));
    for j=1:numel(levels),pass(:,j)=H0&robust>=levels(j);end
    a.ratio=ratio;a.robust_ratio=robust;a.acceptance=pass;a.persistence=ie2a.scan_persistence(pass,100);
end
end
