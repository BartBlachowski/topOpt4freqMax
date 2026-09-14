function summary=run_smoke(cfg,out)
%RUN_SMOKE Minimal real-solver plumbing; never a scientific result.
records=struct([]);topCells=struct([]);topRows=struct([]);maxOrdinal=0;failures=0;
for i=1:height(cfg.methods)
    method=char(cfg.methods.method(i));variant=char(cfg.methods.method_variant(i));label=char(cfg.methods.label(i));
    raw=fullfile(out,'measurement',lower(strrep(label,'-','_')));
    tr=iefinal.run_trajectory(method,variant,cfg.meshes(1,1),cfg.meshes(1,2),cfg.reference_horizon,raw, ...
        Stage1MaxIterations=cfg.reference_horizon);
    tt=tr;tt.xPhys=tr.x_post;a=ie2a.analyze_trajectory(tt);
    ord=a.selected_ordinal;maxOrdinal=max(maxOrdinal,max(ord(isfinite(ord)),[],'all'));
    failures=failures+sum(~a.evaluator_valid);
    rec=struct('method',method,'method_variant',variant,'label',label, ...
        'states',size(tr.x_post,2),'trajectory_dtype',tr.trajectory_dtype, ...
        'evaluator_pass',all(a.evaluator_valid),'selected_ordinals',ord, ...
        'escalation_count',localEscalations(a),'Q',a.Q,'topology_pass',a.H_topology, ...
        'volume_pass',a.H_volume,'hard_gate_pass',a.H_topology&a.H_volume,'native',tr.native);
    cellRec=struct('id',label,'trajectory',tr,'analysis',a);
    accepted=all(a.evaluator_valid)&&a.H0(end);k=NaN;status='SMOKE_UNAVAILABLE';
    if accepted,k=size(tr.x_post,2);status='PASS';end
    rowRec=struct('method',method,'method_variant',variant,'mesh',sprintf('%dx%d',tr.nelx,tr.nely), ...
        'nelx',tr.nelx,'nely',tr.nely,'q',.99,'P',100,'k_enter',k,'status',status,'hard_gate_pass',accepted);
    if i==1,records=rec;topCells=cellRec;topRows=rowRec;
    else,records(i)=rec;topCells(i)=cellRec;topRows(i)=rowRec;end %#ok<AGROW>
    save(fullfile(raw,'smoke.mat'),'tr','a','-v7.3');
end
selector=struct('requested',cfg.olhoff_variant,'executed_variants',{cellstr(cfg.methods.method_variant(3:end))}, ...
    'lp_executed',any(cfg.methods.method_variant=="lp"),'mma_executed',any(cfg.methods.method_variant=="mma"));
localWrite(fullfile(out,'validation','cross_method_candidate_c.json'),records);
localWrite(fullfile(out,'validation','selector.json'),selector);
reference=iefinal.reference_length_replay(fullfile(out,'validation'));
scaling=iefinal.synthetic_scaling_validation(fullfile(out,'analysis'),fullfile(out,'figures'));
topologyArtifacts=iefinal.render_topologies(topCells,topRows,fullfile(out,'topologies'));
summary=struct('status','SMOKE_INTEGRATION_PASS','scientific_results',false,'records',records, ...
    'selector',selector,'maximum_selected_ordinal',maxOrdinal,'adaptive_search_failures',failures, ...
    'reference_length',reference,'scaling_validation',scaling,'topology_artifacts',topologyArtifacts);
end
function e=localEscalations(a)
e=zeros(size(a.selected_ordinal));
for k=1:numel(a.modal),for j=1:3,e(k,j)=a.modal{k}{j}.escalation_count;end,end
end
function localWrite(path,v)
fid=fopen(path,'w');assert(fid>0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(v,PrettyPrint=true));clear c
end
