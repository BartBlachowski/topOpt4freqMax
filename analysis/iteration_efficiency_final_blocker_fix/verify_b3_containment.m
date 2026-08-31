function report=verify_b3_containment(outDir)
%VERIFY_B3_CONTAINMENT End-to-end cell-local failure containment evidence.
% Reproduces the production campaign's cell loop exactly as RUN.M implements it
% (try/catch -> classify -> RUN_ERROR row -> checkpoint -> continue, with
% integrity failures rethrown) over a multi-mesh grid, with genuine method
% failures injected. Meshes are never below the 160x20 production floor.
if nargin<1||isempty(outDir),outDir=tempname('/tmp');end
if ~isfolder(outDir),mkdir(outDir);end
repo=fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repo,'analysis','iteration_efficiency_final'), ...
    fullfile(repo,'analysis','iteration_efficiency_phase2a'), ...
    fullfile(repo,'analysis','three_method_parametric_study'), ...
    fullfile(repo,'analysis','iteration_efficiency_study_design'), ...
    fullfile(repo,'tools','Matlab'));
cfg=iefinal.config('smoke','lp');
meshes=[160 20;240 30;320 40];              % production-floor meshes
methods={{'Proposed','proposed',900},{'Olhoff','lp',3200}};
% Injected genuine method failures: an EARLY mesh and the FINAL mesh.
inject=containers.Map( ...
  {'Olhoff|lp|160x20','Proposed|proposed|320x40'}, ...
  {MException('iefinal:OptimizerFailure','Olhoff LP failed at outer 627.'), ...
   MException('MATLAB:eigs:ARPACKroutineError','eigs did not converge.')});

allRows=struct([]);failures=struct([]);fc=0;cells=struct([]);tc=0;order={};
for mi=1:size(meshes,1)
    for k=1:numel(methods)
        m=methods{k};method=m{1};variant=m{2};B0=m{3};
        mesh=sprintf('%dx%d',meshes(mi,1),meshes(mi,2));
        id=sprintf('%s_%s',lower(method),mesh);
        key=sprintf('%s|%s|%s',method,variant,mesh);
        hashes=struct('evaluator',cfg.manifest.common_evaluator.sha256);
        try
            if isKey(inject,key),throw(inject(key));end
            rows=localSuccessRows(method,variant,meshes(mi,:),B0,cfg,hashes,mi);
            rec=struct('id',id,'trajectory',struct('method',method,'method_variant',variant, ...
                'nelx',meshes(mi,1),'nely',meshes(mi,2),'x_post',repmat(0.5,prod(meshes(mi,:)),140), ...
                'native',struct('total_updates',1)),'analysis',struct());
            tc=tc+1;if tc==1,cells=rec;else,cells(tc)=rec;end
            order{end+1}=[key ' -> PASS']; %#ok<AGROW>
        catch ME
            v=iefinal.classify_cell_failure(ME);
            if ~v.cell_local,rethrow(ME);end
            rows=iefinal.build_error_rows(method,variant,method,meshes(mi,1),meshes(mi,2),B0,cfg,hashes,v);
            fc=fc+1;f=struct('id',id,'mesh',mesh,'method',method,'method_variant',variant, ...
                'identifier',v.identifier,'message',v.message,'class',v.class);
            if fc==1,failures=f;else,failures(fc)=f;end
            order{end+1}=[key ' -> RUN_ERROR (' v.identifier ')']; %#ok<AGROW>
        end
        iefinal.validate_results(rows);
        if isempty(allRows),allRows=rows;else,allRows=[allRows;rows];end %#ok<AGROW>
        save(fullfile(outDir,'rows_checkpoint.mat'),'allRows','failures','-v7.3');
    end
end

% ---- assertions ----
st=string({allRows.status});
r=struct();
r.total_cells=size(meshes,1)*numel(methods);
r.executed_cells=numel(order);
r.campaign_completed_all_cells=(r.executed_cells==r.total_cells);
r.failed_cells=fc;
r.run_error_rows=sum(st=="RUN_ERROR");
r.pass_rows=sum(st=="PASS");
r.rows_per_cell=numel(allRows)/r.total_cells;
r.cell_order=order;
% every failed cell contributes a full 9-row q/P block, all N/A
e=allRows(st=="RUN_ERROR");
r.all_run_error_have_identifier=all(~cellfun(@isempty,{e.error_identifier}));
r.all_run_error_na=all(isnan([e.k_enter]))&&all(isnan([e.k_cert]))&&all(isnan([e.E1]))&& ...
    all(isnan([e.E2]))&&all(isnan([e.E3]))&&all(isnan([e.Q]))&&all(isnan([e.b_ref]))&& ...
    all(isnan([e.B_meas]))&&all(isnan([e.native_iterations]))&&all(isnan([e.native_total_time]));
r.identity_preserved=all(~cellfun(@isempty,{e.mesh}))&&all(~cellfun(@isempty,{e.method}))&& ...
    all(strcmp({e.contract_hash},cfg.manifest.scientific_contract.sha256));
r.failure_after_early_mesh_then_success=any(contains(order,'160x20')&contains(order,'RUN_ERROR'))&& ...
    any(contains(order,'240x30')&contains(order,'PASS'));
r.failure_in_final_mesh_contained=any(contains(order,'320x40')&contains(order,'RUN_ERROR'));
% ---- tables ----
iefinal.write_results(allRows,localMakeOut(outDir));
T=readtable(fullfile(outDir,'tables','results.csv'),'TextType','string');
r.table_rows=height(T);r.table_has_run_error=any(T.status=="RUN_ERROR");
% ---- scaling: RUN_ERROR must be excluded and must not enter common support ----
sc=iefinal.generate_scaling_outputs(allRows,fullfile(outDir,'analysis'),false);
r.scaling_generated=sc.generated;
if sc.generated
    sup=readtable(fullfile(outDir,'analysis','scaling_common_support.csv'),'TextType','string');
    ke=sup(sup.metric=="k_enter",:);
    cm=string(ke.common_meshes);r.common_support_meshes=char(cm);r.common_support_n=ke.n_support;
    r.common_fit_feasible=logical(ke.common_fit_feasible);
    % Proposed lost 320x40 (51200... no: 12800); Olhoff lost 160x20 (3200)
    r.common_excludes_failed=~contains(cm,"3200")&&~contains(cm,"12800");
end
% ---- topology: failed cells must render as unavailable, never fabricated ----
td=fullfile(outDir,'topologies');mkdir(td);
art=iefinal.render_topologies(cells,allRows,td);
r.topology_files=numel(dir(fullfile(td,'*.png')));
r.topology_artifacts=numel(art);
skipped=0;total=0;
for i=1:numel(art)
    for j=1:numel(art(i).cells)
        total=total+1;
        if isfield(art(i).cells{j},'skipped')&&art(i).cells{j}.skipped,skipped=skipped+1;end
    end
end
r.topology_cells_total=total;r.topology_cells_skipped=skipped;
r.failed_topology_suppressed=(skipped>0);
% ---- integrity failures must still be fatal ----
r.integrity_still_fatal=~iefinal.classify_cell_failure( ...
    MException('iefinal:FingerprintMismatch','x')).cell_local && ...
    ~iefinal.classify_cell_failure(MException('MATLAB:badsubscript','x')).cell_local && ...
    ~iefinal.classify_cell_failure(MException('iefinal:ResultSchema','x')).cell_local;
r.checkpoint_exists=isfile(fullfile(outDir,'rows_checkpoint.mat'));
report=r;disp(r);
fid=fopen(fullfile(outDir,'b3_containment.json'),'w');
fprintf(fid,'%s\n',jsonencode(r,PrettyPrint=true));fclose(fid);
fprintf('B3_CONTAINMENT_DONE outdir=%s\n',outDir);
end

function o=localMakeOut(outDir)
o=outDir;if ~isfolder(fullfile(o,'tables')),mkdir(fullfile(o,'tables'));end
if ~isfolder(fullfile(o,'analysis')),mkdir(fullfile(o,'analysis'));end
end

function rows=localSuccessRows(method,variant,mesh,B0,cfg,hashes,mi)
Pvals=[cfg.P_primary cfg.P_sensitivity];levels=cfg.quality_levels;
rows=repmat(iefinal.empty_result_row(),numel(Pvals)*numel(levels),1);r=0;
for ip=1:numel(Pvals)
    for iq=1:numel(levels)
        r=r+1;z=iefinal.empty_result_row();
        z.schema_version='iteration_efficiency_result_v1';z.method=method;z.method_variant=variant;
        z.nelx=mesh(1);z.nely=mesh(2);z.element_count=prod(mesh);z.mesh=sprintf('%dx%d',mesh(1),mesh(2));
        z.q=levels(iq);z.P=Pvals(ip);z.B0=B0;z.B_ref=cfg.B_ref;z.b_ref=2100;z.B_meas=3200;
        z.tail_truncated=false;z.k_enter=100+10*mi;z.k_cert=z.k_enter+Pvals(ip)-1;z.status='PASS';
        z.E1=160+mi;z.E2=160+mi;z.E3=160+mi;z.Q=.99;
        z.topology_pass=true;z.volume_pass=true;z.hard_gate_pass=true;
        z.native_iterations=z.k_enter;z.native_total_time=1+mi;
        z.native_total_time_to_enter=1+mi;z.native_total_time_to_cert=2+mi;
        z.mean_native_iteration_time=(1+mi)/z.k_enter;
        if strcmp(variant,'lp'),z.olhoff_outer_updates=z.k_enter;z.olhoff_lp_calls=z.k_enter;end
        z.trajectory_dtype='double';z.evaluator_id='candidate_c_adaptive_structural_mode_v1';
        z.contract_hash=cfg.manifest.scientific_contract.sha256;z.source_hashes=hashes;
        rows(r)=z;
    end
end
end
