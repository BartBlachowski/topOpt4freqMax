function run_production_campaign(outputDir, opts)
%RUN_PRODUCTION_CAMPAIGN Reviewed two-pass campaign orchestration.
% This function has no authorization bypass; call only through the entry script.
arguments
    outputDir (1,:) char
    opts.OlhoffVariant (1,:) char {mustBeMember(opts.OlhoffVariant,{'lp','mma','both'})} = 'lp'
end
p=ie2a.paths();outputDir=ie2a.assert_output_isolated(outputDir,'production');c=ie2a.load_contract();
if ~isfolder(outputDir),mkdir(outputDir);end
op=ie2a.olhoff_variant_plan(opts.OlhoffVariant);
runs=struct('method',{'Proposed','Yuksel'},'label',{'Proposed','Yuksel'}, ...
    'variant',{'',''},'B0',{900,2000});
for q=1:height(op)
    runs(end+1)=struct('method','Olhoff','label',char(op.result_label(q)), ...
        'variant',char(op.route_id(q)),'B0',3200); %#ok<AGROW>
end
meshes=double(c.production_meshes);levels=double(c.quality.levels(:).');
for im=1:size(meshes,1)
    nx=meshes(im,1);ny=meshes(im,2);
    for jm=1:numel(runs)
        method=runs(jm).method;caseDir=fullfile(outputDir,sprintf('%s_%dx%d',lower(runs(jm).label),nx,ny));
        if ~isfolder(caseDir),mkdir(caseDir);end
        refTr=ie2a.run_method_trajectory(method,nx,ny,3200,fullfile(caseDir,'reference_raw'), ...
            OlhoffVariant=localVariant(runs(jm).variant));
        refA=ie2a.analyze_trajectory(refTr);ref=ie2a.reference_phase(refA.Q,refA.H0, ...
            SolverTerminated=refTr.solver_terminated,EvaluatorValid=refA.evaluator_valid);
        result=struct('method',runs(jm).label,'olhoff_variant',runs(jm).variant,'nelx',nx,'nely',ny, ...
            'reference',ref,'contract_hash',ie2a.frozen_contract_sha256());
        if ~strcmp(ref.status,'PASS'),result.status=ref.status;save(fullfile(caseDir,'result.mat'),'result','-v7.3');continue;end
        budget=ie2a.measurement_budget(runs(jm).B0,ref.b_ref,100,3200);
        measTr=ie2a.run_method_trajectory(method,nx,ny,budget.B_meas,fullfile(caseDir,'measurement_raw'), ...
            OlhoffVariant=localVariant(runs(jm).variant));
        shared=min(size(refTr.xPhys,2),size(measTr.xPhys,2));
        assert(strcmp(ie2a.trajectory_fingerprint(refTr.xPhys(:,1:shared)),ie2a.trajectory_fingerprint(measTr.xPhys(:,1:shared))), ...
            'ie2a:FingerprintMismatch','Reference and measurement trajectory prefixes differ.');
        measA=ie2a.analyze_trajectory(measTr,ref.Q_ref,levels);
        result.budget=budget;result.measurement=measA;result.measurement_fingerprint=measTr.fingerprint;result.status='ANALYZED';
        save(fullfile(caseDir,'result.mat'),'result','-v7.3');
    end
end
end
function v=localVariant(v)
if isempty(v),v='lp';end
end
