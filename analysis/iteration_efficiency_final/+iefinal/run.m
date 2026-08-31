function summary=run(runMode,selector)
%RUN Execute isolated smoke or authorized production architecture.
cfg=iefinal.config(runMode,selector);preflight=iefinal.preflight(cfg);
out=iefinal.new_run_directory(runMode,selector);p=iefinal.paths();
runManifest=struct('schema_version','iteration_efficiency_run_manifest_v1','run_mode',runMode, ...
    'olhoff_variant',selector,'output_directory',out,'started_at',char(datetime('now','TimeZone','local')), ...
    'production_campaign',strcmp(runMode,'production'),'preflight',preflight);
localWriteJson(fullfile(out,'provenance','run_manifest.json'),runManifest);
if strcmp(runMode,'smoke')
    summary=iefinal.run_smoke(cfg,out);
else
    summary=localProduction(cfg,out);
end
summary.output_directory=out;summary.production_campaign_run=strcmp(runMode,'production');
localWriteJson(fullfile(out,'analysis','run_summary.json'),summary);
fprintf('FINAL_HARNESS_%s_COMPLETE selector=%s output=%s\n',upper(runMode),selector,out);
end

function summary=localProduction(cfg,out)
allRows=struct([]);trajectories=struct([]);tc=0;failures=struct([]);fc=0;
for im=1:size(cfg.meshes,1)
    nx=cfg.meshes(im,1);ny=cfg.meshes(im,2);
    for ir=1:height(cfg.methods)
        method=char(cfg.methods.method(ir));variant=char(cfg.methods.method_variant(ir));label=char(cfg.methods.label(ir));
        id=sprintf('%s_%dx%d',lower(strrep(label,'-','_')),nx,ny);
        hashes=localSourceHashes(cfg,variant);
        % One production cell = method x mesh x route. A genuine method failure
        % is contained here so the campaign continues; an integrity failure is
        % rethrown and remains campaign-fatal.
        try
            [rows,rec]=localCell(cfg,out,id,method,variant,ir,nx,ny,hashes);
        catch ME
            verdict=iefinal.classify_cell_failure(ME);
            if ~verdict.cell_local
                fprintf(2,'CAMPAIGN-FATAL integrity failure in cell %s: %s\n',id,verdict.identifier);
                rethrow(ME)
            end
            rows=iefinal.build_error_rows(method,variant,label,nx,ny,cfg.methods.B0(ir),cfg,hashes,verdict);
            rec=struct([]);
            fc=fc+1;frec=struct('id',id,'method',method,'method_variant',variant, ...
                'mesh',sprintf('%dx%d',nx,ny),'identifier',verdict.identifier, ...
                'message',verdict.message,'class',verdict.class, ...
                'report',getReport(ME,'extended','hyperlinks','off'));
            if fc==1,failures=frec;else,failures(fc)=frec;end %#ok<AGROW>
            errDir=fullfile(out,'reference',id);if ~isfolder(errDir),mkdir(errDir);end
            save(fullfile(errDir,'run_error_result.mat'),'rows','frec','-v7.3');
            fprintf('CELL RUN_ERROR %s (%s) - campaign continues\n',id,verdict.identifier);
        end
        iefinal.validate_results(rows);
        if isempty(allRows),allRows=rows;else,allRows=[allRows;rows];end %#ok<AGROW>
        if ~isempty(rec),tc=tc+1;if tc==1,trajectories=rec;else,trajectories(tc)=rec;end,end %#ok<AGROW>
        % Checkpoint after every cell so a later hard failure cannot destroy
        % completed work.
        save(fullfile(out,'analysis','rows_checkpoint.mat'),'allRows','failures','-v7.3');
    end
end
if fc>0
    writetable(struct2table(rmfield(failures,'report')),fullfile(out,'analysis','RUN_ERRORS.csv'));
    localWriteJson(fullfile(out,'analysis','run_errors.json'),failures);
end
[allRows,timing]=iefinal.run_timing_firewall(allRows,fullfile(out,'timing'),cfg,trajectories);
iefinal.write_results(allRows,out);
iefinal.generate_scaling_outputs(allRows,fullfile(out,'analysis'),true);
iefinal.render_topologies(trajectories,allRows,fullfile(out,'topologies'));
summary=struct('status','PRODUCTION_EXECUTION_COMPLETE','row_count',numel(allRows), ...
    'methods',{cellstr(cfg.methods.label)},'meshes',cfg.meshes,'timing',timing, ...
    'failed_cells',fc,'failures',failures);
end

function [rows,rec]=localCell(cfg,out,id,method,variant,ir,nx,ny,hashes)
%LOCALCELL One production cell: reference, budget, measurement, rows.
rec=struct([]);
refDir=fullfile(out,'reference',id);refTr=iefinal.run_trajectory(method,variant,nx,ny,cfg.B_ref,refDir);
refCell=iefinal.analyze_cell(refTr,cfg.methods.B0(ir),cfg);
if ~strcmp(refCell.status,'REFERENCE_PASS')
    budget=struct('B0',cfg.methods.B0(ir));
    rows=iefinal.build_rows(refTr,struct(),refCell.reference,budget,cfg,hashes);
    save(fullfile(refDir,'failed_reference_result.mat'),'refCell','rows','-v7.3');return
end
B=refCell.budget.B_meas;measDir=fullfile(out,'measurement',id);
measTr=iefinal.run_trajectory(method,variant,nx,ny,B,measDir);
shared=min(size(refTr.x_post,2),size(measTr.x_post,2));
assert(strcmp(ie2a.trajectory_fingerprint(refTr.x_post(:,1:shared)), ...
    ie2a.trajectory_fingerprint(measTr.x_post(:,1:shared))),'iefinal:FingerprintMismatch', ...
    'Reference/measurement prefix mismatch for %s.',id);
mt=measTr;mt.xPhys=measTr.x_post;ma=ie2a.analyze_trajectory(mt,refCell.reference.Q_ref,cfg.quality_levels);
rows=iefinal.build_rows(measTr,ma,refCell.reference,refCell.budget,cfg,hashes);
rec=struct('id',id,'trajectory',measTr,'analysis',ma);
save(fullfile(measDir,'authoritative_result.mat'),'refCell','measTr','ma','rows','-v7.3');
end

function h=localSourceHashes(cfg,variant)
p=iefinal.paths();h=struct('evaluator',cfg.manifest.common_evaluator.sha256, ...
    'topology',cfg.manifest.topology.sha256,'trajectory_runner','');
methods=cfg.manifest.methods;
for i=1:numel(methods)
    if iscell(methods),m=methods{i};else,m=methods(i);end
    if strcmp(m.variant,variant)
        path=fullfile(p.repo,m.source);h.trajectory_runner=ie2a.sha256_file(path);return
    end
end
error('iefinal:ManifestSource','No source for variant %s.',variant);
end
function localWriteJson(path,value)
fid=fopen(path,'w');assert(fid>0,'iefinal:Write','Cannot write %s.',path);c=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));clear c
end
