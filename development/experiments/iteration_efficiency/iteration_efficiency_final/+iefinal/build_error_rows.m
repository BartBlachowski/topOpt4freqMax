function rows=build_error_rows(method,variant,label,nelx,nely,B0,cfg,sourceHashes,verdict)
%BUILD_ERROR_ROWS One RUN_ERROR row per q/P for a cell whose method failed.
% Preserves method, mesh, route, configuration and provenance identity, records
% the exception identity and message, and leaves EVERY scientific quantity N/A.
% Nothing here is inferred, carried over from another mesh, or defaulted to a
% zero that could be mistaken for a measurement.
arguments
    method (1,:) char
    variant (1,:) char
    label (1,:) char %#ok<INUSA>
    nelx (1,1) double
    nely (1,1) double
    B0 (1,1) double
    cfg struct
    sourceHashes struct
    verdict struct
end
assert(strcmp(verdict.status,'RUN_ERROR'),'iefinal:ErrorRowStatus', ...
    'build_error_rows is only for cell-local RUN_ERROR failures.');
Pvals=[cfg.P_primary cfg.P_sensitivity];levels=cfg.quality_levels;
rows=repmat(iefinal.empty_result_row(),numel(Pvals)*numel(levels),1);r=0;
for ip=1:numel(Pvals)
    for iq=1:numel(levels)
        r=r+1;z=iefinal.empty_result_row();
        z.schema_version='iteration_efficiency_result_v1';
        z.method=method;z.method_variant=variant;
        z.mesh=sprintf('%dx%d',nelx,nely);z.nelx=nelx;z.nely=nely;z.element_count=nelx*nely;
        z.q=levels(iq);z.P=Pvals(ip);z.B0=B0;z.B_ref=cfg.B_ref;
        z.status='RUN_ERROR';z.censoring_reason='RUN_ERROR';
        z.error_identifier=verdict.identifier;z.error_message=verdict.message;
        % trajectory_dtype is the campaign's declared authoritative storage
        % policy (schema const), not a measurement of this failed cell.
        z.trajectory_dtype=cfg.manifest.trajectory.authoritative_dtype;
        z.evaluator_id='candidate_c_adaptive_structural_mode_v1';
        z.contract_hash=cfg.manifest.scientific_contract.sha256;
        z.source_hashes=sourceHashes;z.provenance_hash='';
        rows(r)=z;
    end
end
end
