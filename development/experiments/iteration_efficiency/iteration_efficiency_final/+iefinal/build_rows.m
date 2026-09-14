function rows=build_rows(tr,analysis,ref,budget,cfg,sourceHashes)
%BUILD_ROWS One method-neutral record per q/P, preserving N/A as NaN/null.
Pvals=[cfg.P_primary cfg.P_sensitivity];levels=cfg.quality_levels;n=size(tr.x_post,2);
if ~strcmp(ref.status,'PASS')
    rows=repmat(localEmpty(),numel(Pvals)*numel(levels),1);r=0;
    for ip=1:numel(Pvals),for iq=1:numel(levels)
        r=r+1;z=localEmpty();z.schema_version='iteration_efficiency_result_v1';
        z.method=tr.method;z.method_variant=tr.method_variant;z.mesh=sprintf('%dx%d',tr.nelx,tr.nely);
        z.nelx=tr.nelx;z.nely=tr.nely;z.element_count=tr.nelx*tr.nely;z.q=levels(iq);z.P=Pvals(ip);
        z.B0=budget.B0;z.B_ref=cfg.B_ref;z.status=ref.status;z.censoring_reason=ref.status;
        z.trajectory_dtype=tr.trajectory_dtype;z.evaluator_id='candidate_c_adaptive_structural_mode_v1';
        z.contract_hash=cfg.manifest.scientific_contract.sha256;z.source_hashes=sourceHashes;z.provenance_hash='';rows(r)=z;
    end,end
    return
end
ratio=analysis.Q./ref.Q_ref;robust=min(ratio,[],2);rows=repmat(localEmpty(),numel(Pvals)*numel(levels),1);r=0;
for ip=1:numel(Pvals)
    P=Pvals(ip);pass=false(n,numel(levels));
    for iq=1:numel(levels),pass(:,iq)=analysis.H0&robust>=levels(iq);end
    persistence=ie2a.scan_persistence(pass,P);
    for iq=1:numel(levels)
        r=r+1;k=persistence.k_enter(iq);kc=persistence.k_cert(iq);
        facts=struct('reference_status',ref.status,'endpoint_found',isfinite(kc), ...
            'structural_mode_not_found',any(~analysis.evaluator_valid), ...
            'solver_terminated',tr.solver_terminated,'solver_termination_after_cert',false, ...
            'topology_persistence_possible',any(analysis.H_topology), ...
            'quality_persistence_possible',any(robust>=levels(iq)), ...
            'pointwise_acceptance_seen',any(pass(:,iq)));
        status=ie2a.classify_status(facts);z=localEmpty();
        z.schema_version='iteration_efficiency_result_v1';z.method=tr.method;z.method_variant=tr.method_variant;
        z.mesh=sprintf('%dx%d',tr.nelx,tr.nely);z.nelx=tr.nelx;z.nely=tr.nely;z.element_count=tr.nelx*tr.nely;
        z.q=levels(iq);z.P=P;z.B0=budget.B0;z.B_ref=cfg.B_ref;z.b_ref=ref.b_ref;z.B_meas=budget.B_meas;
        z.tail_truncated=budget.certification_tail_truncated;z.k_enter=k;z.k_cert=kc;z.status=status;
        if startsWith(status,'PASS'),z.censoring_reason='';else,z.censoring_reason=status;end
        if isfinite(k)
            z.E1=analysis.Q(k,1);z.E2=analysis.Q(k,2);z.E3=analysis.Q(k,3);z.Q=robust(k);
            z.topology_pass=analysis.H_topology(k);z.volume_pass=analysis.H_volume(k);
            z.hard_gate_pass=analysis.H_topology(k)&&analysis.H_volume(k);
            z=localAccounting(z,tr,k);
        end
        z.trajectory_dtype=tr.trajectory_dtype;z.evaluator_id='candidate_c_adaptive_structural_mode_v1';
        z.contract_hash=cfg.manifest.scientific_contract.sha256;z.source_hashes=sourceHashes;z.provenance_hash='';
        rows(r)=z;
    end
end
end

function z=localAccounting(z,tr,k)
switch tr.method
    case 'Proposed'
        z.native_iterations=k;
    case 'Yuksel'
        s1=tr.native.stage1_updates;z.native_iterations=s1+k;z.yuksel_stage1_iterations=s1;
        z.yuksel_stage2_iterations=k;z.yuksel_total_iterations=s1+k;
    case 'Olhoff'
        z.native_iterations=k;z.olhoff_outer_updates=k;
        if strcmp(tr.method_variant,'lp')
            z.olhoff_lp_calls=min(k,tr.native.lp_calls);z.olhoff_failed_lp_calls=tr.native.failed_lp_calls;
            if tr.native.lp_backend_iterations_observed
                v=tr.telemetry.lpBackendIterations(1:k);z.olhoff_lp_backend_iterations=sum(v(isfinite(v)));
            end
        else
            inner=tr.native.inner_iterations(1:k);conv=tr.native.inner_converged(1:k);cap=tr.native.inner_cap_hit(1:k);
            z.olhoff_mma_total_inner_iterations=sum(inner);z.olhoff_mma_mean_inner_iterations=mean(inner);
            z.olhoff_mma_median_inner_iterations=median(inner);z.olhoff_mma_p95_inner_iterations=localP95(inner);
            z.olhoff_mma_max_inner_iterations=max(inner);z.olhoff_mma_cap_hit_count=sum(cap);
            z.olhoff_mma_cap_hit_fraction=mean(cap);z.olhoff_mma_converged_inner_count=sum(conv);
            z.olhoff_mma_converged_inner_fraction=mean(conv);
        end
end
end
function x=localP95(v),v=sort(v(:));x=v(max(1,ceil(.95*numel(v))));end
function z=localEmpty(),z=iefinal.empty_result_row();end
