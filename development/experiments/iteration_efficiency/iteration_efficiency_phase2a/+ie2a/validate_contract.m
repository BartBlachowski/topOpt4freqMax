function report = validate_contract(c, opts)
%VALIDATE_CONTRACT Fail closed on any frozen scientific-contract deviation.
arguments
    c struct
    opts.VerifyFiles (1,1) logical = true
end
errors={};
errors=must(errors,strcmp(c.schema_version,'2H.1') && strcmp(c.contract_id,'iteration_efficiency_candidate_c_refreeze_2026-08-31'),'Phase-2H contract identity');
errors=must(errors,isequal(double(c.production_meshes),[160 20;240 30;320 40;400 50;480 60;560 70;640 80;720 90;800 100]),'nine mesh sequence');
errors=must(errors,~c.production_authorized,'contract must not self-authorize production');
errors=must(errors,c.reference.B_ref==3200 && c.reference.P==100 && c.reference.L_ref==500 && c.reference.epsilon_ref==0.001,'reference constants');
errors=must(errors,isequal(double(c.quality.levels(:).'),[.98 .99 .995]),'quality levels');
errors=must(errors,isequal(string({c.quality.evaluators.id}),["E1" "E2" "E3"]),'co-primary evaluator set');
errors=must(errors,c.quality.co_primary && contains(c.quality.robust_definition,'min'),'robust evaluator rule');
errors=must(errors,strcmp(c.quality.candidate,'C') && strcmp(c.quality.design_field,'actual_gray') && strcmp(c.quality.classifier_version,'candidate_c_unanimous_v1'),'Candidate C identity');
ms=c.quality.mode_selection;
errors=must(errors,ms.voidKE_max_strict==.5 && ms.voidSE_max_strict==.5 && ms.densityParticipation_min_strict==.5 && strcmp(ms.logic,'ALL_THREE_UNANIMOUS'),'unanimous classifier');
errors=must(errors,ms.initial_modes==3 && ms.escalation_factor==2 && ~ms.scientific_mode_ceiling && strcmp(ms.failure_status,'STRUCTURAL_MODE_NOT_FOUND'),'adaptive mode search');
errors=must(errors,contains(c.quality.E2_E3_shared_mass_law,'1e5_x6') && contains(c.quality.binary_projection_role,'excluded_from_Q'),'mass law and binary exclusion');
errors=must(errors,strcmp(c.measurement_budget.formula,'B_meas = min(max(B0, b_ref + P - 1), B_ref)'),'B_meas formula');
errors=must(errors,c.measurement_budget.progress_triggered_extension==false,'no progress-triggered extension');
errors=must(errors,isequal(double(c.persistence.OAT_P(:).'),[50 200]) && c.persistence.P==100,'persistence constants');
errors=must(errors,strcmp(c.trajectory_storage.new_trajectories,'lossless_double_return_equivalent_fields') && ~c.trajectory_storage.observer_inside_common_evaluator,'trajectory storage semantics');
errors=must(errors,contains(c.trajectory_storage.single_precision_permission,'identical_gate_decisions_k_enter_k_cert'),'single-precision qualification rule');
errors=must(errors,c.topology.A_sig==0.01 && contains(c.topology.hard_gate,'each_detached_component_area_lt_A_sig'),'topology componentwise hard gate');
errors=must(errors,strcmp(c.topology.aggregate_detached_area_role,'diagnostic_only') && strcmp(c.topology.n_islands_all_role,'diagnostic_only'),'topology diagnostic-only fields');
errors=must(errors,isequal(double(c.topology.a_sig_by_mesh(:).'),[4 9 16 25 36 49 64 81 100]),'topology thresholds by mesh');
errors=must(errors,c.topology.F8_exhaustive_640x80.aggregate_detached_median_elements==64 && c.topology.F8_exhaustive_640x80.aggregate_detached_p95_elements==147 && c.topology.F8_exhaustive_640x80.aggregate_detached_max_elements==674,'F8 corrected anchors');
methods=localCells(c.methods); expectedMethods=["Proposed" "Yuksel" "Olhoff"];
errors=must(errors,isequal(string(cellfun(@(s)s.id,methods,'UniformOutput',false)),expectedMethods),'method identities');
errors=must(errors,isequal(cellfun(@(s)s.B0,methods),[900 2000 3200]),'method B0 values');
errors=must(errors,strcmp(methods{1}.profile_id,'proposed_practical_move02_tol001'),'Proposed profile');
errors=must(errors,strcmp(methods{2}.profile_id,'yuksel_practical_move01_tol001'),'Yuksel profile');
errors=must(errors,strcmp(methods{3}.profile_id,'olhoff_fig3a_s1_native_bimodal_p100_move005_0025_fixed1600_v1'),'Olhoff profile');
errors=must(errors,isequal(string(methods{3}.route_policy.selector_values(:).'),["lp" "mma" "both"]) && strcmp(methods{3}.route_policy.principal,'lp') && strcmp(methods{3}.route_policy.secondary,'mma') && methods{3}.route_policy.separate_rows,'Olhoff route policy');
expectedStatus=["PASS" "PASS_WITH_LATER_SOLVER_TERMINATION" "STRUCTURAL_MODE_NOT_FOUND" "REFERENCE_SOLVER_TERMINATION" "REFERENCE_NOT_ESTABLISHED" "SOLVER_TERMINATION" "INVALID_TOPOLOGY" "QUALITY_NOT_REACHED" "PERSISTENT_NONACCEPTANCE" "OTHER"];
errors=must(errors,isequal(string(c.statuses.precedence(:).'),expectedStatus),'status precedence');
errors=must(errors,c.timing.threads==1 && c.timing.serial && c.timing.repetitions==3 && c.timing.discarded_warmup_per_method==1,'timing policy');
errors=must(errors,strcmp(c.timing.endpoint_strategy,'clean_fixed_horizon_replays_after_offline_endpoint_freeze') && c.timing.T_reference_separate,'timing boundaries');
errors=must(errors,strcmp(c.scaling.model,'y=C*Ne^p') && c.scaling.minimum_valid_meshes==3 && c.scaling.common_support_companion_required,'scaling policy');
errors=must(errors,strcmp(c.topology_rendering.shared_renderer,'tools/Matlab/renderTopologyDensity.m'),'shared topology renderer');
errors=must(errors,strcmp(c.phase2h_refreeze.production_status,'BLOCKED') && c.phase2h_refreeze.precision_qualification.required && c.phase2h_refreeze.cross_method_qualification.required && c.phase2h_refreeze.reference_length_qualification.required,'three qualification locks');
if opts.VerifyFiles
    p=ie2a.paths();
    for i=1:numel(c.normative_documents)
        d=c.normative_documents(i); path=fullfile(p.repo,d.path);
        errors=must(errors,isfile(path),['normative file ' d.path]);
        if isfile(path), errors=must(errors,strcmp(ie2a.sha256_file(path),d.sha256),['normative hash ' d.path]); end
    end
    for i=1:numel(methods)
        path=fullfile(p.repo,methods{i}.source);
        errors=must(errors,isfile(path) && strcmp(ie2a.sha256_file(path),methods{i}.source_sha256),['source hash ' methods{i}.id]);
    end
end
report=struct('pass',isempty(errors),'errors',{errors},'contract_id',c.contract_id);
if ~report.pass
    error('ie2a:ContractViolation','Frozen contract validation failed: %s',strjoin(errors,'; '));
end
end
function c=localCells(x)
if iscell(x),c=x(:).';else,c=arrayfun(@(s){s},x);end
end
function errors=must(errors,condition,label)
if ~condition, errors{end+1}=label; end %#ok<AGROW>
end
