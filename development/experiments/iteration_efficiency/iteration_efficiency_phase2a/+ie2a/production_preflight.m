function report = production_preflight(opts)
%PRODUCTION_PREFLIGHT Dedicated fail-closed implementation and authorization gate.
arguments
    opts.RequireAuthorization (1,1) logical = false
    opts.AuthorizationToken (1,:) char = ''
    opts.SelectedOlhoffVariant (1,:) char {mustBeMember(opts.SelectedOlhoffVariant,{'lp','mma','both'})} = 'lp'
    opts.ThrowOnFailure (1,1) logical = true
end
p=ie2a.paths(); c=ie2a.load_contract(); checks=struct(); details=struct();
checks.contract_hash=strcmp(ie2a.sha256_file(p.contract),ie2a.frozen_contract_sha256());
try, ie2a.validate_contract(c);checks.contract_semantics=true;catch ME,checks.contract_semantics=false;details.contract=ME.message;end
checks.mesh_sequence=isequal(double(c.production_meshes),[160 20;240 30;320 40;400 50;480 60;560 70;640 80;720 90;800 100]);
if iscell(c.methods),profiles=string(cellfun(@(s)s.profile_id,c.methods,'UniformOutput',false));else,profiles=string({c.methods.profile_id});end
checks.profile_bindings=isequal(profiles(:).',["proposed_practical_move02_tol001","yuksel_practical_move01_tol001","olhoff_fig3a_s1_native_bimodal_p100_move005_0025_fixed1600_v1"]);
checks.evaluator_hash=strcmp(ie2a.sha256_file(fullfile(p.repo,c.quality.source)),c.quality.source_sha256);
checks.candidate_c_identity=strcmp(c.quality.candidate,'C')&&strcmp(c.quality.classifier_version,'candidate_c_unanimous_v1')&&strcmp(c.quality.design_field,'actual_gray');
freezePath=fullfile(p.repo,c.phase2h_refreeze.freeze_record);
checks.phase2h_freeze_record=isfile(freezePath)&&strcmp(ie2a.sha256_file(freezePath),ie2a.frozen_freeze_record_sha256());
checks.topology_rule=contains(c.topology.hard_gate,'each_detached_component_area_lt_A_sig')&&strcmp(c.topology.aggregate_detached_area_role,'diagnostic_only');
checks.budget_engine=ie2a.measurement_budget(900,600,100,3200).B_meas==900 && ie2a.measurement_budget(900,3200,100,3200).B_meas==3200;
checks.output_isolation=localIsolation(p);
checks.timing=c.timing.threads==1&&c.timing.serial&&c.timing.repetitions==3;
checks.instrumentation=contains(fileread(fullfile(p.repo,'tools','Matlab','topopt_history_record.m')),'topopt_iteration_observer_v1');
qnames={'precision','cross_method','reference_length'};
qkeys={'precision_qualification','cross_method_qualification','reference_length_qualification'};
for i=1:numel(qnames)
    spec=c.phase2h_refreeze.(qkeys{i});path=fullfile(p.repo,spec.path);
    qr=ie2a.validate_qualification(path,qnames{i},c,SelectedOlhoffVariant=opts.SelectedOlhoffVariant);
    checks.(['candidate_c_' qnames{i}])=qr.pass;details.(['candidate_c_' qnames{i}])=qr;
end
checks.olhoff_variant_policy=ismember(opts.SelectedOlhoffVariant,{'lp','mma','both'});
if ismember(opts.SelectedOlhoffVariant,{'mma','both'})
    details.olhoff_mma='Secondary numerical MMA route remains qualification-gated and separately accounted.';
end
checks.table_pipeline=isfile(fullfile(p.phase2a,'+ie2a','generate_tables.m'))&&isfile(fullfile(p.phase2a,'+ie2a','generate_scaling.m'));
checks.figure_pipeline=isfile(fullfile(p.repo,c.topology_rendering.shared_renderer))&&isfile(fullfile(p.repo,'analysis','iteration_efficiency_study_design','render_iteration_efficiency_topology_grid.m'));
checks.production_driver=isfile(fullfile(p.phase2a,'+ie2a','run_production_campaign.m'))&&isfile(fullfile(p.phase2a,'+ie2a','run_method_trajectory.m'));
try, details.negative_controls=ie2a.run_negative_controls();checks.negative_controls=true;catch ME,checks.negative_controls=false;details.negative=ME.message;end
checks.authorization=~opts.RequireAuthorization || strcmp(opts.AuthorizationToken,'AUTHORIZE_FROZEN_NINE_MESH_PRODUCTION_AFTER_REVIEW');
names=fieldnames(checks); pass=true;for i=1:numel(names),pass=pass&&checks.(names{i});end
report=struct('pass',pass,'checks',checks,'details',details,'production_authorized',checks.authorization&&opts.RequireAuthorization);
if ~pass && opts.ThrowOnFailure
    failed=names(~cellfun(@(n)checks.(n),names));
    error('ie2a:PreflightFailed','Production preflight failed closed: %s',strjoin(failed,', '));
end
end
function ok=localIsolation(p)
try
    ie2a.assert_output_isolated(p.production,'production');
    try,ie2a.assert_output_isolated(fullfile(p.repo,'examples','Performance','final_campaign'),'production');ok=false;catch,ok=true;end
catch,ok=false;end
end
