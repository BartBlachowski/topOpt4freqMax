function report=preflight(cfg)
%PREFLIGHT Fail closed on stale evaluator/contract/manifest and production lock.
p=iefinal.paths();checks=struct();details=struct();
checks.manifest_exists=isfile(p.manifest);checks.schema_exists=isfile(p.schema);
checks.meshes=isequal(double(cfg.manifest.production_meshes), ...
    [160 20;240 30;320 40;400 50;480 60;560 70;640 80;720 90;800 100]);
checks.candidate=strcmp(cfg.manifest.common_evaluator.id,'candidate_c_adaptive_structural_mode_v1');
evalPath=fullfile(p.repo,cfg.manifest.common_evaluator.source);
checks.evaluator_hash=isfile(evalPath)&&strcmp(ie2a.sha256_file(evalPath),cfg.manifest.common_evaluator.sha256);
checks.topology_hash=strcmp(ie2a.sha256_file(fullfile(p.repo,cfg.manifest.topology.implementation)),cfg.manifest.topology.sha256);
checks.method_source_hashes=true;
for i=1:numel(cfg.manifest.methods)
    s=localItem(cfg.manifest.methods,i);path=fullfile(p.repo,s.source);
    checks.method_source_hashes=checks.method_source_hashes&&isfile(path)&&strcmp(ie2a.sha256_file(path),s.sha256);
end
checks.pinned_component_hashes=true;
for i=1:numel(cfg.manifest.pinned_components)
    s=localItem(cfg.manifest.pinned_components,i);path=fullfile(p.repo,s.path);
    checks.pinned_component_hashes=checks.pinned_component_hashes&&isfile(path)&&strcmp(ie2a.sha256_file(path),s.sha256);
end
checks.contract_hash=strcmp(ie2a.sha256_file(p.contract),cfg.manifest.scientific_contract.sha256);
try,c=jsondecode(fileread(p.contract));ie2a.validate_contract(c,VerifyFiles=false);checks.contract_semantics=true;
catch ME,checks.contract_semantics=false;details.contract=ME.message;end
checks.double_policy=strcmp(cfg.manifest.trajectory.authoritative_dtype,'double');
checks.selector=ismember(cfg.olhoff_variant,{'lp','mma','both'});
checks.no_self_authorization=~cfg.manifest.production_authorized;
% No authorization token is embedded or accepted during integration.
checks.production_authorization=~strcmp(cfg.run_mode,'production');
names=fieldnames(checks);pass=all(cellfun(@(n)logical(checks.(n)),names));
report=struct('pass',pass,'checks',checks,'details',details);
if ~pass
    failed=names(~cellfun(@(n)logical(checks.(n)),names));
    error('iefinal:PreflightFailed','Final harness failed closed: %s',strjoin(failed,', '));
end
end
function x=localItem(a,i)
if iscell(a),x=a{i};else,x=a(i);end
end
