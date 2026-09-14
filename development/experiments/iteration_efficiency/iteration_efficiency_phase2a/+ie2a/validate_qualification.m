function report = validate_qualification(path, kind, contract, opts)
%VALIDATE_QUALIFICATION Strict, version- and provenance-bound Phase-2H gate.
arguments
    path (1,:) char
    kind (1,:) char
    contract struct
    opts.SelectedOlhoffVariant (1,:) char = 'lp'
end
allowed={'precision','cross_method','reference_length'};
if ~ismember(kind,allowed),error('ie2a:QualificationKind','Unknown qualification kind.');end
expectedSchema=['candidate_c_' kind '_qualification_v1'];
errors={}; q=struct(); p=ie2a.paths();
if ~isfile(path),errors{end+1}='artifact missing';
else
    try,q=jsondecode(fileread(path));catch ME,errors{end+1}=['invalid JSON: ' ME.message];end
end
if isempty(errors)
    required={'schema_version','pass','scope','candidate','classifier_version','evaluator_sha256','contract_sha256','input_provenance_sha256','results'};
    for i=1:numel(required),if ~isfield(q,required{i}),errors{end+1}=['missing ' required{i}];end,end %#ok<AGROW>
end
if isempty(errors)
    errors=must(errors,strcmp(q.schema_version,expectedSchema),'schema');
    errors=must(errors,islogical(q.pass)&&isscalar(q.pass)&&q.pass,'pass');
    errors=must(errors,strcmp(q.scope,kind),'scope');
    errors=must(errors,strcmp(q.candidate,'C'),'candidate');
    errors=must(errors,strcmp(q.classifier_version,contract.quality.classifier_version),'classifier');
    errors=must(errors,strcmp(q.evaluator_sha256,contract.quality.source_sha256),'evaluator hash');
    errors=must(errors,strcmp(q.contract_sha256,ie2a.sha256_file(p.contract)),'contract hash');
    errors=must(errors,isstruct(q.input_provenance_sha256)&&~isempty(fieldnames(q.input_provenance_sha256)),'input provenance');
    errors=must(errors,isstruct(q.results)&&~isempty(fieldnames(q.results)),'results');
    if isfield(q,'olhoff_variant')
        selected=string(opts.SelectedOlhoffVariant);qualified=string(q.olhoff_variant);
        errors=must(errors,selected==qualified || (selected=="lp"&&qualified=="both") || (selected=="mma"&&qualified=="both"),'Olhoff variant');
    else
        errors{end+1}='missing olhoff_variant';
    end
end
report=struct('pass',isempty(errors),'kind',kind,'path',path,'errors',{errors});
end
function errors=must(errors,condition,label)
if ~condition,errors{end+1}=label;end
end
