function status = classify_status(facts)
%CLASSIFY_STATUS Apply frozen status/censoring precedence.
required={'reference_status','endpoint_found','structural_mode_not_found','solver_terminated','solver_termination_after_cert', ...
    'topology_persistence_possible','quality_persistence_possible','pointwise_acceptance_seen'};
for i=1:numel(required), assert(isfield(facts,required{i}),'ie2a:StatusFacts','Missing fact %s.',required{i}); end
if facts.endpoint_found
    if facts.solver_terminated && facts.solver_termination_after_cert
        status='PASS_WITH_LATER_SOLVER_TERMINATION';
    else
        status='PASS';
    end
elseif facts.structural_mode_not_found || strcmp(facts.reference_status,'STRUCTURAL_MODE_NOT_FOUND')
    status='STRUCTURAL_MODE_NOT_FOUND';
elseif strcmp(facts.reference_status,'REFERENCE_SOLVER_TERMINATION')
    status='REFERENCE_SOLVER_TERMINATION';
elseif strcmp(facts.reference_status,'REFERENCE_NOT_ESTABLISHED')
    status='REFERENCE_NOT_ESTABLISHED';
elseif facts.solver_terminated
    status='SOLVER_TERMINATION';
elseif ~facts.topology_persistence_possible
    status='INVALID_TOPOLOGY';
elseif ~facts.quality_persistence_possible
    status='QUALITY_NOT_REACHED';
elseif facts.pointwise_acceptance_seen
    status='PERSISTENT_NONACCEPTANCE';
else
    status='OTHER';
end
end
