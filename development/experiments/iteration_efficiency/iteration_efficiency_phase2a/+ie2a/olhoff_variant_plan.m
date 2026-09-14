function plan = olhoff_variant_plan(variant)
%OLHOFF_VARIANT_PLAN Expand the settled LP/MMA reporting selector.
arguments
    variant (1,:) char {mustBeMember(variant,{'lp','mma','both'})} = 'lp'
end
switch variant
    case 'lp', ids={'lp'};
    case 'mma', ids={'mma'};
    otherwise, ids={'lp','mma'};
end
route_id=strings(numel(ids),1);role=route_id;result_label=route_id;runner=route_id;
qualification_required=false(numel(ids),1);
for i=1:numel(ids)
    if strcmp(ids{i},'lp')
        route_id(i)='lp';role(i)='principal';result_label(i)='Olhoff-LP';
        runner(i)='analysis/olhoff_stabilization_audit/olhoffOptStabilized.m';
    else
        route_id(i)='mma';role(i)='secondary_paper_literal';result_label(i)='Olhoff-MMA';
        runner(i)='Matlab/reproduction2007/runner/run_repro2007.m';qualification_required(i)=true;
    end
end
plan=table(route_id,role,result_label,runner,qualification_required);
end
