function plan=method_plan(olhoffVariant)
%METHOD_PLAN Expand selector without silently executing the other route.
op=ie2a.olhoff_variant_plan(olhoffVariant);
method=["Proposed";"Yuksel";repmat("Olhoff",height(op),1)];
method_variant=["proposed";"yuksel";op.route_id];
label=["Proposed";"Yuksel";op.result_label];
route_role=["principal";"principal";op.role];
B0=[900;2000;repmat(3200,height(op),1)];
plan=table(method,method_variant,label,route_role,B0);
end
