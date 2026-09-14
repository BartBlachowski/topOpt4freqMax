function out = generate_scaling(rows)
%GENERATE_SCALING Per-method fits plus mandatory common-support companion fits.
required={'method','mesh','Ne','k_enter','status'};
assert(all(ismember(required,rows.Properties.VariableNames)),'ie2a:ScalingSchema','Scaling rows lack required fields.');
methods=unique(string(rows.method),'stable'); fits=struct([]); certified=ismember(string(rows.status),["PASS","PASS_WITH_LATER_SOLVER_TERMINATION"]);
support=string(rows.mesh(certified));
for i=1:numel(methods)
    idx=string(rows.method)==methods(i)&certified;
    f=ie2a.fit_power_law(rows.Ne(idx),rows.k_enter(idx),string(rows.mesh(idx))); f.method=methods(i);f.support='available';fits=[fits;f]; %#ok<AGROW>
end
common=unique(support);
for i=1:numel(methods),common=intersect(common,string(rows.mesh(string(rows.method)==methods(i)&certified)),'stable');end
commonFits=struct([]);
for i=1:numel(methods)
    idx=string(rows.method)==methods(i)&certified&ismember(string(rows.mesh),common);
    f=ie2a.fit_power_law(rows.Ne(idx),rows.k_enter(idx),string(rows.mesh(idx)));f.method=methods(i);f.support='common';commonFits=[commonFits;f]; %#ok<AGROW>
end
out=struct('available_support_fits',fits,'common_support_fits',commonFits,'common_meshes',common, ...
    'wording','empirical scaling over the tested mesh range');
end
