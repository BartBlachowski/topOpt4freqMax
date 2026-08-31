function T = quality_effort(rows)
%QUALITY_EFFORT Build absolute-quality evidence; absent Q is never inferred.
required={'method','mesh','status','Q_E1','Q_E2','Q_E3'};
for i=1:numel(required), assert(ismember(required{i},rows.Properties.VariableNames), ...
        'ie2a:AbsoluteQualityMissing','Required absolute-quality column %s is absent.',required{i}); end
T=rows(:,required);
T.best_observed=false(height(T),1);
meshes=unique(string(T.mesh));
for i=1:numel(meshes)
    ix=string(T.mesh)==meshes(i);
    vals=[T.Q_E1(ix),T.Q_E2(ix),T.Q_E3(ix)];
    best=max(vals,[],1,'omitnan');
    T.best_observed(ix)=any(vals==best,2);
end
end
