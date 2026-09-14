function fit = fit_power_law(Ne, y, labels)
%FIT_POWER_LAW Frozen unit-weight log-OLS scaling fit with LOO diagnostics.
arguments
    Ne (:,1) double
    y (:,1) double
    labels (:,1) string = strings(0,1)
end
if isempty(labels),labels=string(Ne);end
valid=isfinite(Ne)&isfinite(y)&Ne>0&y>0; x=log(Ne(valid)); z=log(y(valid)); lab=labels(valid);
fit=struct('C',NaN,'p',NaN,'R2_log',NaN,'n_valid',numel(x),'Ne_min',NaN,'Ne_max',NaN, ...
    'included_meshes',strjoin(lab,','),'p_LOO_min',NaN,'p_LOO_max',NaN, ...
    'weakly_identified',true,'exclusions',strjoin(labels(~valid),','));
if numel(x)<3, return; end
b=[ones(numel(x),1),x]\z; pred=[ones(numel(x),1),x]*b;
ssr=sum((z-pred).^2); sst=sum((z-mean(z)).^2);
fit.C=exp(b(1)); fit.p=b(2); fit.R2_log=1-ssr/sst; fit.Ne_min=min(Ne(valid)); fit.Ne_max=max(Ne(valid));
loo=nan(numel(x),1);
for i=1:numel(x), keep=true(numel(x),1);keep(i)=false; bb=[ones(sum(keep),1),x(keep)]\z(keep);loo(i)=bb(2);end
fit.p_LOO_min=min(loo);fit.p_LOO_max=max(loo);
fit.weakly_identified=fit.R2_log<.8 || (fit.p_LOO_min<=0&&fit.p_LOO_max>=0) || (fit.p_LOO_max-fit.p_LOO_min)>abs(fit.p);
end
