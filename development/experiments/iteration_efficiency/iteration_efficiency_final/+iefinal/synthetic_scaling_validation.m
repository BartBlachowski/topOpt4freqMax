function report=synthetic_scaling_validation(analysisDir,figureDir)
%SYNTHETIC_SCALING_VALIDATION Exercise reporting only; never paper evidence.
Ne=[3200;7200;12800;20000];methods=["Proposed";"Yuksel";"Olhoff-LP"];
T=table();
for i=1:numel(methods)
    scale=[.8 1.1 1.5];expo=[.55 .62 .70];
    ke=scale(i)*Ne.^expo(i);kc=ke+99;tm=ke.*(1e-5*Ne.^.65);cost=tm./ke;
    q=.995+zeros(size(Ne));top=true(size(Ne));
    X=table(repmat(methods(i),4,1),Ne,ke,kc,tm,tm+99.*cost,cost,q,top, ...
        'VariableNames',{'method','element_count','k_enter','k_cert','time_to_enter','time_to_cert','mean_iteration_cost','Q','topology_pass'});
    T=[T;X]; %#ok<AGROW>
end
writetable(T,fullfile(analysisDir,'SMOKE_SYNTHETIC_SCALING_INPUT_NOT_SCIENTIFIC.csv'));
[fits,support]=iefinal.fit_scaling_table(T,{'k_enter','k_cert','time_to_enter','time_to_cert','mean_iteration_cost'});
writetable(fits,fullfile(analysisDir,'SMOKE_SYNTHETIC_SCALING_FITS_NOT_SCIENTIFIC.csv'));
writetable(support,fullfile(analysisDir,'SMOKE_SYNTHETIC_SCALING_SUPPORT_NOT_SCIENTIFIC.csv'));
f=figure('Visible','off','Color','white');tl=tiledlayout(f,2,2,'TileSpacing','compact');
metrics={'k_enter','k_cert','time_to_enter','mean_iteration_cost'};
for j=1:numel(metrics)
    ax=nexttile(tl);hold(ax,'on');for i=1:numel(methods),ix=T.method==methods(i);loglog(ax,T.element_count(ix),T.(metrics{j})(ix),'-o','DisplayName',methods(i));end
    grid(ax,'on');xlabel(ax,'element count');ylabel(ax,strrep(metrics{j},'_',' '));
    % Disclose the support the companion fit actually used, so the plotted
    % series can never imply a wider common fit than was performed.
    ns=support.n_support(support.metric==string(metrics{j}));
    if isempty(ns),ns=0;end
    title(ax,sprintf('SMOKE / SYNTHETIC / NOT SCIENTIFIC (common support n=%d)',ns(1)));
end
legend(nexttile(tl,1),'Location','best');exportgraphics(f,fullfile(figureDir,'SMOKE_SYNTHETIC_SCALING_NOT_SCIENTIFIC.png'),'Resolution',160);close(f);
isCommon=fits.support=="common";
report=struct('pass',all(isfinite(fits.C(isCommon&fits.fitted))&isfinite(fits.p(isCommon&fits.fitted))), ...
    'scientific_results',false,'fit_count',height(fits),'C_reported',true,'p_reported',true, ...
    'common_support_enforced',true,'common_support_n',max(support.n_support), ...
    'leave_one_out',true);
end
