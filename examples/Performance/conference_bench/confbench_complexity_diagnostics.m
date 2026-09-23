function files = confbench_complexity_diagnostics(cfg, records, scaling)
%CONFBENCH_COMPLEXITY_DIAGNOSTICS Native counts, unit costs and fit validation.
% Uses recorded results only; never calls a solver or changes timing records.
% Fits require the campaign scaling gate and finite positive, ok observations.
% Both residual curves use LOG-space least squares (free p and fixed p=1.5).
% The transition is the pre-existing 2^17 full-DOF threshold, not a fitted split.
% Unit-cost decompositions are descriptive, not predictions of iteration count.
od = fullfile(cfg.outputDir, 'complexity_diagnostics');
if exist(od, 'dir') ~= 7; mkdir(od); end
allowed = isfield(scaling,'fitted') && scaling.fitted;
keys = {'proposed','yuksel','olhoff'};
names = cellfun(@confbench_paper_label, keys, 'UniformOutput', false);
colors = [0 .447 .741; .85 .325 .098];
units = {{'Reference solve','SIMP iteration'}, ...
    {'Stage 1 iteration','Stage 2 iteration'}, ...
    {'Outer-exclusive / outer iteration','Nested MMA iteration'}};
models = {'Free p (log fit)','Fixed p=1.5 (log fit)'};
raw = struct([]); fits = struct([]); validation = struct([]); residuals = struct([]);
countsFig = newFigure('Native counts'); countsLayout = tiledlayout(countsFig,1,3,'TileSpacing','compact');
costFig = newFigure('Cost per native unit'); costLayout = tiledlayout(costFig,1,3,'TileSpacing','compact');
resFig = newFigure('Relative residuals'); resLayout = tiledlayout(resFig,1,3,'TileSpacing','compact');
ratioFig = newFigure('Transition ratios'); ratioLayout = tiledlayout(ratioFig,1,3,'TileSpacing','compact');
cvFig = newFigure('Validation'); cvLayout = tiledlayout(cvFig,1,3,'TileSpacing','compact');
for m = 1:3
    R = records(strcmp({records.method_key},keys{m}));
    if isempty(R); continue; end
    [x,order] = sort(arrayfun(@(r) prod(r.mesh), R)); x=x(:); R=R(order);
    dofs = arrayfun(@(r) 2*prod(r.mesh+1),R); dofs=dofs(:);
    high = dofs >= 2^17;
    count = nan(numel(R),2); time = count;
    ok = logical([R.ok]); ok=ok(:);
    for i=1:numel(R)
        for j=1:2
            count(i,j)=value(R(i).counts,sprintf('count%d',j));
            time(i,j)=value(R(i).times,sprintf('time%d',j));
        end
    end
    cost=time./count; stage=sum(time,2);
    eligible=allowed & ok & isfinite(stage) & stage>0;
    axCount=nexttile(countsLayout,m); hold(axCount,'on');
    axCost=nexttile(costLayout,m); hold(axCost,'on');
    axRatio=nexttile(ratioLayout,m); hold(axRatio,'on');
    for j=1:2
        valid=isfinite(cost(:,j)) & cost(:,j)>0 & count(:,j)>0;
        train=eligible & valid & ~high;
        [C,p]=powerFit(x(train),cost(train,j),NaN);
        predicted=C*x.^p;
        plot(axCount,x,count(:,j),'-o','Color',colors(j,:),'DisplayName',units{m}{j});
        plot(axCost,x(valid),cost(valid,j),'o','Color',colors(j,:), ...
            'MarkerFaceColor',colors(j,:),'DisplayName',units{m}{j});
        if isfinite(p)
            plot(axCost,x(~high),predicted(~high),'-','Color',colors(j,:), ...
                'DisplayName',sprintf('Below threshold fit: p=%.3f',p));
            if any(high)
                first=find(high,1); ix=max(1,first-1):numel(x);
                plot(axCost,x(ix),predicted(ix),'--','Color',colors(j,:), ...
                    'HandleVisibility','off');
            end
            plot(axRatio,x(valid),cost(valid,j)./predicted(valid),'-o', ...
                'Color',colors(j,:),'DisplayName',units{m}{j});
        end
        fits=append(fits,struct('method',R(1).method,'unit',units{m}{j}, ...
            'fit_space','log cost','training_rule','ok; full DOFs < 131072', ...
            'n',sum(train),'C',C,'p',p));
        for i=1:numel(R)
            raw=append(raw,struct('method',R(i).method,'nelx',R(i).mesh(1), ...
                'nely',R(i).mesh(2),'elements',x(i),'full_dofs',dofs(i), ...
                'status',R(i).status,'ok',ok(i),'unit',units{m}{j}, ...
                'count',count(i,j),'stage_component_s',time(i,j), ...
                'cost_per_unit_s',cost(i,j),'above_threshold',high(i), ...
                'used_in_prethreshold_fit',train(i),'predicted_unit_cost_s',predicted(i), ...
                'observed_over_predicted',cost(i,j)/predicted(i)));
        end
    end
    decorate(axCount,names{m},'Native count (counts are not added)',true);
    decorate(axCost,names{m},'Seconds per named computational unit',true);
    decorate(axRatio,names{m},'Observed / below-threshold prediction',false);
    yline(axRatio,1,':','HandleVisibility','off');
    transition(axCount,x,high); transition(axCost,x,high); transition(axRatio,x,high);
    axRes=nexttile(resLayout,m); hold(axRes,'on');
    errors=nan(2,2);
    for k=1:2
        fixed=NaN; if k==2; fixed=1.5; end
        [C,p]=powerFit(x(eligible),stage(eligible),fixed);
        pred=C*x.^p;
        plot(axRes,x,100*(pred./stage-1),'-o','Color',colors(k,:), ...
            'DisplayName',sprintf('%s; p=%.3f',models{k},p));
        loo=nan(size(stage));
        if sum(eligible)>=4
            for i=find(eligible).'
                train=eligible; train(i)=false;
                [a,b]=powerFit(x(train),stage(train),fixed);
                loo(i)=a*x(i)^b;
            end
        end
        [a,b]=powerFit(x(eligible & ~high),stage(eligible & ~high),fixed);
        forward=a*x.^b;
        predictions={pred,loo,forward}; masks={eligible,eligible,eligible & high};
        labels={'in_sample','leave_one_mesh_out','train_below_predict_above_threshold'};
        for q=1:3
            metrics=errorsOf(predictions{q}(masks{q}),stage(masks{q}));
            validation=append(validation,struct('method',R(1).method,'model',models{k}, ...
                'evaluation',labels{q},'n',metrics.n,'MAPE_pct',metrics.mape, ...
                'RMSE_s',metrics.rmse,'log_RMSE',metrics.logrmse));
            if q>=2; errors(k,q-1)=metrics.mape; end
        end
        for i=1:numel(R)
            residuals=append(residuals,struct('method',R(i).method,'elements',x(i), ...
                'model',models{k},'fit_space','log stage time','C',C,'p',p, ...
                'stage_time_s',stage(i),'eligible',eligible(i),'fitted_s',pred(i), ...
                'relative_error_pct',100*(pred(i)/stage(i)-1), ...
                'leave_one_out_prediction_s',loo(i),'forward_prediction_s',forward(i)));
        end
    end
    decorate(axRes,names{m},'100 x (prediction / observation - 1)',false);
    yline(axRes,0,':','HandleVisibility','off'); transition(axRes,x,high);
    axCV=nexttile(cvLayout,m); bar(axCV,errors.');
    set(axCV,'XTickLabel',{'Leave one out','Below -> above'});
    ylabel(axCV,'Mean absolute percentage error (%)'); title(axCV,names{m},'Interpreter','none');
    grid(axCV,'on'); legend(axCV,models,'Location','northoutside','FontSize',8);
end
title(countsLayout,{'Native computational counts versus mesh size', ...
    'Proposed reference count is a solve count. Du-Olhoff outer and inner counts are distinct.'});
title(costLayout,{'Cost per native computational unit', ...
    'Solid lines: log-power fit below 2^{17} full DOFs. Dashed extensions: predictions, not fitted high-mesh data.'});
title(resLayout,{'Stage-time relative residuals: comparable fitting objectives', ...
    'Both curves minimize squared log-time error; fixed p = 1.5 is a reference hypothesis.'});
title(ratioLayout,{'Timing transition: observed unit cost / prediction from lower meshes', ...
    'A ratio of 1 means agreement. The threshold is from the prior audit, not estimated from these points.'});
title(cvLayout,{'Stage-time validation: equal weight per mesh', ...
    'Below -> above trains below 2^{17} full DOFs. These checks do not establish extrapolation accuracy.'});
if ~allowed
    for f={countsFig,costFig,resFig,ratioFig,cvFig}
        annotation(f{1},'textbox',[.25 .01 .5 .035],'String', ...
            'NO FIT: campaign scaling rules refuse fitting','EdgeColor','none','HorizontalAlignment','center');
    end
end
files=struct();
figs={countsFig,costFig,resFig,ratioFig,cvFig};
base={'native_counts','cost_per_native_unit','stage_time_residuals','transition_ratios','stage_time_validation'};
for k=1:numel(figs)
    files.(base{k})=fullfile(od,[base{k} '.png']);
    exportgraphics(figs{k},files.(base{k}),'Resolution',180,'BackgroundColor','white');
    % The figures are built invisible; the CreateFcn makes a reopened .fig show.
    set(figs{k},'CreateFcn','set(gcbo,''Visible'',''on'')');
    files.([base{k} '_fig'])=fullfile(od,[base{k} '.fig']);
    savefig(figs{k},files.([base{k} '_fig'])); close(figs{k});
end
writetable(struct2table(raw),fullfile(od,'native_costs.csv'));
writetable(struct2table(fits),fullfile(od,'prethreshold_fits.csv'));
writetable(struct2table(residuals),fullfile(od,'stage_time_residuals.csv'));
writetable(struct2table(validation),fullfile(od,'stage_time_validation.csv'));
files.directory=od;
fid=fopen(fullfile(od,'README.md'),'w'); cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,['# Complexity diagnostics\n\nDerived from recorded timings and counts; no solver rerun. ' ...
    'Stage time = Time 1 + Time 2. Other and total wall time are not changed.\n\n' ...
    '- native_counts.png: each method''s native counts, never summed across nested levels.\n' ...
    '- cost_per_native_unit.png: measured component time / native count, with log-power fits below 2^17 full DOFs.\n' ...
    '- transition_ratios.png: observed unit cost divided by that below-threshold prediction.\n' ...
    '- stage_time_residuals.png: free and fixed 1.5 exponents fitted in the SAME log-time space.\n' ...
    '- stage_time_validation.png: leave-one-mesh-out and below-to-above-threshold prediction MAPE.\n\n' ...
    'Each PNG has a MATLAB .fig of the same name for re-opening and restyling.\n\n' ...
    'The CSVs contain full-precision observations, fitted parameters, residuals, predictions, ' ...
    'MAPE, RMSE in seconds and RMSE in log time. Residual sign is prediction / observation - 1. ' ...
    'Fits require the campaign scaling gate and finite positive ok records. ' ...
    'Measured counts and costs remain visible even when a fit is refused; use CSV status/eligibility fields.\n\n' ...
    'The vertical line marks the FIRST OBSERVED mesh at or above 2^17 full DOFs, ' ...
    'computed as 2*(nelx+1)*(nely+1). It does not estimate a continuous breakpoint. ' ...
    'Only two current meshes lie above it. This is consistent with the prior sparse-assembly audit, ' ...
    'not a new causal experiment. Do not interpret nine observations as proof of asymptotic complexity ' ...
    'or repeated-run timing uncertainty. Unit-cost fits do not predict iteration counts.\n\n' ...
    'The original four complexity plots retain their historical free-log/fixed-seconds objectives; ' ...
    'these supplementary residual comparisons use log space for BOTH models.\n\n' ...
    'Regenerate via confbench_refresh_complexity_plots(campaignDir), with conference_bench and ' ...
    'analysis/Olhoff on the MATLAB path. New benchmark runs generate this set automatically.\n']);
end

function f=newFigure(name)
f=figure('Name',name,'Visible','off','Color','white','Position',[50 50 1500 600]);
end
function decorate(ax,name,label,logY)
set(ax,'XScale','log'); if logY; set(ax,'YScale','log'); end
xlabel(ax,'Number of elements N_e'); ylabel(ax,label); title(ax,name,'Interpreter','none');
grid(ax,'on'); box(ax,'on'); legend(ax,'Location','northoutside','FontSize',8,'Interpreter','none');
end
function transition(ax,x,high)
if any(high) && any(~high)
    xline(ax,x(find(high,1)),':','First mesh above threshold','Color',[.4 .4 .4], ...
        'HandleVisibility','off','LabelVerticalAlignment','bottom','FontSize',8);
end
end
function [C,p]=powerFit(x,y,fixed)
good=isfinite(x)&isfinite(y)&x>0&y>0; x=x(good); y=y(good); C=NaN; p=NaN;
if numel(x)<3; return; end
if isnan(fixed)
    b=[ones(numel(x),1),log(x(:))]\log(y(:)); C=exp(b(1)); p=b(2);
else
    p=fixed; C=exp(mean(log(y)-p*log(x)));
end
end
function e=errorsOf(pred,y)
good=isfinite(pred)&isfinite(y)&pred>0&y>0; pred=pred(good); y=y(good);
e=struct('n',numel(y),'mape',NaN,'rmse',NaN,'logrmse',NaN);
if isempty(y); return; end
e.mape=100*mean(abs(pred./y-1)); e.rmse=sqrt(mean((pred-y).^2)); e.logrmse=sqrt(mean(log(pred./y).^2));
end
function s=append(s,r)
if isempty(s); s=r; else; s(end+1)=r; end
end
function v=value(s,key)
v=NaN;
if isfield(s,key) && isnumeric(s.(key)) && isscalar(s.(key)); v=s.(key); end
end
