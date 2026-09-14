function [fits,support]=fit_scaling_table(T,metrics,opts)
%FIT_SCALING_TABLE C and p on explicitly constructed available/common support.
% Two labelled families are emitted per metric:
%   support = "available" -- each series over its own fit-eligible meshes;
%   support = "common"    -- every series restricted to the intersection of the
%                           fit-eligible meshes of all series participating in
%                           that metric's comparison.
% The common family is built from the intersection BEFORE fitting, so a mesh
% cannot enter a common fit merely because one method has a value there, and a
% common fit can never silently fall back to method-specific support.
%
% Eligibility is the frozen rule and is applied by the caller (P == P_primary
% and a fit-eligible status) plus, per metric, the frozen positive-finite test.
arguments
    T table
    metrics cell
    opts.MinimumValidMeshes (1,1) double {mustBeInteger,mustBePositive} = 3
end
methods=unique(string(T.method),'stable');
rows=struct([]);r=0;supportRows=struct([]);sr=0;
for im=1:numel(metrics)
    metric=metrics{im};
    y=double(T.(metric));Ne=double(T.element_count);
    % Frozen per-metric eligibility: positive and finite only.
    eligible=isfinite(Ne)&Ne>0&isfinite(y)&y>0;
    % Series that participate in this metric's comparison at all.
    participants=strings(0,1);
    for i=1:numel(methods)
        if any(eligible&string(T.method)==methods(i)),participants(end+1,1)=methods(i);end %#ok<AGROW>
    end
    % Explicit intersection of eligible meshes across all participants.
    if isempty(participants)
        common=zeros(0,1);
    else
        common=unique(Ne(eligible&string(T.method)==participants(1)));
        for i=2:numel(participants)
            common=intersect(common,unique(Ne(eligible&string(T.method)==participants(i))));
        end
    end
    common=sort(common(:));
    feasible=numel(common)>=opts.MinimumValidMeshes;
    sr=sr+1;srec=struct('metric',string(metric), ...
        'participants',strjoin(participants,','),'n_participants',numel(participants), ...
        'common_meshes',strjoin(string(common(:).'),','),'n_support',numel(common), ...
        'minimum_valid_meshes',opts.MinimumValidMeshes,'common_fit_feasible',feasible);
    if sr==1,supportRows=srec;else,supportRows(sr)=srec;end %#ok<AGROW>

    for i=1:numel(methods)
        isMethod=string(T.method)==methods(i);
        participates=any(participants==methods(i));

        % ---- available support: this series' own eligible meshes ----
        ownMeshes=sort(unique(Ne(eligible&isMethod)));
        fA=ie2a.fit_power_law(Ne(isMethod),y(isMethod),string(Ne(isMethod)));
        r=r+1;rec=localRec(methods(i),metric,"available",fA, ...
            numel(ownMeshes),strjoin(string(ownMeshes(:).'),','),true,"");
        if r==1,rows=rec;else,rows(r)=rec;end %#ok<AGROW>

        % ---- common support: restricted to the intersection ----
        ixC=isMethod&ismember(Ne,common);
        if ~participates
            fC=ie2a.fit_power_law(zeros(0,1),zeros(0,1),strings(0,1));
            note="series has no eligible value for this metric";
        elseif ~feasible
            % Fail closed: do not fit, do not fall back to a wider support.
            fC=ie2a.fit_power_law(zeros(0,1),zeros(0,1),strings(0,1));
            note=sprintf('common support has %d mesh(es), below the frozen minimum of %d', ...
                numel(common),opts.MinimumValidMeshes);
        else
            fC=ie2a.fit_power_law(Ne(ixC),y(ixC),string(Ne(ixC)));
            note="";
        end
        r=r+1;rows(r)=localRec(methods(i),metric,"common",fC, ...
            numel(common),strjoin(string(common(:).'),','),feasible&&participates,note); %#ok<AGROW>
    end
end
fits=struct2table(rows);
if sr==0,support=table();else,support=struct2table(supportRows);end
end

function rec=localRec(method,metric,supportLabel,f,nSupport,supportMeshes,fitted,note)
rec=struct('method',method,'metric',string(metric),'support',supportLabel, ...
    'n_support',nSupport,'support_meshes',string(supportMeshes),'fitted',logical(fitted), ...
    'C',f.C,'p',f.p,'R2_log',f.R2_log,'n_valid',f.n_valid,'Ne_min',f.Ne_min,'Ne_max',f.Ne_max, ...
    'included_meshes',string(f.included_meshes),'p_LOO_min',f.p_LOO_min,'p_LOO_max',f.p_LOO_max, ...
    'weakly_identified',f.weakly_identified,'exclusions',string(f.exclusions),'note',string(note));
end
