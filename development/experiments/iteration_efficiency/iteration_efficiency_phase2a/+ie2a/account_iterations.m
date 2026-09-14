function a = account_iterations(method, native, eligibleK)
%ACCOUNT_ITERATIONS Frozen method-native and acceptance-axis accounting.
arguments
    method (1,:) char
    native struct
    eligibleK (1,1) double {mustBeInteger,mustBeNonnegative}
end
switch lower(method)
    case 'proposed'
        a=struct('eligible_iteration',eligibleK,'native_iteration',eligibleK, ...
            'chronological_update',eligibleK,'unit','OC update');
    case 'yuksel'
        assert(isfield(native,'stage1_updates'),'ie2a:YukselAccounting','stage1_updates is required.');
        a=struct('eligible_iteration',eligibleK,'native_iteration',eligibleK, ...
            'chronological_update',native.stage1_updates+eligibleK, ...
            'stage1_updates',native.stage1_updates,'unit','Stage-2 OC update');
    case 'olhoff'
        assert(~isfield(native,'nInner_as_solver_iterations'),'ie2a:OlhoffAccounting', ...
            'nInner may not be reported as solver iterations.');
        variant=lower(char(localField(native,'variant','lp')));
        if strcmp(variant,'lp')
            a=struct('variant','lp','route_role','principal','eligible_iteration',eligibleK, ...
                'native_iteration',eligibleK,'chronological_update',eligibleK, ...
                'unit','successful outer update','outer_iterations',eligibleK, ...
                'lp_calls',localField(native,'lp_calls',NaN), ...
                'genuine_solver_iterations',localField(native,'genuine_solver_iterations',NaN));
        elseif strcmp(variant,'mma')
            inner=double(localField(native,'inner_iterations',nan(0,1)));inner=inner(:);
            converged=logical(localField(native,'inner_converged',false(size(inner))));
            cap=logical(localField(native,'inner_cap_hit',false(size(inner))));
            assert(numel(inner)==eligibleK&&numel(converged)==eligibleK&&numel(cap)==eligibleK, ...
                'ie2a:OlhoffAccounting','MMA requires one inner-work record per outer update.');
            a=struct('variant','mma','route_role','secondary_paper_literal', ...
                'eligible_iteration',eligibleK,'native_iteration',eligibleK, ...
                'chronological_update',eligibleK,'unit','successful outer update', ...
                'outer_iterations',eligibleK,'total_mma_inner_iterations',sum(inner), ...
                'mean_mma_inner_iterations',mean(inner),'median_mma_inner_iterations',median(inner), ...
                'p95_mma_inner_iterations',localP95(inner),'inner_cap_hits',sum(cap), ...
                'converged_inner_fraction',mean(converged));
        else
            error('ie2a:OlhoffVariant','Unknown Olhoff variant %s.',variant);
        end
    otherwise
        error('ie2a:UnknownMethod','Unknown frozen method: %s',method);
end
end
function x=localP95(v)
v=sort(v(isfinite(v)));if isempty(v),x=NaN;else,x=v(max(1,ceil(.95*numel(v))));end
end
function x=localField(s,n,d)
if isfield(s,n),x=s.(n);else,x=d;end
end
