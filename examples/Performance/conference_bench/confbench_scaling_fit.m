function scaling = confbench_scaling_fit(cfg, records)
%CONFBENCH_SCALING_FIT  Fit T(Ne) = C * Ne^p, but only when that is legitimate.
%
%   A scaling exponent may be fitted ONLY to complete campaign data.  Smoke
%   runs, preflight runs, truncated budgets and censored rows are all refused,
%   with the reason recorded rather than the fit quietly returning something.
%
%   Component times are preserved in the detailed CSV so component-wise scaling
%   can be examined later from the same data.
%
%   See also CONFBENCH_EXPORT, CONFBENCH_CAVEATS.

scaling = struct('fitted', false, 'reason', '', 'model', 'T(Ne) = C * Ne^p', ...
    'methods', struct('method', {}, 'C', {}, 'p', {}, 'R2', {}, 'n', {}, 'meshes', {}));

if ~cfg.fitScaling
    scaling.reason = 'cfg.fitScaling is false';
    return
end
if ~cfg.performanceCampaign
    scaling.reason = ['this run is not a complete performance campaign; a ' ...
        'scaling exponent must not be fitted to smoke or preflight data'];
    return
end
if ~cfg.scientificEvidence
    scaling.reason = 'this run is not scientific evidence (truncated budget or sub-floor mesh)';
    return
end

keys = unique({records.method_key}, 'stable');
for i = 1:numel(keys)
    sel = records(strcmp({records.method_key}, keys{i}));
    ok = sel(logical([sel.ok]));
    if numel(ok) < 3
        scaling.methods(end+1) = struct('method', confbench_display_name(keys{i}), ...
            'C', NaN, 'p', NaN, 'R2', NaN, 'n', numel(ok), 'meshes', {{}}); %#ok<AGROW>
        continue
    end
    Ne = arrayfun(@(r) r.mesh(1)*r.mesh(2), ok).';
    T  = arrayfun(@(r) r.times.total_wall_time_s, ok).';
    good = isfinite(Ne) & isfinite(T) & Ne > 0 & T > 0;
    Ne = Ne(good); T = T(good);
    A = [ones(numel(Ne),1), log(Ne)];
    beta = A\log(T);
    pred = A*beta;
    ss = 1 - sum((log(T)-pred).^2)/max(sum((log(T)-mean(log(T))).^2), eps);
    meshes = arrayfun(@(r) sprintf('%dx%d', r.mesh(1), r.mesh(2)), ok(good), ...
        'UniformOutput', false);
    scaling.methods(end+1) = struct('method', confbench_display_name(keys{i}), ...
        'C', exp(beta(1)), 'p', beta(2), 'R2', ss, 'n', numel(Ne), ...
        'meshes', {meshes}); %#ok<AGROW>
end
scaling.fitted = true;
scaling.caveat = confbench_caveats().scaling;
end
