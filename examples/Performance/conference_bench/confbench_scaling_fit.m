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

% ---- per-outer-iteration cost, for methods whose records carry an outer count
% (the Du-Olhoff reconstruction).  Same rows as the total-time fit.  The outer
% count need not be monotone in the mesh, so total time alone is not a
% per-iteration scaling law; both are reported.
scaling.per_outer = struct('model', 'T/N_outer (Ne) = C * Ne^p', ...
    'methods', struct('method', {}, 'quantity', {}, 'C', {}, 'p', {}, 'R2', {}, 'n', {}, 'meshes', {}));
for i = 1:numel(keys)
    sel = records(strcmp({records.method_key}, keys{i}));
    ok = sel(logical([sel.ok]));
    if isempty(ok) || ~all(arrayfun(@(r) isfield(r, 'counts') && isfield(r.counts, 'outer_iterations') && ...
            isfield(r, 'times') && isfield(r.times, 'total_wall_time_per_outer_s'), ok))
        continue
    end
    Ne = arrayfun(@(r) r.mesh(1)*r.mesh(2), ok).';
    meshes = arrayfun(@(r) sprintf('%dx%d', r.mesh(1), r.mesh(2)), ok, 'UniformOutput', false);
    Q = {'total_wall_time_per_outer_s', 'outer_time_excluding_inner_per_outer_mean_s', ...
         'eigen_time_per_outer_mean_s', 'inner_time_per_outer_mean_s', 'inner_time_per_inner_iteration_mean_s'};
    for q = 1:numel(Q)
        y = arrayfun(@(r) r.times.(Q{q}), ok).';
        [C, p, R2, n] = local_powerFit(Ne, y);
        scaling.per_outer.methods(end+1) = struct('method', confbench_display_name(keys{i}), ...
            'quantity', Q{q}, 'C', C, 'p', p, 'R2', R2, 'n', n, 'meshes', {meshes}); %#ok<AGROW>
    end
end
end

function [C, p, R2, n] = local_powerFit(Ne, T)
good = isfinite(Ne) & isfinite(T) & Ne > 0 & T > 0;
Ne = Ne(good); T = T(good); n = numel(Ne);
if n < 3, C = NaN; p = NaN; R2 = NaN; return; end
A = [ones(n,1), log(Ne)];
beta = A\log(T);
pred = A*beta;
R2 = 1 - sum((log(T)-pred).^2)/max(sum((log(T)-mean(log(T))).^2), eps);
C = exp(beta(1)); p = beta(2);
end
