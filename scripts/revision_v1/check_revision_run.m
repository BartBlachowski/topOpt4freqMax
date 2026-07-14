function [ok, reason] = check_revision_run(label, run)
%CHECK_REVISION_RUN  Apply the declared revision acceptance rule to one run.
%
%   [ok, reason] = check_revision_run(label, run)
%
%   This introduces NO new convergence criteria.  It is the same rule already
%   implemented by exp2_authoritative_sweep/localClassify and stated in
%   REVISION_EXECUTION_PLAN.md:
%
%       not capped  AND  design_change <= declared tolerance
%
%   where the declared tolerance is the run's own configured
%   optimization.convergence_tol, and "capped" means iterations >= max_iters.
%   A run that reaches the iteration cap is a FAILURE, not a result.
%
%   RUN is a struct describing a single completed optimization:
%     .success        logical  -- solver returned without exception
%     .iterations     double   -- iterations executed
%     .cap            double   -- iteration cap (optimization.max_iters)
%     .design_change  double   -- final design change (NaN if not reported)
%     .tol            double   -- declared tolerance (optimization.convergence_tol)
%
%   Rejection order (first match wins):
%     R1  invalid schema                -- run is not a struct / fields absent
%     R2  success = false               -- the run did not complete
%     R3  missing termination metadata  -- iterations or cap absent/non-finite
%     R4  iteration cap reached         -- iterations >= cap
%     R5  missing convergence metadata  -- design_change absent or NaN
%     R6  unconverged design change     -- design_change > tol
%
%   See also CHECK_EXPERIMENT_RESULT.

ok = false;

% ---- R1: schema ---------------------------------------------------------
if ~isstruct(run) || ~isscalar(run)
    reason = sprintf('%s: invalid result schema (not a scalar struct)', label);
    return;
end
required = {'success', 'iterations', 'cap', 'design_change', 'tol'};
missing = required(~isfield(run, required));
if ~isempty(missing)
    reason = sprintf('%s: invalid result schema (missing field(s): %s)', ...
        label, strjoin(missing, ', '));
    return;
end

% ---- R2: run completed --------------------------------------------------
if ~islogical(run.success) && ~isnumeric(run.success)
    reason = sprintf('%s: invalid result schema (success is not logical)', label);
    return;
end
if ~run.success
    reason = sprintf('%s: success=false (the run did not complete)', label);
    return;
end

% ---- R3: termination metadata ------------------------------------------
if isempty(run.iterations) || ~isfinite(run.iterations) || run.iterations < 0 || ...
        isempty(run.cap) || ~isfinite(run.cap) || run.cap <= 0
    reason = sprintf(['%s: missing termination metadata ' ...
        '(iterations/cap absent or non-finite); cannot prove the run was not capped'], label);
    return;
end

% ---- R4: iteration cap = failure ---------------------------------------
if run.iterations >= run.cap
    reason = sprintf(['%s: reached iteration cap %d/%d without converging; ' ...
        'a capped run is a failure, not a result'], ...
        label, round(run.iterations), round(run.cap));
    return;
end

% ---- R5: convergence metadata ------------------------------------------
if isempty(run.design_change) || ~isfinite(run.design_change)
    reason = sprintf(['%s: missing convergence metadata ' ...
        '(final design change not reported); convergence cannot be verified'], label);
    return;
end

% ---- R6: declared tolerance --------------------------------------------
if isempty(run.tol) || ~isfinite(run.tol) || run.tol <= 0
    reason = sprintf('%s: invalid result schema (declared tolerance absent or non-positive)', label);
    return;
end
if run.design_change > run.tol
    reason = sprintf(['%s: final design change %.3e exceeds the declared ' ...
        'tolerance %.3e (optimization.convergence_tol); run is unconverged'], ...
        label, run.design_change, run.tol);
    return;
end

ok = true;
reason = '';
end
