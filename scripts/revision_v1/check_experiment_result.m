function [ok, issues] = check_experiment_result(r)
%CHECK_EXPERIMENT_RESULT  Validate a result struct for master-runner acceptance.
%
%   [ok, issues] = check_experiment_result(r)
%
%   ok     -- true iff the struct is complete and internally consistent
%   issues -- cell array of one-line problem strings (empty when ok=true)
%
%   This function checks schema completeness and logical consistency.
%   It does NOT check whether the experiment succeeded; inspect r.success
%   and r.termination for that.  A properly filled failure result passes
%   this check (ok=true, r.success=false).
%
%   CHECKS PERFORMED
%   ----------------
%   S1  Required top-level fields present
%   S2  r.termination.reason is not 'unknown'
%   S3  r.iterations.count and .cap are finite and non-negative
%   S4  r.convergence scalars are finite (not NaN)
%   S5  r.histories.objective is non-empty
%   S6  r.timing.total_s is finite and positive
%   C1  r.success=true requires r.termination.converged=true
%   C2  r.success=true requires r.termination.capped=false
%   C3  r.success=true requires r.termination.mode_lost=false
%   C4  r.termination.capped=true requires iterations.count == iterations.cap
%   C5  r.termination.mode_lost=true requires mac_history to be non-empty
%
%   See also MAKE_EXPERIMENT_RESULT, FILL_RESULT_PROVENANCE.

issues = {};

% ---- S1: struct with required top-level fields --------------------------
if ~isstruct(r)
    issues{end+1} = 'result is not a struct';
    ok = false;
    return;
end

required = {'success','termination','iterations','convergence', ...
            'histories','mode_tracking','timing','provenance','artifacts'};
for k = 1:numel(required)
    if ~isfield(r, required{k})
        issues{end+1} = sprintf('missing top-level field: r.%s', required{k}); %#ok<AGROW>
    end
end
if ~isempty(issues), ok = false; return; end

% ---- S2: termination reason explicitly set ------------------------------
if strcmp(r.termination.reason, 'unknown')
    issues{end+1} = ...
        'r.termination.reason is ''unknown''; set to ''converged'', ''iteration_cap'', ''mode_invalid'', or ''exception''';
end

% ---- S3: iterations -----------------------------------------------------
if ~isfield(r.iterations, 'count') || ~isfinite(r.iterations.count) || r.iterations.count < 0
    issues{end+1} = 'r.iterations.count must be a non-negative finite number';
end
if ~isfield(r.iterations, 'cap') || ~isfinite(r.iterations.cap) || r.iterations.cap <= 0
    issues{end+1} = 'r.iterations.cap must be a positive finite number';
end

% ---- S4: convergence scalars --------------------------------------------
if ~isfield(r.convergence, 'final_design_change') || ~isfinite(r.convergence.final_design_change)
    issues{end+1} = 'r.convergence.final_design_change is NaN or missing';
end
if ~isfield(r.convergence, 'constraint_residual') || ~isfinite(r.convergence.constraint_residual)
    issues{end+1} = 'r.convergence.constraint_residual is NaN or missing';
end

% ---- S5: objective history non-empty ------------------------------------
if ~isfield(r.histories, 'objective') || isempty(r.histories.objective)
    issues{end+1} = 'r.histories.objective is empty';
end

% ---- S6: total timing ---------------------------------------------------
if ~isfield(r.timing, 'total_s') || ~isfinite(r.timing.total_s) || r.timing.total_s <= 0
    issues{end+1} = 'r.timing.total_s must be a positive finite number';
end

% ---- C1-C3: success implies converged and no failure flags --------------
if r.success
    if ~r.termination.converged
        issues{end+1} = 'r.success=true but r.termination.converged=false';
    end
    if r.termination.capped
        issues{end+1} = 'r.success=true but r.termination.capped=true';
    end
    if r.termination.mode_lost
        issues{end+1} = 'r.success=true but r.termination.mode_lost=true';
    end
end

% ---- C4: capped flag requires count == cap ------------------------------
if r.termination.capped
    cnt = r.iterations.count;
    cap = r.iterations.cap;
    if isfinite(cnt) && isfinite(cap) && cnt ~= cap
        issues{end+1} = sprintf( ...
            'r.termination.capped=true but iterations.count=%d != cap=%d', cnt, cap);
    end
end

% ---- C5: mode_lost flag requires MAC history ----------------------------
if r.termination.mode_lost
    if ~isfield(r.mode_tracking, 'mac_history') || isempty(r.mode_tracking.mac_history)
        issues{end+1} = ...
            'r.termination.mode_lost=true but r.mode_tracking.mac_history is empty';
    end
end

ok = isempty(issues);
end
