function [cls, breakdown, reason] = check_a4_run(r)
%CHECK_A4_RUN  A4 acceptance classifier (A4_SPECIFICATION_V3 Part 5).
%
%   [cls, breakdown, reason] = CHECK_A4_RUN(r)
%
%   Returns:
%     cls        'REJECTED' | 'ACCEPTED' | 'ACCEPTED_WITH_BREAKDOWN'
%     breakdown  '' | 'B1' | 'B2' | 'B3' | 'B4'
%     reason     human-readable justification
%
%   THIS IS THE SINGLE IMPLEMENTATION OF THE RULE. It mirrors
%   check_revision_run.m, but the A4 rule is DELIBERATELY DIFFERENT:
%
%     *** A capped run and a lost mode are RESULTS, NOT REJECTIONS. ***
%
%   The campaign-wide rule "a capped run is a failure, not a result" is correct
%   for evidence runs.  It is WRONG for A4, whose purpose is to characterize
%   failure.  A gate that rejected mode loss and capping would discard exactly
%   the runs that carry the finding.  See spec §5.1.
%
%   Class A -- REJECTED (experiment failure; the machinery broke)
%     exception / solver abort; NaN-Inf in the endpoint; final eigensolve
%     failed; volume constraint violated beyond tolerance; missing artifact;
%     factor drift; non-deterministic replay.
%
%   Class B -- ACCEPTED (clean run)
%     converged before the cap, feasible, tracked mode retained (MAC >= 0.8),
%     no contamination signature.  ONLY Class B arms may serve as an accuracy
%     reference in the H0/H1 decision (§5.3).
%
%   Class C -- ACCEPTED_WITH_BREAKDOWN (a valid scientific observation)
%     B1 mode migration            MAC >= 0.8 but j* ~= 1
%     B2 frozen-mode breakdown     MAC < 0.8, otherwise clean
%     B3 spurious contamination    refresh hit the §4.3.1 screen, or
%                                  omega1_min << omega1_tracked
%     B4 sensitivity instability   limit cycle + high omitted-term ratio
%
%   Required fields of r (see spec §7.5):
%     success, exception_id, iterations, cap, final_design_change, tol,
%     feasibility, omega1_tracked, omega1_min, mac_to_phi0, mode_index_jstar,
%     refresh_inadmissible, limit_cycle, omitted_term_ratio

cls = 'REJECTED'; breakdown = ''; reason = '';

macThreshold          = localGet(r, 'mac_threshold', 0.8);
% Feasibility tolerance. The spec (§5.2, Class A) says a run is REJECTED when the
% "volume constraint [is] violated beyond feasibility tolerance (indicates a
% BROKEN OC)" but declares no number -- it is an implementation detail.
% IMPLEMENTATION DECISION: 1e-3 RELATIVE, i.e. |mean(x)-Vf|/Vf.
% Rationale: the OC Lagrange-multiplier bisection terminates on a bracket
% tolerance and therefore leaves a NORMAL relative volume residual of order
% 1e-5. A threshold of 1e-8 would reject healthy OC runs. 1e-3 is two orders of
% magnitude above the normal residual, so it fires only on a genuinely broken
% volume update -- which is what the spec asks it to detect.
feasTol               = localGet(r, 'feasibility_tol', 1e-3);
spuriousGapRatio      = localGet(r, 'spurious_gap_ratio', 0.5);   % omega1_min < 0.5*omega1_tracked
omittedTermThreshold  = localGet(r, 'omitted_term_threshold', 0.5);

% ---------------------------------------------------------------- Class A
if ~isstruct(r) || numel(r) ~= 1
    reason = 'invalid result schema (not a scalar struct)';
    return;
end

required = {'success','iterations','cap','final_design_change','tol', ...
            'omega1_tracked','omega1_min','mac_to_phi0','mode_index_jstar'};
missing = required(~isfield(r, required));
if ~isempty(missing)
    reason = sprintf('invalid result schema (missing field(s): %s)', strjoin(missing, ', '));
    return;
end

% B3 has priority over every other classification when the refresh itself was
% inadmissible: the arm produced no usable reference and must not be read as
% accuracy evidence, whatever its endpoint says.
refreshInadmissible = localGet(r, 'refresh_inadmissible', false);

if ~refreshInadmissible
    % An exception that is NOT the inadmissible-refresh signal is a hard failure.
    excId = localGet(r, 'exception_id', '');
    if ~isempty(excId)
        reason = sprintf('exception during run: %s', excId);
        return;
    end
    if ~localGet(r, 'success', false)
        reason = 'success=false (the run did not complete)';
        return;
    end
    if ~isfinite(r.omega1_tracked) || ~isfinite(r.omega1_min)
        reason = 'endpoint eigensolve produced non-finite frequencies';
        return;
    end
    feas = localGet(r, 'feasibility', 0);
    if isfinite(feas) && feas > feasTol
        reason = sprintf('volume constraint violated: feasibility %.3e > %.3e (broken OC)', ...
            feas, feasTol);
        return;
    end
    if localGet(r, 'factor_drift', false)
        reason = 'factor drift: the arm differs from the base config in more than N';
        return;
    end
    if localGet(r, 'nondeterministic', false)
        reason = 'non-deterministic replay';
        return;
    end
end

% ---------------------------------------------------------------- Class C
capped = isfinite(r.iterations) && isfinite(r.cap) && r.cap > 0 && r.iterations >= r.cap;
converged = isfinite(r.final_design_change) && isfinite(r.tol) && ...
            r.final_design_change <= r.tol && ~capped;

% B3 -- spurious-mode contamination (the refresh locked onto a disconnected
% island, or a non-physical mode descended below the tracked one).
spurious = false;
if refreshInadmissible
    spurious = true;
elseif isfinite(r.omega1_min) && isfinite(r.omega1_tracked) && r.omega1_tracked > 0
    spurious = r.omega1_min < spuriousGapRatio * r.omega1_tracked;
end
if spurious
    cls = 'ACCEPTED_WITH_BREAKDOWN';
    breakdown = 'B3';
    if refreshInadmissible
        reason = ['B3: refresh reference inadmissible -- every candidate failed the ' ...
                  'support-connectivity screen (spec 4.3.1). This arm is DISQUALIFIED as ' ...
                  'an accuracy reference.'];
    else
        reason = sprintf(['B3: spurious-mode contamination -- omega1_min %.4f << ' ...
            'omega1_tracked %.4f. This arm is DISQUALIFIED as an accuracy reference.'], ...
            r.omega1_min, r.omega1_tracked);
    end
    return;
end

% B4 -- sensitivity-omission instability (limit cycle + large omitted term).
limitCycle = localGet(r, 'limit_cycle', false);
omittedRatio = localGet(r, 'omitted_term_ratio', NaN);
if limitCycle && isfinite(omittedRatio) && omittedRatio > omittedTermThreshold
    cls = 'ACCEPTED_WITH_BREAKDOWN';
    breakdown = 'B4';
    reason = sprintf(['B4: bounded limit cycle with omitted-term ratio %.3f > %.2f. ' ...
        'Non-convergence is attributable to (refresh x omitted df/dx), NOT to freezing. ' ...
        'Says nothing about the frozen approximation.'], omittedRatio, omittedTermThreshold);
    return;
end

% B2 -- frozen-mode breakdown (the tracked mode is no longer recognizable).
if isfinite(r.mac_to_phi0) && r.mac_to_phi0 < macThreshold
    cls = 'ACCEPTED_WITH_BREAKDOWN';
    breakdown = 'B2';
    reason = sprintf(['B2: frozen-mode breakdown -- MAC to Phi0 = %.4f < %.2f. ' ...
        'The initial mode is a poor proxy for the optimized mode.'], ...
        r.mac_to_phi0, macThreshold);
    return;
end

% B1 -- mode migration (mode still found, but no longer the lowest).
if isfinite(r.mode_index_jstar) && r.mode_index_jstar ~= 1
    cls = 'ACCEPTED_WITH_BREAKDOWN';
    breakdown = 'B1';
    reason = sprintf(['B1: mode migration -- tracked mode is index %d (MAC %.4f >= %.2f). ' ...
        'The approximation is intact; the ORDERING shifted. Not a failure of freezing.'], ...
        r.mode_index_jstar, r.mac_to_phi0, macThreshold);
    return;
end

% Capping with no other signature is still an observation, not a rejection.
if capped
    cls = 'ACCEPTED_WITH_BREAKDOWN';
    breakdown = 'B4';
    reason = sprintf(['B4 (unattributed): reached the iteration cap %d/%d with final design ' ...
        'change %.3e > tol %.3e, without a limit-cycle/omitted-term signature. Reported as an ' ...
        'observation; may NOT be used as a converged accuracy reference.'], ...
        r.iterations, r.cap, r.final_design_change, r.tol);
    return;
end

% ---------------------------------------------------------------- Class B
if converged
    cls = 'ACCEPTED';
    breakdown = '';
    reason = sprintf(['clean run: converged %d/%d, design change %.3e <= %.3e, MAC %.4f, ' ...
        'j*=%d. Eligible as an accuracy reference.'], ...
        r.iterations, r.cap, r.final_design_change, r.tol, r.mac_to_phi0, r.mode_index_jstar);
    return;
end

reason = sprintf('unconverged without a recognized breakdown signature (change %.3e > tol %.3e)', ...
    r.final_design_change, r.tol);
cls = 'REJECTED';
end

% =========================================================================

function v = localGet(s, name, default)
v = default;
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
end
end
