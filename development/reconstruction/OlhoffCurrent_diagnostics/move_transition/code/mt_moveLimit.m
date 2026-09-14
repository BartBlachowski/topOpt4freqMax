function [mv, state] = mt_moveLimit(cfg, outer, hist, state, tcfg)
%MT_MOVELIMIT  Move-limit policy with a SELECTABLE stage-transition signal.
%
%   [mv, state] = MT_MOVELIMIT(cfg, outer, hist, state, tcfg)
%
%   This is the ONE experimental factor of the move-transition study.  It is
%   DEFAULT-OFF: with no tcfg, or with
%
%       tcfg.metric = 'boundVariableStall'
%
%   it delegates verbatim to the production controller olh.move.limit and is
%   therefore bit-identical to production.  Nothing under +impl/ is modified by
%   this study; the production source manifest must still hash unchanged after
%   it.
%
%   THE CANDIDATE  (tcfg.metric = 'maxUtilization')
%   -----------------------------------------------
%       tcfg.threshold    = 0.5     frozen, inherited from the admission-rule
%                                   preregistration (tau_rel)
%       tcfg.persistence  = 10      frozen, = cfg.move.continuation.window
%
%   The ladder descends one rung to the next EXISTING level of cfg.move.levels
%   when, and only when, the design-utilization ratio
%
%       r_rho(j) = max_e |drho_e(j)| / move(j)
%
%   has been below tcfg.threshold on tcfg.persistence CONSECUTIVE completed
%   iterations at the current, unchanged move level.  Semantics are literal and
%   are implemented once, in mt_utilCount.
%
%   WHY THE RATIO AND NOT max|drho| ITSELF
%   --------------------------------------
%   innerLoop builds the box  lo = max(rhomin-rho, -move),  hi = min(1-rho, move),
%   so max|drho| <= move identically and r_rho lies in [0,1].  An ABSOLUTE
%   threshold on max|drho| is manufacturable -- halving the move halves the
%   bound on the statistic with no change in the design's behaviour.  r_rho is
%   dimensionless in the move: a design saturating its bound has r_rho = 1
%   whatever the bound is, so a descent cannot reduce it.  r_rho asks whether
%   the move bound is still ACTIVE, i.e. whether the design is still using the
%   step it is allowed.
%
%   EVIDENCE CLASS.  The ladder, its levels, its transition criterion and every
%   number here are CLASS C reconstruction.  "move limit", "trust region",
%   "step size" and "continuation" occur ZERO times in Du & Olhoff (2007) and in
%   Olhoff & Du (2014).  Nothing in this file may be described as the authors'.
%
%   state fields: as olh.move.limit, plus util.count (the persistence counter
%   that was in force at this call) and util.startJ.
%
%   See also OLH.MOVE.LIMIT, MT_UTILCOUNT, MT_OLHOFFSOLVET.

% ---- default-off: the production controller, verbatim -------------------
if nargin < 5 || isempty(tcfg) || ~isfield(tcfg,'metric') || ...
        strcmp(tcfg.metric, 'boundVariableStall')
    [mv, state] = olh.move.limit(cfg, outer, hist, state);
    return
end

assert(strcmp(tcfg.metric,'maxUtilization'), 'mt_moveLimit:Metric', ...
    'unknown move.transition.metric ''%s''', tcfg.metric);
policy = olh.config.getPath(cfg, 'move.policy');
assert(strcmp(policy,'ladder'), 'mt_moveLimit:Policy', ...
    ['the utilization-gated transition is defined only for the ladder ' ...
     'policy; cfg.move.policy = ''%s'''], policy);

levels = olh.config.getPath(cfg, 'move.levels');

% State shape identical to olh.move.limit's, so nothing downstream can tell the
% controllers apart by their state.
if isempty(state)
    state = struct('mv',olh.config.getPath(cfg,'move.initial'),'stage',1, ...
                   'lastBeta',NaN,'stall',0,'ratioHist',[],'coalSeen',false, ...
                   'lastStage',0,'lastRealized',NaN);
end
if ~isfield(state,'util')
    state.util = struct('count',0,'startJ',1,'rrho',NaN);
end
if ~state.coalSeen && outer > 1 && ~isempty(hist.N) && any(hist.N >= 2)
    state.coalSeen = true;   % unused by this policy; kept for state fidelity
end

% ---- the utilization history over COMPLETED iterations ------------------
% hist is populated for iterations 1..outer-1 at this point in olhoffSolve:
% hist.dxOuter(j) is the realized max|drho| at j and hist.move(j) is the move
% limit that was actually in force at j.
upTo   = outer - 1;
startJ = max(state.lastStage, 1);   % first iteration at the current level
count  = 0;
if upTo >= 1
    rrho  = hist.dxOuter(1:upTo) ./ hist.move(1:upTo);
    count = mt_utilCount(rrho, startJ, upTo, tcfg.threshold);
    state.util.rrho = rrho(upTo);
else
    state.util.rrho = NaN;
end

% ---- descend only on the frozen criterion -------------------------------
% The counter is reset ONLY by an actual transition (or by a violation, inside
% mt_utilCount).  At the terminal rung there is no next level, so no transition
% occurs and the counter is left to keep growing -- which is what the study
% needs to measure.  This differs in shape from olh.move.limit, which advances
% its re-arm clock even at the terminal rung; the difference is unreachable
% before the last level and is recorded here rather than hidden.
if count >= tcfg.persistence && state.stage < numel(levels)
    state.stage     = state.stage + 1;
    state.lastStage = outer;
    count           = 0;
end

state.util.count  = count;
state.util.startJ = startJ;

mv       = levels(state.stage);
state.mv = mv;
end
