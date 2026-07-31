function H = topopt_history_mark(H, iter, kind, detail)
%TOPOPT_HISTORY_MARK  Record a stage or continuation transition.
%
%   H = TOPOPT_HISTORY_MARK(H, ITER, KIND, DETAIL) records that a transition of
%   type KIND occurred at global iteration ITER.  KIND is 'stage' for a native
%   stage handoff or 'continuation' for a penalization/projection step.  DETAIL
%   is a short human-readable string, e.g. 'beta 8 -> 16'.
%
%   These are recorded as events rather than reconstructed afterwards because
%   k_cont -- the iteration at which the last mandatory continuation transition
%   and any minimum-polish requirement clear -- is what separates a method's
%   schedule floor from its convergence (plan section 4.2.3).  A schedule
%   inferred from parameter values after the fact would miss any transition
%   triggered by run-time state, such as a grayness-triggered beta advance.

if nargin < 4 || isempty(detail)
    detail = '';
end
validKinds = {'stage', 'continuation'};
if ~any(strcmp(kind, validKinds))
    error('topopt_history_mark:InvalidKind', ...
        'kind must be one of: %s', strjoin(validKinds, ', '));
end

H.markers(end+1) = struct('iter', double(iter), 'kind', char(kind), ...
    'detail', char(detail));
end
