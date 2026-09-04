function acc = confbench_accounting(T, tolAbs, tolRel, xTolRel)
%CONFBENCH_ACCOUNTING  The uniform timing contract:  T_total = T1 + T2 + T_overhead.
%
%   acc = CONFBENCH_ACCOUNTING(times, tolAbs, tolRel, xTolRel)
%
%   `times` is a method's time struct: it must carry time1, time2,
%   overhead_time_s and total_wall_time_s, and may carry
%   independent_crosscheck_residual_s.
%
%   Two residuals are computed and they are NOT the same kind of thing:
%
%     timing_accounting_residual_s      the identity above.  Its components are
%                                       nested measured intervals, so this is
%                                       near-zero by construction and catches
%                                       a component that was mis-derived.
%     independent_crosscheck_residual_s the caller-side total minus the
%                                       solver's OWN self-reported wall time --
%                                       two separate measurements of the same
%                                       interval, so it catches a mis-nested
%                                       timer that the identity cannot see.
%
%   Each carries its predeclared tolerance and a boolean flag, so a reader of
%   the artifacts never has to guess what "within tolerance" meant.
acc = struct('total_wall_time_s', NaN, 'component_sum_s', NaN, ...
    'timing_accounting_residual_s', NaN, ...
    'timing_accounting_relative_residual', NaN, ...
    'timing_accounting_tolerance_s', tolAbs, ...
    'timing_accounting_tolerance_rel', tolRel, ...
    'timing_accounting_fail', true, ...
    'independent_crosscheck_residual_s', NaN, ...
    'independent_crosscheck_relative', NaN, ...
    'independent_crosscheck_tolerance_rel', xTolRel, ...
    'independent_crosscheck_fail', true);

if ~isstruct(T) || ~isfield(T, 'total_wall_time_s'); return; end
times = T;
T  = times.total_wall_time_s;
t1 = times.time1;
t2 = times.time2;
ov = times.overhead_time_s;

acc.total_wall_time_s = T;
acc.component_sum_s = t1 + t2 + ov;
acc.timing_accounting_residual_s = T - acc.component_sum_s;
acc.timing_accounting_relative_residual = acc.timing_accounting_residual_s/max(T, eps);
acc.timing_accounting_fail = ~(abs(acc.timing_accounting_residual_s) <= ...
    max(tolAbs, tolRel*abs(T)));

if isfield(times, 'independent_crosscheck_residual_s')
    r = times.independent_crosscheck_residual_s;
    acc.independent_crosscheck_residual_s = r;
    acc.independent_crosscheck_relative = r/max(T, eps);
    acc.independent_crosscheck_fail = ~(abs(r) <= max(0.05, xTolRel*abs(T)));
end
end
