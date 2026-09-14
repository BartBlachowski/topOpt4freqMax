function seconds = confbench_stage_time(T)
%CONFBENCH_STAGE_TIME Sum the recorded native stages before display rounding.
% Missing stage timers (for example a failed solve) remain unmeasured.
seconds = NaN;
if ~isfield(T, 'time1') || ~isfield(T, 'time2'); return; end
if ~isnumeric(T.time1) || ~isscalar(T.time1) || ~isfinite(T.time1) || ...
        ~isnumeric(T.time2) || ~isscalar(T.time2) || ~isfinite(T.time2)
    return
end
seconds = T.time1 + T.time2;
end
