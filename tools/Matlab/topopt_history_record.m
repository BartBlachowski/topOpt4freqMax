function H = topopt_history_record(H, rec)
%TOPOPT_HISTORY_RECORD  Append one iteration to a history recorder.
%
%   H = TOPOPT_HISTORY_RECORD(H, REC) appends a row.  REC is a struct:
%
%   Required
%     iter        global optimization iteration, 1-based
%     xPhys       current physical density field (vector)
%     volfrac     target volume fraction
%
%   Optional
%     stage, stage_iter    method-native stage index and local iteration
%     xPhysPrev            previous physical density field; omit on the first
%                          recorded iteration, where the increment is undefined
%     x, xOld, move_limit  design field before/after and the move limit, used
%                          for the design-variable increment and the fraction
%                          of variables sitting at their limit
%     objective, elapsed_s, omega1, mode_index, mac
%
%   Derived quantities are computed HERE and cannot be supplied by the caller,
%   so d_inf, d_rms, rV and grayness have one definition across all methods.
%   Anything absent stays NaN; no field is ever filled by an extra solve.
%
%   Definitions:
%     d_inf     = max_e |xPhys_e(k) - xPhys_e(k-1)|          (all elements)
%     d_rms     = sqrt(mean((xPhys(k) - xPhys(k-1)).^2))
%     rV        = |mean(xPhys) - volfrac| / volfrac
%     grayness  = mean(4 * xPhys .* (1 - xPhys))
%
%   d_inf is taken over every element, including passive ones, which is the
%   convention of plan section 4.2.  A solver that natively measures its own
%   change over an active subset only will therefore see a slightly different
%   number here; that is intended.

if H.n >= H.capacity
    % Growing rather than erroring keeps a run alive if it exceeds its budget;
    % the overshoot is visible in H.n against H.capacity.
    grow = max(64, ceil(0.5 * H.capacity));
    H = localGrow(H, grow);
end

k = H.n + 1;

xPhys = double(rec.xPhys(:));
volfrac = double(rec.volfrac);

H.iter(k) = rec.iter;
H.stage(k)      = localGet(rec, 'stage', 1);
H.stage_iter(k) = localGet(rec, 'stage_iter', NaN);

if isfield(rec, 'xPhysPrev') && ~isempty(rec.xPhysPrev)
    d = xPhys - double(rec.xPhysPrev(:));
    H.d_inf(k) = max(abs(d));
    H.d_rms(k) = sqrt(mean(d.^2));
end

if volfrac > 0
    H.rV(k) = abs(mean(xPhys) - volfrac) / volfrac;
end
H.grayness(k) = mean(4 * xPhys .* (1 - xPhys));

if isfield(rec, 'x') && isfield(rec, 'xOld') && ~isempty(rec.xOld)
    dx = double(rec.x(:)) - double(rec.xOld(:));
    H.d_inf_design(k) = max(abs(dx));
    H.d_rms_design(k) = sqrt(mean(dx.^2));
    if isfield(rec, 'move_limit') && ~isempty(rec.move_limit) && rec.move_limit > 0
        % "At the move limit" allows for the rounding of a clamped update.
        H.move_active_frac(k) = mean(abs(dx) >= rec.move_limit * (1 - 1e-9));
    end
end

H.objective(k)  = localGet(rec, 'objective', NaN);
H.elapsed_s(k)  = localGet(rec, 'elapsed_s', NaN);
H.omega1(k)     = localGet(rec, 'omega1', NaN);
H.mode_index(k) = localGet(rec, 'mode_index', NaN);
H.mac(k)        = localGet(rec, 'mac', NaN);

H.n = k;
end

function v = localGet(rec, name, defaultValue)
if isfield(rec, name) && ~isempty(rec.(name))
    v = double(rec.(name));
else
    v = defaultValue;
end
end

function H = localGrow(H, grow)
pad = NaN(grow, 1);
names = {'iter','stage','stage_iter','d_inf','d_rms','rV','grayness', ...
    'd_inf_design','d_rms_design','objective','elapsed_s', ...
    'move_active_frac','omega1','mode_index','mac'};
for i = 1:numel(names)
    H.(names{i}) = [H.(names{i}); pad];
end
H.capacity = H.capacity + grow;
end
