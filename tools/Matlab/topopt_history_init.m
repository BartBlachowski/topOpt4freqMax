function H = topopt_history_init(maxRows, meta)
%TOPOPT_HISTORY_INIT  Preallocate an iteration-history recorder.
%
%   H = TOPOPT_HISTORY_INIT(MAXROWS, META) returns an empty history sized for
%   MAXROWS iterations.  META is a struct of run-level context stored alongside
%   the arrays -- at minimum the sign and definition of the native objective,
%   which differs between methods and is meaningless without it.
%
%   Schema: examples/Performance/PLAN_two_table_redesign.md section 5.
%
%   The recorder is deliberately shared by all three solvers.  The controlled
%   benchmark compares d_inf and rV across methods, so those quantities must
%   have exactly one definition; per-solver copies would let them drift while
%   still looking comparable.  Computing them lives in TOPOPT_HISTORY_RECORD so
%   a caller cannot supply its own.
%
%   See also TOPOPT_HISTORY_RECORD, TOPOPT_HISTORY_MARK, TOPOPT_HISTORY_FINISH.

if nargin < 2 || isempty(meta)
    meta = struct();
end
maxRows = max(1, floor(double(maxRows)));

z = NaN(maxRows, 1);

H = struct();
H.n = 0;
H.capacity = maxRows;
H.meta = meta;

% Iteration identity
H.iter       = z;   % global optimization iteration, 1-based
H.stage      = z;   % method-native stage index (1 for single-stage methods)
H.stage_iter = z;   % iteration within the current stage

% Common acceptance quantities, on the PHYSICAL density field
H.d_inf   = z;      % max_e |xPhys_e(k) - xPhys_e(k-1)|
H.d_rms   = z;      % RMS of the same increment
H.rV      = z;      % |V(xPhys) - Vtarget| / Vtarget
H.grayness = z;     % mean(4 * xPhys .* (1 - xPhys))

% The same increment on the DESIGN variable.  Recorded because xPhys is the
% identity map on the design field for sensitivity-filtered methods but a
% projected field for density-filtered ones, so a criterion applied to xPhys is
% not automatically neutral between them.
H.d_inf_design = z;
H.d_rms_design = z;

% Context
H.objective        = z;   % native objective; see meta.objective_definition
H.elapsed_s        = z;   % cumulative wall time since the recorder was armed
H.move_active_frac = z;   % fraction of design variables at their move limit
H.omega1           = z;   % only where naturally available; never forced
H.mode_index       = z;   % tracked mode, where the method tracks one
H.mac              = z;   % only where the native method already computes it

% Transition markers: stage handoffs and continuation steps.  k_cont is derived
% from these, so they are recorded as events rather than inferred afterwards.
H.markers = struct('iter', {}, 'kind', {}, 'detail', {});
end
