% RUN_CC_GAP23  Section 3.3 / Fig. 7: clamped-clamped beam with m_c = m_b/2 at the mid-point of the lower edge, max (omega_3 - omega_2) -> 810 (+548 %), omega_3 omega_4 omega_5 trimodal
%
%   Olhoff & Du (2014), "Structural Topology Optimization with Respect to
%   Eigenfrequencies of Vibration", CISM 2014.
%
%   Usage:  res = run_cc_gap23;                     % defaults from olhoff2014_case
%           res = run_cc_gap23(struct('nelx',240,'nely',30));
%
%   See run_olhoff_case for the report format and the declared decision rule,
%   and PLAN_Olhoff2014_exact.md for the [E]/[D]/[R] exactness contract.

function res = run_cc_gap23(overrides)
if nargin < 1, overrides = struct(); end
res = run_olhoff_case('cc_gap23', overrides);
end
