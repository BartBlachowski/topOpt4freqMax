% RUN_CC_N1  Section 3.1 / Fig. 3c: clamped-clamped beam, max omega_1 -> 456.4 (bimodal)
%
%   Olhoff & Du (2014), "Structural Topology Optimization with Respect to
%   Eigenfrequencies of Vibration", CISM 2014.
%
%   Usage:  res = run_cc_n1;                     % defaults from olhoff2014_case
%           res = run_cc_n1(struct('nelx',240,'nely',30));
%
%   See run_olhoff_case for the report format and the declared decision rule,
%   and PLAN_Olhoff2014_exact.md for the [E]/[D]/[R] exactness contract.

function res = run_cc_n1(overrides)
if nargin < 1, overrides = struct(); end
res = run_olhoff_case('cc_n1', overrides);
end
