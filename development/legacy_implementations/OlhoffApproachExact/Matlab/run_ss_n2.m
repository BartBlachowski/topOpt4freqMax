% RUN_SS_N2  Section 3.2 / Fig. 6a: simply supported beam, max omega_2 -> 598.3 (bimodal)
%
%   Olhoff & Du (2014), "Structural Topology Optimization with Respect to
%   Eigenfrequencies of Vibration", CISM 2014.
%
%   Usage:  res = run_ss_n2;                     % defaults from olhoff2014_case
%           res = run_ss_n2(struct('nelx',240,'nely',30));
%
%   See run_olhoff_case for the report format and the declared decision rule,
%   and PLAN_Olhoff2014_exact.md for the [E]/[D]/[R] exactness contract.

function res = run_ss_n2(overrides)
if nargin < 1, overrides = struct(); end
res = run_olhoff_case('ss_n2', overrides);
end
