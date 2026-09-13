function name = confbench_olhoff_preset()
%CONFBENCH_OLHOFF_PRESET  The ONE place the benchmark names its Olhoff preset.
%
%   name = CONFBENCH_OLHOFF_PRESET() returns the canonical OlhoffCurrent preset
%   the conference benchmark runs in its Du-Olhoff column:
%
%       duOlhoffPedersenAdaptiveBoxSensitivityFiltered
%
%   Pedersen (2000) low-density stiffness with linear mass, a per-element
%   adaptive move box and the natural design-change stop -- a DISTINCT
%   formulation from the historical SIMP + eq. (4b) preset
%   duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered that the 2026-09-11
%   campaign ran (display label "Du-Olhoff reconstruction (M4)" at the time).
%
%   The name is stated here explicitly rather than read from "whatever is
%   production", so that a benchmark can never change formulation silently.
%   CONFBENCH_PREFLIGHT refuses to run unless this name is ALSO the production
%   preset recorded in analysis/OlhoffCurrent/PROVENANCE.json, and asserts the
%   resolved formulation field by field for exactly this preset.
%
%   See also OLHOFFCURRENT_PRESETS, OLHOFFCURRENT_PRODUCTION_PRESET, CONFBENCH_PREFLIGHT.
name = 'duOlhoffPedersenAdaptiveBoxSensitivityFiltered';
end
