function cfg = legacyBinaryDiagonal(cfg)
%LEGACYBINARYDIAGONAL  The pre-M4 reconstruction.
%
%   Historical label: the early reconstruction; the style of algo/defaultCfg.m.
%   Classification: SCIENTIFIC_PRESET -- a genuinely different formulation.
%
%   Differences from duOlhoffFrozenM4, all scientific:
%     multiplicity  the memoryless relative-difference classifier of sec. 3.5.1
%                   ("binary"), at tolerance 0.02, WITHOUT diagonal offsets --
%                   i.e. (25d) applied exactly as printed, which assumes the N
%                   eigenvalues are exactly equal
%     filtering     only the diagonal f_jj are filtered, not every f_sk
%     move          fixed at 0.05; no ladder, and therefore no settledMove guard
%
%   This preset exists because it is the OTHER end of the option space and
%   exercises code paths the frozen realization never reaches.  It is retained
%   as a scientific alternative, not as an obsolete alias.
if nargin < 1, cfg = olh.config.defaults(); end
cfg = olh.presets.duOlhoffFrozenM4(cfg);
cfg = olh.config.assign(cfg, ...
    'multiplicity.method',          'binary', ...
    'multiplicity.tolerance',       0.02, ...
    'multiplicity.diagonalOffsets', false, ...
    'filter.applyTo',               'diagonal', ...
    'move.policy',                  'fixed', ...
    'move.initial',                 0.05, ...
    'stop.guards.settledMove',      false);
end
