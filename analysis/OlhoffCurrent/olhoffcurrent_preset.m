function info = olhoffcurrent_preset()
%OLHOFFCURRENT_PRESET  THE production preset of the Olhoff implementation.
%
%   There is exactly one, it has a name, and production scripts name it rather
%   than reassembling it from a handful of switches.
%
%       duOlhoffFixedPenaltySensitivityFiltered
%
%   The name is descriptive of the MATHEMATICS, not of our audit history.  The
%   two axes in it are the two that distinguish this realization from every
%   other preset in the family:
%
%     FixedPenalty          SIMP p = 3 held CONSTANT, no p continuation.
%                           (Du & Olhoff sec. 2.1 says p is "normally assigned
%                           values increasing from 1 to 3"; fixing it is a
%                           RECONSTRUCTION RULING, made because the reported
%                           initial eigenfrequencies fit p = 3 and not p = 1.)
%     SensitivityFiltered   Sigmund (1997) SENSITIVITY filter, applied to every
%                           f_sk rather than the diagonal only, at a fixed
%                           PHYSICAL radius R = 0.06*b.  No density filter, no
%                           Heaviside projection.
%
%   Full formulation, in scientific terms and with the provenance class of every
%   choice, is printed by
%
%       olh.config.describe(olhoffcurrent_config(nelx, nely))
%
%   and needs no knowledge of this project's audit history to read.
%
%   HISTORICAL ALIASES -- PROVENANCE ONLY, NEVER API
%   ------------------------------------------------
%   The same realization appears in the historical record under the audit codes
%   M4, TMA, B0 and REG160, and upstream as the preset name duOlhoffFrozenM4.
%   Those are provenance aliases and experiment identifiers.  They are recorded
%   so old evidence can be matched to new runs; they are NOT canonical
%   user-facing terminology and no production script should use them.
%
%   NO SEPARATE COPY OF THE MATHEMATICS
%   -----------------------------------
%   This preset does not restate the ~30 fields it needs.  It DELEGATES to the
%   promoted upstream preset olh.presets.duOlhoffFrozenM4, so the production
%   realization cannot silently drift from the accepted canonical one: there is
%   only one definition, and it is the promoted one.
%
%   See also OLHOFFCURRENT_CONFIG, OLHOFFCURRENT_RUN.

info = struct( ...
    'name',            'duOlhoffFixedPenaltySensitivityFiltered', ...
    'upstreamPreset',  'duOlhoffFrozenM4', ...
    'classification',  'SCIENTIFIC_PRESET', ...
    'label',           'Du-Olhoff reconstruction, fixed penalty, sensitivity filtered', ...
    'mustNotBeLabelled', 'Olhoff 2007', ...
    'epistemicClass',  ['reconstruction (class C): internally coherent, ' ...
                        'not a claimed historical implementation'], ...
    'historicalAliases', {{'M4', 'TMA', 'B0', 'REG160', 'duOlhoffFrozenM4'}});
end
