function T = list()
%LIST  The named presets, with their classification and provenance.
%
%   classification (Phase 4 of the refactor brief):
%     SCIENTIFIC_PRESET  a distinct mathematical formulation
%     EXPERIMENT_PRESET  a preregistered audit realization
%     RUNTIME_PRESET     differs only in runtime/test policy, not mathematics
%     OBSOLETE_ALIAS     kept so an old name still resolves
%
%   Every preset is a pure function cfg = olh.presets.<name>(cfg).  A preset
%   populates canonical fields and does nothing else: no mathematics, no run
%   state, no solver call.

T = {
% name                      classification        historical label(s)
'duOlhoffFrozenM4',        'SCIENTIFIC_PRESET',  'TMA / B0 / REG160 / "frozen M4" / the conference realization'
'duOlhoffMatureM4',        'EXPERIMENT_PRESET',  'Bmature / R2'
'restorationLadderGuard',  'EXPERIMENT_PRESET',  'R1'
'noDescentFixedMove',      'EXPERIMENT_PRESET',  'nodescent / S0'
'pContinuationCoupled',    'EXPERIMENT_PRESET',  'P1'
'pContinuationDecoupled',  'EXPERIMENT_PRESET',  'PD1'
'pMassCompatible',         'EXPERIMENT_PRESET',  'PM1'
'projectionIdentity',      'EXPERIMENT_PRESET',  'D160 (the beta=0 control)'
'projected',               'EXPERIMENT_PRESET',  'T160 / T240 / T320 / T800'
'legacyBinaryDiagonal',    'SCIENTIFIC_PRESET',  'the pre-M4 reconstruction; algo/defaultCfg.m style'
};
end
