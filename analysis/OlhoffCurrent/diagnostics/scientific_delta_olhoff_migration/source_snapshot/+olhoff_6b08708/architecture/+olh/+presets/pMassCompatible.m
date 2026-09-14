function cfg = pMassCompatible(cfg)
%PMASSCOMPATIBLE  Decoupled p continuation with the printed low-p mass model.
%
%   Historical label: PM1 (audit_pm1_printed_mass_continuation).
%   Classification: EXPERIMENT_PRESET.
%   Parent: pContinuationDecoupled.  Departure: mass continuation.
%
%   While p is below its final value the PRINTED linear model eq. (2) is in
%   force; the terminal model eq. (4b) resumes exactly when p first reaches its
%   final value.  The switch introduces no threshold of its own -- it is tied to
%   the existing p schedule -- and the final analysis always uses eq. (4b), so
%   the terminal formulation matches the frozen baseline.
%
%   Rationale: eq. (4)'s low-density cut-off exists to suppress spurious
%   localized modes that arise when the stiffness/mass ratio collapses at p=3,
%   q=1 (sec. 2.2).  At p=1 that ratio is not small and the cut-off is arguably
%   unwarranted.  Both models are class A; SCHEDULING BETWEEN THEM IS CLASS D.
if nargin < 1, cfg = olh.config.defaults(); end
cfg = olh.presets.pContinuationDecoupled(cfg);
cfg = olh.config.assign(cfg, ...
    'material.mass.continuation.enabled',  true, ...
    'material.mass.continuation.lowPModel','eq2');
end
