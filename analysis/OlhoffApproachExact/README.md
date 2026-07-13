# Archived diagnostic reconstruction attempt

`OlhoffApproachExact` is retained to preserve the reconstruction campaign and
its negative scientific result. It is not a canonical Du--Olhoff
implementation, a reference implementation, a benchmark implementation, or a
source of reviewer evidence.

The completed campaign could not establish a paper-faithful reconstruction of
Du and Olhoff (2007). Diagnostics ruled out the implemented FE formulation,
material interpolation, sensitivities, generalized gradients, multiplicity
handling, mode tracking, optimizer stabilization, persistent MMA, the tested
regularization variants, and the tested support interpretations as sufficient
explanations of the benchmark discrepancy. The remaining discrepancy is most
plausibly due to benchmark under-specification or unpublished implementation
details.

Accordingly:

- all MATLAB code and experiment reports in this directory are diagnostic
  archive material;
- historical uses of “exact,” “faithful,” or “paper-exact” record the intent at
  the time and are superseded by this final verdict;
- no result here may support quantitative frequency-gap, convergence, speedup,
  scaling, or optimality claims;
- no production revision experiment may import this directory;
- the only local Olhoff comparison implementation allowed in active revision
  work is `analysis/OlhoffApproach`.

The archive remains useful for documenting what was tested and eliminated. It
must not be presented as a successful reproduction.
