function S = schema()
%SCHEMA  The canonical configuration schema: THE single source of defaults.
%
%   S is an n x 6 cell array, one row per configurable quantity:
%
%       {path, kind, default, domain, provenanceClass, doc}
%
%   path    dotted path into the canonical cfg struct
%   kind    'enum' | 'double' | 'int' | 'logical' | 'char' | 'vector' | 'any'
%   default the value olh.config.defaults() installs
%   domain  enum   -> cellstr of admissible values
%           double -> [lo hi] inclusive bounds ([] = unbounded)
%           int    -> [lo hi]
%           vector -> [minLen maxLen] (Inf allowed); [] = any length
%           other  -> []
%   class   'A' explicitly specified by a Du-Olhoff source
%           'B' directly implied / reconstructable from a source
%           'C' under-specified reconstruction choice
%           'D' later experimental modification or new method
%           See architecture/docs/SCIENTIFIC_CONFIG_PROVENANCE.md for the
%           evidence behind every letter.
%   doc     one line, in terms of the science, never of an experiment label
%
%   DEFAULTS POLICY (Phase 15).  The values below are the frozen conference
%   realization's scientific choices at 160x20.  There is deliberately no
%   second set of defaults anywhere: algo/defaultCfg.m is SUPERSEDED and differs
%   from this table in eleven scientific fields.  Presets and explicit overrides
%   are the only ways to depart from this table.

S = {
% ---- domain -------------------------------------------------------------
'domain.a',                     'double', 8,      [0 Inf],  'A', 'beam length'
'domain.b',                     'double', 1,      [0 Inf],  'A', 'beam height'
'domain.thickness',             'double', 1,      [0 Inf],  'A', 'out-of-plane thickness'
'domain.mesh.nelx',             'int',    160,    [1 Inf],  'C', 'elements along the length; NE is never reported in the paper'
'domain.mesh.nely',             'int',    20,     [1 Inf],  'C', 'elements through the height'
'domain.boundary.condition',    'enum',   'simplySupported', {'simplySupported','clampedSimple','clamped'}, 'A', 'Fig. 2(a-c) support cases'
'domain.boundary.support',      'enum',   'midHeight',       {'midHeight','corner','face'},                 'C', 'idealization of a SIMPLE support; the paper draws corner, its numbers fit midHeight'
'domain.boundary.axialRestraint','enum',  'bothEnds',        {'oneEnd','bothEnds'},                          'C', 'which ends carry ux restraint'
'domain.element.type',          'enum',   'Q4',   {'Q4','Q6'},                    'C', 'the paper says only "plane stress elements"'
'domain.element.massMatrix',    'enum',   'consistent', {'consistent','lumped'},  'C', 'element mass matrix form'
% ---- solid material -----------------------------------------------------
'material.solid.E',             'double', 1e7,    [0 Inf],  'A', 'Young modulus'
'material.solid.nu',            'double', 0.3,    [-1 0.5], 'A', 'Poisson ratio'
'material.solid.density',       'double', 1,      [0 Inf],  'A', 'solid mass density rho_m'
% ---- stiffness interpolation -------------------------------------------
'material.stiffness.model',     'enum',   'simp', {'simp','pedersen'},  'A', 'eq. (1) SIMP rho^p; pedersen = Pedersen (2000) eq. (5), rho^p above the threshold and linear rho*rho0^(p-1) below it, named in sec. 2.2 as the alternative to the mass cut-off'
'material.stiffness.linearBelow','double', 0.1,   [0 1],     'A', 'pedersen: threshold rho0 below which the stiffness is linear; Pedersen (2000) prints 0.1 ("one hundredth of the penalization of the mass")'
'material.stiffness.p',         'double', 3,      [1 Inf],   'A', 'penalization power of eq. (1)'
'material.stiffness.continuation.enabled',  'logical', false, [], 'A', 'sec. 2.1: p is "normally assigned values increasing from 1 to 3"'
'material.stiffness.continuation.schedule', 'vector', [],  [1 Inf], 'B', 'successive values of p; endpoints 1 and 3 are printed, the schedule is not'
'material.stiffness.continuation.driver',   'enum', 'moveLadderStage', {'moveLadderStage','ownCounter'}, 'C', 'what advances p: the move ladder stage, or p own counter consuming the same stall event'
'material.stiffness.continuation.blockStopUntilFinal', 'logical', true, [], 'B', 'a p=3 problem may not be declared converged while p<3'
% ---- mass interpolation -------------------------------------------------
'material.mass.model',          'enum',   'eq4b', {'eq2','eq4','eq4a','eq4b'}, 'A', 'printed mass interpolation: (2) linear, (4) discontinuous, (4a) C0, (4b) C1'
'material.mass.q',              'double', 1,      [1 Inf],  'A', 'eq. (2) exponent; "normally, q=1 is chosen"'
'material.mass.lowDensityExponent','double',6,    [1 Inf],  'A', 'eq. (4) exponent r; "r is chosen to be about r=6"'
'material.mass.cutoff',         'double', 0.1,    [0 1],    'A', 'density below which the low-density branch applies'
'material.mass.continuation.enabled',  'logical', false, [], 'D', 'switch mass model while p is below its final value'
'material.mass.continuation.lowPModel','enum', 'eq2', {'eq2','eq4','eq4a','eq4b'}, 'D', 'model in force during the low-p phase'
% ---- design field -------------------------------------------------------
'design.initial',               'double', 0.5,    [0 1],    'A', 'uniform initial density'
'design.minimum',               'double', 1e-3,   [0 1],    'A', 'rho_min of the box (7e)/(25f)'
'design.volumeFraction',        'double', 0.5,    [0 1],    'A', 'alpha of the volume constraint (25e)'
% ---- filtering ----------------------------------------------------------
'filter.type',                  'enum',   'sensitivity', {'sensitivity','density','none'}, 'A', 'sec. 1: Sigmund (1997) filter "applied to the sensitivities"; density filtering is NOT published'
'filter.radiusPhysical',        'any',    0.06,   [],       'C', 'filter radius in physical units; never stated in the paper. Overrides radiusElements when non-empty'
'filter.radiusElements',        'any',    [],     [],       'C', 'filter radius in element units'
'filter.applyTo',               'enum',   'all',  {'diagonal','all'}, 'C', 'sensitivity filtering: only f_jj, or every f_sk. The paper has one sensitivity vector and does not say'
% ---- projection ---------------------------------------------------------
'projection.enabled',           'logical', false, [],       'D', 'tanh Heaviside projection; absent from every source'
'projection.eta',               'double', 0.5,    [0 1],    'D', 'projection threshold'
'projection.beta.levels',       'vector', [],     [1 Inf],  'D', 'monotone non-decreasing projection sharpness levels'
'projection.continuation.trigger','enum', 'outerConvergence', {'outerConvergence'}, 'D', 'what advances the sharpness level'
% ---- eigenproblem -------------------------------------------------------
'eigen.targetMode',             'int',    1,      [1 Inf],  'A', 'n: which eigenfrequency is maximized'
'eigen.maxCluster',             'int',    4,      [1 Inf],  'C', 'Nmax; J = n + Nmax modes are extracted'
'eigen.solver',                 'enum',   'eigs', {'eigs','dense'}, 'C', 'generalized eigensolver'
'eigen.tolerance',              'double', 1e-12,  [0 Inf],  'C', 'eigs convergence tolerance'
'eigen.maxIterations',          'int',    5000,   [1 Inf],  'C', 'eigs iteration cap'
'eigen.krylovFactor',           'int',    4,      [1 Inf],  'C', 'eigs subspace size factor: p = max(20, factor*J)'
% ---- multiplicity -------------------------------------------------------
'multiplicity.method',          'enum',   'subspace', {'binary','latch','hysteresis','subspace'}, 'C', 'how the multiplicity N of omega_n is decided each outer iteration'
'multiplicity.tolerance',       'double', 0.05,   [0 1],    'A', 'sec. 3.5.1 measure: relative frequency difference. The VALUE is never given'
'multiplicity.enterTolerance',  'double', 0.01,   [0 1],    'C', 'hysteresis entry threshold'
'multiplicity.exitTolerance',   'double', 0.05,   [0 1],    'C', 'hysteresis exit threshold'
'multiplicity.subspaceSize',    'int',    2,      [1 Inf],  'C', 'fixed cluster size when method is subspace'
'multiplicity.diagonalOffsets', 'logical', true,  [],       'C', 'retain diag(lambda_j - lambda_n) in (25d). Printed (25d) assumes EXACT degeneracy; this is reconstruction'
'multiplicity.offDiagonal',     'logical', true,  [],       'A', 'true = full (25d) determinant; false = force f_sk''drho=0, the Krog & Olhoff LP route'
% ---- inner optimizer ----------------------------------------------------
'optimizer.inner.type',         'enum',   'mma',  {'mma','lp'}, 'A', 'sec. 3.5.3: "the MMA method (Svanberg 1987) has been used"'
'optimizer.inner.variable',     'enum',   'increment', {'increment','design'}, 'C', 'increment: MMA state reset each outer iteration; design: asymptotes persist'
'optimizer.inner.variant',      'enum',   'published', {'published','asfound'}, 'B', 'published = Svanberg Sept-2007 constants; asfound = local lineage copy'
'optimizer.inner.asymptoteHistory','enum','inner', {'inner','outer'}, 'C', 'with variable=design: which history adapts the MMA asymptotes -- the inner sub-iterate sequence, or the OUTER design sequence rho_k, rho_k-1, rho_k-2 with the asymptotes held during the inner loop (Svanberg 1987 usage across the nested scheme; the paper is silent)'
'optimizer.inner.tolerance',    'double', 0.05,   [0 Inf],  'C', 'relative inner step test; Fig. 1 gives no criterion'
'optimizer.inner.minIterations','int',    5,      [0 Inf],  'C', 'sub-iterates always taken'
'optimizer.inner.maxIterations','int',    500,    [1 Inf],  'C', 'inner iteration cap'
% ---- move limit ---------------------------------------------------------
'move.policy',                  'enum',   'ladder', {'fixed','geometric','ladder','trustRatio','adaptive'}, 'C', 'the paper places NO bound on drho other than the box (25f); adaptive = per-element box contracted/expanded by Svanberg''s asymptote rule on the OUTER design history'
'move.initial',                 'double', 0.04,   [0 Inf],  'C', 'move limit, and the starting value for geometric/trustRatio; Inf = no move box, the design is bounded by (25f) alone'
'move.minimum',                 'double', 0.002,  [0 1],    'C', 'floor for geometric/trustRatio'
'move.levels',                  'vector', [0.04 0.02 0.01 0.005], [1 Inf], 'C', 'descending ladder levels'
'move.geometric.ratio',         'double', 0.97,   [0 1],    'C', 'geometric contraction ratio'
'move.adaptive.grow',           'double', 1.2,    [1 Inf],  'C', 'adaptive: box growth factor for an element moving monotonically over the last two outer steps (Svanberg asyincr)'
'move.adaptive.shrink',         'double', 0.7,    [0 1],    'C', 'adaptive: box contraction factor for an element that reversed direction (Svanberg asydecr); floor = move.minimum, ceiling = move.initial'
'move.geometric.afterCoalescence','logical', true, [],      'C', 'start contracting only once N>=2 is first seen'
'move.trust.loRatio',           'double', 0.30,   [0 Inf],  'D', 'shrink below this realized/predicted gain ratio'
'move.trust.hiRatio',           'double', 0.70,   [0 Inf],  'D', 'grow above this ratio'
'move.trust.shrink',            'double', 0.7,    [0 1],    'D', 'contraction factor'
'move.trust.grow',              'double', 1.1,    [1 Inf],  'D', 'expansion factor'
'move.continuation.signal',     'enum',   'boundVariable', {'boundVariable','designRms','stageExhaustion'}, 'C', 'what advances the ladder: the bound variable beta of (25a), ||drho||/sqrt(NE), or the frozen two-branch stage-exhaustion rule E = A OR B'
'move.continuation.window',     'int',    10,     [1 Inf],  'C', 'stall detector window W'
'move.continuation.tolerance',  'double', 5e-3,   [0 Inf],  'C', 'relative-progress threshold below which a stall is declared'
% ---- stopping -----------------------------------------------------------
'stop.rule',                    'enum',   'designChange', {'designChange','stageExhaustion'}, 'C', 'what admits outer convergence: the sec. 3.5.1 design-increment test with its guards, or the frozen two-branch exhaustion rule at the last move level'
'stop.norm',                    'enum',   'l2',   {'l2','max'}, 'B', 'sec. 3.5.1 writes "the norm" unqualified; l2 is the natural reading'
'stop.tolerance',               'double', 0.05,   [0 Inf],  'C', 'epsilon of Fig. 1; never given in the paper'
'stop.toleranceRule',           'enum',   'meshScaled', {'explicit','meshScaled'}, 'C', 'meshScaled recomputes stop.tolerance as 0.05*sqrt(NE/3200) AFTER mesh overrides, so eps means the same RMS density change at every resolution'
'stop.field',                   'enum',   'designVariable', {'designVariable'}, 'A', 'sec. 3.5.1 monitors the DESIGN increment. Under projection that is dz, not d(rho_phys)'
'stop.guards.settledMove',      'logical', true,  [],       'C', 'assert convergence only when the move limit is unchanged from the previous iteration'
'stop.guards.settledWindow',    'int',    1,      [1 Inf],  'C', 'with settledMove: number of consecutive iterations the move limit must have been unchanged (1 = the previous iteration only)'
'stop.guards.boxInactiveFraction','double', 0,    [0 Inf],  'C', 'assert convergence only when max|drho| <= fraction * move limit, i.e. the step is small because the design stopped and not because the move box bound it; 0 = off'
'stop.guards.ladderExhausted',  'logical', false, [],       'D', 'assert convergence only when no remaining ladder level exceeds epsilon/sqrt(NE)'
'stop.guards.maxDesignChange',  'logical', false, [],       'D', 'assert convergence only when max|d(design)| < epsilon/sqrt(NE)'
% ---- runtime ------------------------------------------------------------
'runtime.maxOuter',             'int',    400,    [1 Inf],  'C', 'outer iteration cap; reaching it is CAP_HIT, not convergence'
'runtime.singleThread',         'logical', true,  [],       'C', 'required for meaningful complexity measurement'
'runtime.diagnostics',          'logical', false, [],       'C', 'per-iteration diagnostic record; provably inert'
'runtime.verbose',              'logical', false, [],       'C', 'per-iteration console table'
'runtime.name',                 'char',   '',     [],       'C', 'free-text run label; never read by solver mathematics'
};
end
