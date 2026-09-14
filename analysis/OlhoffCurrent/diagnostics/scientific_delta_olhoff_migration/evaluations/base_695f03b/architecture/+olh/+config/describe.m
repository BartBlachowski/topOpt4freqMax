function s = describe(cfg)
%DESCRIBE  Answer "what mathematical formulation am I running?" from cfg alone.
%
%   olh.config.describe(cfg) prints the formulation in scientific terms, with
%   the provenance class of every choice.  No audit name, runner name or
%   historical patch is consulted -- everything below is read from the
%   configuration object.
%
%   With an output argument the text is returned instead of printed.

g = @(p) olh.config.getPath(cfg, p);
S = olh.config.schema();
cls = @(p) S{find(strcmp(p,S(:,1)),1), 5};

L = {};
add = @(varargin) 0;  %#ok<NASGU>
    function push(fmt, varargin)
        L{end+1} = sprintf(fmt, varargin{:}); %#ok<AGROW>
    end

push('DU-OLHOFF EIGENFREQUENCY MAXIMIZATION -- effective formulation');
push('%s', repmat('=',1,72));
if isfield(cfg,'provenance')
    push('preset            %s', cfg.provenance.preset);
    if ~isempty(cfg.provenance.overrides)
        ov = cfg.provenance.overrides;
        for k = 1:2:numel(ov)
            push('  override        %s = %s', ov{k}, local_show(ov{k+1}));
        end
    end
end
push('');
push('PROBLEM');
push('  maximize          omega_%d, bound formulation (25a-f)', g('eigen.targetMode'));
push('  domain            %g x %g x %g, %d x %d elements  [%s]', ...
     g('domain.a'), g('domain.b'), g('domain.thickness'), ...
     g('domain.mesh.nelx'), g('domain.mesh.nely'), cls('domain.mesh.nelx'));
push('  supports          %s, %s, axial restraint at %s  [%s]', ...
     g('domain.boundary.condition'), g('domain.boundary.support'), ...
     g('domain.boundary.axialRestraint'), cls('domain.boundary.support'));
push('  volume fraction   %g  [%s]', g('design.volumeFraction'), cls('design.volumeFraction'));
push('');
push('MATERIAL INTERPOLATION');
if g('material.stiffness.continuation.enabled')
    sch = g('material.stiffness.continuation.schedule');
    push('  stiffness         SIMP rho^p, p CONTINUATION %s, driver %s  [%s]', ...
         mat2str(sch), g('material.stiffness.continuation.driver'), ...
         cls('material.stiffness.continuation.enabled'));
    if g('material.stiffness.continuation.blockStopUntilFinal')
        push('                    convergence blocked until p reaches %g', sch(end));
    end
else
    push('  stiffness         SIMP rho^p, p = %g FIXED  [%s]', ...
         g('material.stiffness.p'), cls('material.stiffness.p'));
end
massName = struct('eq2','eq. (2) linear','eq4','eq. (4) discontinuous at the cut-off', ...
                  'eq4a','eq. (4a) C0','eq4b','eq. (4b) C1');
push('  mass              %s  [%s]', massName.(g('material.mass.model')), cls('material.mass.model'));
push('                    q = %g, r = %g, cut-off %g', ...
     g('material.mass.q'), g('material.mass.lowDensityExponent'), g('material.mass.cutoff'));
if g('material.mass.continuation.enabled')
    push('  mass continuation %s while p is below its final value  [%s]', ...
         massName.(g('material.mass.continuation.lowPModel')), cls('material.mass.continuation.enabled'));
end
push('');
push('REGULARIZATION');
switch g('filter.type')
    case 'sensitivity'
        push('  filter            Sigmund (1997) SENSITIVITY filter  [%s]  <- the printed choice', cls('filter.type'));
        push('                    applied to %s of the generalized gradients  [%s]', ...
             g('filter.applyTo'), cls('filter.applyTo'));
    case 'density'
        push('  filter            DENSITY filter  [D]  <- DEPARTS from the printed choice');
        push('                    sec. 1 states the filter was applied to the SENSITIVITIES');
    case 'none'
        push('  filter            NONE  <- checkerboarding unrestrained');
end
if isempty(g('filter.radiusPhysical'))
    push('  radius            %g elements  [%s]', g('filter.radiusElements'), cls('filter.radiusElements'));
else
    R = g('filter.radiusPhysical');
    push('  radius            %g physical = %g elements  [%s]', ...
         R, R/(g('domain.b')/g('domain.mesh.nely')), cls('filter.radiusPhysical'));
end
if g('projection.enabled')
    push('  projection        tanh Heaviside, eta = %g, beta %s  [D]', ...
         g('projection.eta'), mat2str(g('projection.beta.levels')));
    push('                    advancing on %s', g('projection.continuation.trigger'));
    push('                    *** ABSENT FROM EVERY DU-OLHOFF SOURCE ***');
else
    push('  projection        disabled');
end
push('');
push('FIELDS');
if g('projection.enabled')
    push('  design variable   z');
    push('  filtered density  zTilde = (H z)/Hs');
    push('  physical density  rho = rhomin + (1-rhomin)*P(zTilde)   <- enters FE, mass, volume');
else
    push('  design variable   rho, and it IS the physical density');
end
push('');
push('MULTIPLICITY');
switch g('multiplicity.method')
    case 'subspace'
        push('  detection         NONE: N fixed at %d  [%s]', g('multiplicity.subspaceSize'), cls('multiplicity.method'));
    case 'binary'
        push('  detection         memoryless relative-difference test, tol %g  [A measure / C value]', g('multiplicity.tolerance'));
    otherwise
        push('  detection         %s, tol %g  [%s]', g('multiplicity.method'), ...
             g('multiplicity.tolerance'), cls('multiplicity.method'));
end
if g('multiplicity.diagonalOffsets')
    push('  (25d) form        diagonal offsets diag(lambda_j - lambda_n) RETAINED  [C]');
else
    push('  (25d) form        AS PRINTED, which assumes exact degeneracy  [A]');
end
push('  off-diagonals     %s  [%s]', local_tf(g('multiplicity.offDiagonal'), ...
     'retained, full determinant', 'forced to vanish (Krog & Olhoff route)'), cls('multiplicity.offDiagonal'));
push('');
push('OPTIMIZER');
push('  inner             %s, %s variant, on the %s  [%s]', upper(g('optimizer.inner.type')), ...
     g('optimizer.inner.variant'), g('optimizer.inner.variable'), cls('optimizer.inner.type'));
push('  inner exit        rel. step < %g, at least %d, at most %d  [%s]', ...
     g('optimizer.inner.tolerance'), g('optimizer.inner.minIterations'), ...
     g('optimizer.inner.maxIterations'), cls('optimizer.inner.tolerance'));
push('');
push('MOVE LIMIT   [the paper bounds drho ONLY by the box (25f)]');
switch g('move.policy')
    case 'fixed',      push('  policy            FIXED at %g  [%s]', g('move.initial'), cls('move.policy'));
    case 'ladder',     push('  policy            LADDER %s  [%s]', mat2str(g('move.levels')), cls('move.policy'));
                       push('  descends when     %s stalls over a window of %d, rel. progress < %g  [%s]', ...
                            g('move.continuation.signal'), g('move.continuation.window'), ...
                            g('move.continuation.tolerance'), cls('move.continuation.signal'));
    case 'geometric',  push('  policy            GEOMETRIC ratio %g, floor %g  [%s]', ...
                            g('move.geometric.ratio'), g('move.minimum'), cls('move.policy'));
    case 'trustRatio', push('  policy            TRUST RATIO band [%g, %g]  [D]', ...
                            g('move.trust.loRatio'), g('move.trust.hiRatio'));
end
push('');
push('STOPPING');
push('  metric            %s norm of the %s increment  [%s]', ...
     upper(g('stop.norm')), g('stop.field'), cls('stop.norm'));
push('  threshold         eps = %.6g  (%s)  [%s]', g('stop.tolerance'), ...
     g('stop.toleranceRule'), cls('stop.tolerance'));
gu = {};
if g('stop.guards.settledMove'),     gu{end+1} = 'settledMove'; end
if g('stop.guards.ladderExhausted'), gu{end+1} = 'ladderExhausted'; end
if g('stop.guards.maxDesignChange'), gu{end+1} = 'maxDesignChange'; end
if isempty(gu), push('  guards            none');
else,           push('  guards            %s', strjoin(gu,', ')); end
push('  iteration cap     %d  (reaching it is CAP_HIT, not convergence)', g('runtime.maxOuter'));
push('%s', repmat('=',1,72));
push('Provenance classes: A specified by a Du-Olhoff source | B implied by one |');
push('C under-specified reconstruction choice | D later experimental modification.');

txt = strjoin(L, newline);
if nargout > 0, s = txt; else, fprintf('%s\n', txt); end
end

function s = local_tf(tf, a, b)
if tf, s = a; else, s = b; end
end
function s = local_show(v)
if ischar(v), s = ['''' v ''''];
elseif isnumeric(v) && isscalar(v), s = num2str(v);
elseif islogical(v) && isscalar(v), s = mat2str(v);
elseif isnumeric(v), s = mat2str(v);
else, s = class(v);
end
end
