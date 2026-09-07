function [cfg, warnings] = validate(cfg)
%VALIDATE  Strict validation of a canonical configuration.
%
%   Rejects unknown fields, unknown enum values, wrong types and shapes,
%   out-of-range values, and incompatible combinations.  Never coerces: a
%   configuration is either accepted as written or refused with a named error.
%
%   Warnings are reserved for configurations that are mathematically valid but
%   scientifically suspicious.
%
%   See also OLH.CONFIG.SCHEMA.

S = olh.config.schema();
warnings = {};

% ---- 1. unknown fields ---------------------------------------------------
known  = S(:,1);
actual = olh.config.paths(cfg);
actual = setdiff(actual, olh.config.paths(struct('provenance',struct('x',0))));
extra  = setdiff(actual, [known; {'provenance'}]);
extra  = extra(~startsWith(extra, 'provenance.'));
if ~isempty(extra)
    error('olh:config:unknownField', ...
        ['Unknown configuration field(s):\n    %s\n' ...
         'The canonical schema is olh.config.schema.  If this came from a ' ...
         'legacy flat config, pass it through olh.config.fromLegacy first.'], ...
        strjoin(extra, sprintf('\n    ')));
end

% ---- 2. per-field type, domain and presence ------------------------------
for i = 1:size(S,1)
    [path, kind, ~, dom] = deal(S{i,1}, S{i,2}, S{i,3}, S{i,4});
    [v, found] = olh.config.getPath(cfg, path);
    if ~found
        error('olh:config:missingField','Required field %s is absent.', path);
    end
    local_checkOne(path, kind, dom, v);
end

% ---- 3. incompatible combinations ---------------------------------------
g = @(p) olh.config.getPath(cfg, p);

if g('projection.enabled') && strcmp(g('filter.type'),'sensitivity')
    error('olh:config:projectionNeedsDensityFilter', ...
       ['projection.enabled=true with filter.type=''sensitivity'' is not implemented.\n' ...
        'Projection carries sensitivities to the design variable by the chain rule ' ...
        'through the DENSITY filter; the Sigmund sensitivity filter is then not applied ' ...
        'at all.  Set filter.type=''density'' and record that this departs from the ' ...
        'printed choice (sec. 1 filters the sensitivities).']);
end
if g('projection.enabled') && isempty(g('projection.beta.levels'))
    error('olh:config:projectionNeedsSchedule', ...
        'projection.enabled=true requires a non-empty projection.beta.levels.');
end
lev = g('projection.beta.levels');
if ~isempty(lev) && any(diff(lev(:)) < 0)
    error('olh:config:projectionScheduleNotMonotone', ...
        'projection.beta.levels must be monotone non-decreasing.');
end
if g('projection.enabled') && strcmp(g('optimizer.inner.type'),'lp')
    error('olh:config:projectionLpUnsupported', ...
        'Projection is implemented for the MMA inner loop only.');
end
if g('projection.enabled') && strcmp(g('optimizer.inner.variable'),'design')
    error('olh:config:projectionDesignVarUnsupported', ...
        'Projection is implemented for optimizer.inner.variable=''increment'' only.');
end

if g('material.stiffness.continuation.enabled')
    sch = g('material.stiffness.continuation.schedule');
    if isempty(sch)
        error('olh:config:pScheduleMissing', ...
            'material.stiffness.continuation.enabled=true requires a schedule.');
    end
    if any(diff(sch(:)) < 0)
        error('olh:config:pScheduleNotMonotone', ...
            'material.stiffness.continuation.schedule must be monotone non-decreasing.');
    end
    if strcmp(g('material.stiffness.continuation.driver'),'moveLadderStage') ...
            && ~strcmp(g('move.policy'),'ladder')
        error('olh:config:pDriverNeedsLadder', ...
           ['material.stiffness.continuation.driver=''moveLadderStage'' requires ' ...
            'move.policy=''ladder'': there is no ladder stage to index otherwise.']);
    end
    if strcmp(g('material.stiffness.continuation.driver'),'ownCounter') ...
            && ~strcmp(g('move.policy'),'ladder')
        error('olh:config:pDriverNeedsLadder', ...
           ['material.stiffness.continuation.driver=''ownCounter'' consumes the ' ...
            'ladder stall event and requires move.policy=''ladder''.']);
    end
end

if g('material.mass.continuation.enabled')
    if ~g('material.stiffness.continuation.enabled')
        error('olh:config:massContinuationNeedsP', ...
           ['material.mass.continuation.enabled=true has no meaning without a p ' ...
            'schedule: the low-p mass model is in force exactly while p is below ' ...
            'its final value.']);
    end
    if strcmp(g('material.mass.continuation.lowPModel'), g('material.mass.model'))
        error('olh:config:massContinuationIsNoOp', ...
            'material.mass.continuation.lowPModel equals material.mass.model; the schedule would be inert.');
    end
end

if g('stop.guards.ladderExhausted') && ~strcmp(g('move.policy'),'ladder')
    error('olh:config:ladderGuardNeedsLadder', ...
       ['stop.guards.ladderExhausted asks whether any REMAINING ladder level ' ...
        'exceeds the RMS tolerance, which requires move.policy=''ladder''.']);
end

if isempty(g('filter.radiusPhysical')) && isempty(g('filter.radiusElements')) ...
        && ~strcmp(g('filter.type'),'none')
    error('olh:config:noFilterRadius', ...
        'filter.type=''%s'' requires filter.radiusPhysical or filter.radiusElements.', g('filter.type'));
end
rp = g('filter.radiusPhysical');
if ~isempty(rp) && (~isscalar(rp) || ~isnumeric(rp) || rp <= 0)
    error('olh:config:badFilterRadius','filter.radiusPhysical must be empty or a positive scalar.');
end
re = g('filter.radiusElements');
if ~isempty(re) && (~isscalar(re) || ~isnumeric(re) || re <= 0)
    error('olh:config:badFilterRadius','filter.radiusElements must be empty or a positive scalar.');
end

if g('multiplicity.subspaceSize') > g('eigen.maxCluster')
    error('olh:config:subspaceTooLarge', ...
        'multiplicity.subspaceSize (%d) exceeds eigen.maxCluster (%d).', ...
        g('multiplicity.subspaceSize'), g('eigen.maxCluster'));
end
if strcmp(g('multiplicity.method'),'hysteresis') ...
        && g('multiplicity.exitTolerance') <= g('multiplicity.enterTolerance')
    error('olh:config:hysteresisInverted', ...
        'multiplicity.exitTolerance must exceed multiplicity.enterTolerance.');
end

if strcmp(g('move.policy'),'ladder') && any(diff(g('move.levels')) > 0)
    error('olh:config:ladderNotDescending','move.levels must be non-increasing.');
end

if strcmp(g('domain.boundary.support'),'midHeight') && mod(g('domain.mesh.nely'),2) ~= 0
    error('olh:config:midSupportNeedsEvenNely', ...
        'domain.boundary.support=''midHeight'' needs an even domain.mesh.nely (got %d).', ...
        g('domain.mesh.nely'));
end

if g('design.initial') < g('design.minimum')
    error('olh:config:initialBelowMinimum','design.initial is below design.minimum.');
end

% ---- 4. valid but scientifically suspicious ------------------------------
if strcmp(g('move.policy'),'ladder') && ~g('stop.guards.settledMove')
    warnings{end+1} = ['move.policy=''ladder'' without stop.guards.settledMove: ' ...
        'under a ladder ||drho||_inf <= mv_k, so on an iteration where the move ' ...
        'limit changes the measured step reports the SCHEDULE, not the design, ' ...
        'and the convergence test is uninterpretable there.'];
end
if g('multiplicity.tolerance') > 0.1
    warnings{end+1} = sprintf(['multiplicity.tolerance = %g. Sec. 3.5.1 calls for a ' ...
        '"predefined, very small tolerance"; this is large enough to merge ' ...
        'genuinely separated modes.'], g('multiplicity.tolerance'));
end
if strcmp(g('filter.type'),'density') && ~g('projection.enabled')
    warnings{end+1} = ['filter.type=''density'' with projection disabled: this is the ' ...
        'identity-projection control. It departs from the printed choice (sec. 1 ' ...
        'filters the SENSITIVITIES) without gaining the projection.'];
end
if strcmp(g('filter.type'),'none')
    warnings{end+1} = 'filter.type=''none'': checkerboarding and mesh dependence are unrestrained.';
end
if ~g('multiplicity.offDiagonal') && strcmp(g('optimizer.inner.type'),'mma')
    warnings{end+1} = ['multiplicity.offDiagonal=false imposes the Krog & Olhoff (1999) ' ...
        'equalities f_sk''drho=0. Sec. 3.5.3 presents that as an option, not as what was run.'];
end
if strcmp(g('multiplicity.method'),'subspace') && ~g('multiplicity.diagonalOffsets')
    warnings{end+1} = ['multiplicity.method=''subspace'' with diagonalOffsets=false applies ' ...
        '(25d) as printed - which assumes EXACT degeneracy - to a fixed window whose ' ...
        'eigenvalues are generally separated.'];
end
if g('material.stiffness.continuation.enabled') && ~g('material.stiffness.continuation.blockStopUntilFinal')
    warnings{end+1} = ['p continuation without blockStopUntilFinal: the run may be declared ' ...
        'converged at an intermediate p, i.e. for a different problem.'];
end
end

% =========================================================================
function local_checkOne(path, kind, dom, v)
switch kind
    case 'enum'
        if ~(ischar(v) || isstring(v)) || ~any(strcmp(char(v), dom))
            error('olh:config:badEnum','%s must be one of {%s}; got ''%s''.', ...
                path, strjoin(dom, ', '), local_show(v));
        end
    case 'logical'
        if ~(islogical(v) && isscalar(v))
            error('olh:config:badType','%s must be a logical scalar (true/false); got %s.', ...
                path, local_show(v));
        end
    case 'char'
        if ~(ischar(v) || isstring(v))
            error('olh:config:badType','%s must be text.', path);
        end
    case {'double','int'}
        if ~(isnumeric(v) && isscalar(v) && isreal(v) && ~isnan(v))
            error('olh:config:badType','%s must be a real numeric scalar; got %s.', path, local_show(v));
        end
        if strcmp(kind,'int') && mod(v,1) ~= 0
            error('olh:config:badType','%s must be an integer; got %g.', path, v);
        end
        if ~isempty(dom) && (v < dom(1) || v > dom(2))
            error('olh:config:outOfRange','%s must lie in [%g, %g]; got %g.', path, dom(1), dom(2), v);
        end
    case 'vector'
        if isempty(v), return; end
        if ~(isnumeric(v) && isvector(v) && isreal(v) && all(~isnan(v)))
            error('olh:config:badType','%s must be a real numeric vector or empty.', path);
        end
    case 'any'
        % checked by a dedicated rule in validate
    otherwise
        error('olh:config:badSchema','Unknown schema kind ''%s'' for %s.', kind, path);
end
end

function s = local_show(v)
if ischar(v) || isstring(v), s = char(v);
elseif isnumeric(v) && isscalar(v), s = num2str(v);
elseif islogical(v) && isscalar(v), s = mat2str(v);
else, s = sprintf('<%s %s>', class(v), mat2str(size(v)));
end
end
