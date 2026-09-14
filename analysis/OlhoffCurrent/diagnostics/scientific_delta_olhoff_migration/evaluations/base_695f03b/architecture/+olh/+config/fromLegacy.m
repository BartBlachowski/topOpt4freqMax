function cfg = fromLegacy(flat)
%FROMLEGACY  Map a legacy flat configuration onto the canonical schema.
%
%   The legacy form is the flat struct built by algo/defaultCfg.m, by the frozen
%   TMA_FROZEN_CFGS.mat, and by every audit runner.  This function is the ONLY
%   place that knows those field names.
%
%   It is lossless: olh.config.toLegacy(olh.config.fromLegacy(x)) reproduces x.
%   Anything it does not recognise is an error, never a silent drop.
%
%   Two legacy couplings are resolved here, and only here:
%
%     multRule=='subspace'   ->  multiplicity.diagonalOffsets = true
%          The legacy solver derived the diagonal-offset form of (25d) from the
%          multiplicity rule name.  Reproducing that derivation HERE keeps every
%          historical run bitwise while letting the two choices be set
%          independently in canonical form.
%
%     projection.on          ->  filter.type = 'density'
%          Turning projection on silently replaced the Sigmund sensitivity
%          filter with a density filter plus a chain rule.  That substitution
%          becomes visible as a filter type.

cfg  = olh.config.defaults();
seen = {};

    function put(path, value)
        cfg = olh.config.setPath(cfg, path, value);
    end
    function v = take(name, default)
        seen{end+1} = name;
        if isfield(flat, name) && ~isempty(flat.(name))
            v = flat.(name);
        elseif isfield(flat, name) && islogical(flat.(name))
            v = flat.(name);
        else
            v = default;
        end
    end
    function tf = has(name)
        seen{end+1} = name;
        tf = isfield(flat, name) && ~isempty(flat.(name));
    end

% ---- domain --------------------------------------------------------------
put('domain.a',         take('a', 8));
put('domain.b',         take('b', 1));
put('domain.thickness', take('t', 1));
put('domain.mesh.nelx', take('nelx', 160));
put('domain.mesh.nely', take('nely', 20));
bcMap = struct('a','simplySupported','b','clampedSimple','c','clamped');
bc = lower(take('bc','a'));
if ~isfield(bcMap, bc), error('olh:config:legacyBc','unknown legacy bc ''%s''', bc); end
put('domain.boundary.condition', bcMap.(bc));
supMap = struct('mid','midHeight','corner','corner','face','face');
sup = lower(take('support','mid'));
if ~isfield(supMap, sup), error('olh:config:legacySupport','unknown legacy support ''%s''', sup); end
put('domain.boundary.support', supMap.(sup));
axMap = struct('one','oneEnd','both','bothEnds');
ax = lower(take('axial','one'));
if ~isfield(axMap, ax), error('olh:config:legacyAxial','unknown legacy axial ''%s''', ax); end
put('domain.boundary.axialRestraint', axMap.(ax));
put('domain.element.type',       take('elemType','Q4'));
put('domain.element.massMatrix', lower(take('massType','consistent')));

% ---- solid ---------------------------------------------------------------
put('material.solid.E',       take('E', 1e7));
put('material.solid.nu',      take('nu', 0.3));
put('material.solid.density', take('rhom', 1));

% ---- stiffness -----------------------------------------------------------
put('material.stiffness.p', take('p', 3));
hasP = has('pSchedule');
put('material.stiffness.continuation.enabled', hasP);
if hasP
    put('material.stiffness.continuation.schedule', flat.pSchedule(:).');
    if isfield(flat,'pDecouple') && ~isempty(flat.pDecouple) && flat.pDecouple
        put('material.stiffness.continuation.driver','ownCounter');
    else
        put('material.stiffness.continuation.driver','moveLadderStage');
    end
end
seen{end+1} = 'pDecouple';
% blockStopUntilFinal was unconditional in the legacy solver whenever a
% schedule was present; it had no field of its own.
put('material.stiffness.continuation.blockStopUntilFinal', true);

% ---- mass ----------------------------------------------------------------
massMap = struct('lin','eq2','x4','eq4','x4a','eq4a','x4b','eq4b');
    function m = massName(v)
        key = lower(char(v));
        if ~isempty(key) && key(1) >= '0' && key(1) <= '9', key = ['x' key]; end
        if ~isfield(massMap, key)
            error('olh:config:legacyMassInterp','unknown legacy massInterp ''%s''', char(v));
        end
        m = massMap.(key);
    end
put('material.mass.model', massName(take('massInterp','4')));
hasML = has('massLowP');
put('material.mass.continuation.enabled', hasML);
if hasML
    put('material.mass.continuation.lowPModel', massName(flat.massLowP));
end

% ---- design --------------------------------------------------------------
put('design.initial',        take('rho0', 0.5));
put('design.minimum',        take('rhomin', 1e-3));
put('design.volumeFraction', take('volfrac', 0.5));

% ---- filtering and projection -------------------------------------------
projOn = isfield(flat,'projection') && ~isempty(flat.projection) ...
         && isfield(flat.projection,'on') && flat.projection.on;
seen{end+1} = 'projection';
fm = lower(take('filterMode','diag'));
switch fm
    case 'diag', put('filter.applyTo','diagonal');
    case 'all',  put('filter.applyTo','all');
    case 'none', put('filter.applyTo','all');   % scope is irrelevant; type carries it
    otherwise, error('olh:config:legacyFilterMode','unknown legacy filterMode ''%s''', fm);
end
if projOn
    put('filter.type','density');
elseif strcmp(fm,'none')
    put('filter.type','none');
else
    put('filter.type','sensitivity');
end
if isfield(flat,'rminPhys') && ~isempty(flat.rminPhys) && flat.rminPhys > 0
    put('filter.radiusPhysical', flat.rminPhys);
    put('filter.radiusElements', []);
else
    put('filter.radiusPhysical', []);
    put('filter.radiusElements', take('rminEl', 3.0));
end
seen{end+1} = 'rminPhys'; seen{end+1} = 'rminEl';

put('projection.enabled', projOn);
if projOn
    put('projection.eta',         flat.projection.eta);
    put('projection.beta.levels', flat.projection.betaSchedule(:).');
end

% ---- eigenproblem --------------------------------------------------------
put('eigen.targetMode',  take('n', 1));
put('eigen.maxCluster',  take('Nmax', 4));
put('eigen.solver',      lower(take('solver','eigs')));

% ---- multiplicity --------------------------------------------------------
mrMap = struct('binary','binary','latch','latch','hyst','hysteresis','subspace','subspace');
mr = lower(take('multRule','binary'));
if ~isfield(mrMap, mr), error('olh:config:legacyMultRule','unknown legacy multRule ''%s''', mr); end
put('multiplicity.method',        mrMap.(mr));
put('multiplicity.tolerance',     take('tolMult', 0.02));
put('multiplicity.enterTolerance',take('tolEnter', 0.01));
put('multiplicity.exitTolerance', take('tolExit', 0.05));
put('multiplicity.subspaceSize',  take('subN', 2));
% THE COUPLING, resolved exactly as the legacy solver derived it.
put('multiplicity.diagonalOffsets', strcmp(mr,'subspace'));
put('multiplicity.offDiagonal',     logical(take('offDiag', true)));

% ---- inner optimizer -----------------------------------------------------
put('optimizer.inner.type',     lower(take('innerSolver','mma')));
ivMap = struct('drho','increment','rho','design');
iv = lower(take('innerVar','drho'));
if ~isfield(ivMap, iv), error('olh:config:legacyInnerVar','unknown legacy innerVar ''%s''', iv); end
put('optimizer.inner.variable',      ivMap.(iv));
put('optimizer.inner.variant',       lower(take('mmaVariant','published')));
put('optimizer.inner.tolerance',     take('tolInner', 1e-2));
put('optimizer.inner.minIterations', take('minInner', 5));
put('optimizer.inner.maxIterations', take('maxInner', 300));

% ---- move ----------------------------------------------------------------
mfMap = struct('S0','fixed','S1','geometric','S2','ladder','S3','trustRatio');
mf = upper(take('moveFamily','S0'));
if ~isfield(mfMap, mf), error('olh:config:legacyMoveFamily','unknown legacy moveFamily ''%s''', mf); end
put('move.policy',  mfMap.(mf));
put('move.initial', take('move', 0.05));
put('move.minimum', take('moveMin', 0.002));
put('move.levels',  take('s2Levels', [0.05 0.02 0.01 0.005]));
put('move.geometric.ratio',            take('s1Gamma', 0.97));
put('move.geometric.afterCoalescence', logical(take('s1AfterCoal', true)));
put('move.trust.loRatio', take('s3Lo', 0.30));
put('move.trust.hiRatio', take('s3Hi', 0.70));
put('move.trust.shrink',  take('s3Down', 0.7));
put('move.trust.grow',    take('s3Up', 1.1));
sigMap = struct('beta','boundVariable','drms','designRms');
sig = lower(take('s2Signal','beta'));
if ~isfield(sigMap, sig), error('olh:config:legacyS2Signal','unknown legacy s2Signal ''%s''', sig); end
put('move.continuation.signal',    sigMap.(sig));
put('move.continuation.window',    take('s2Window', 10));
put('move.continuation.tolerance', take('s2Tol', 5e-3));

% ---- stopping ------------------------------------------------------------
put('stop.norm',      lower(take('outerNorm','l2')));
put('stop.tolerance', take('tolOuter', 1e-3));
% A legacy config carries a NUMBER, not a rule.  Preserve it verbatim: several
% audit runners computed it with the mesh-scaling law and several did not, and
% re-deriving it here could change a stored tolerance in its last bit.
put('stop.toleranceRule','explicit');
og = lower(take('outerGuard','none'));
switch og
    case 'none',        put('stop.guards.settledMove', false);
    case 'settledmove', put('stop.guards.settledMove', true);
    otherwise, error('olh:config:legacyOuterGuard','unknown legacy outerGuard ''%s''', og);
end
rg = upper(char(take('restorationGuard','')));
seen{end+1} = 'restorationGuard';
put('stop.guards.ladderExhausted', strcmp(rg,'R1'));
put('stop.guards.maxDesignChange', strcmp(rg,'R2'));
if ~isempty(rg) && ~any(strcmp(rg,{'R1','R2'}))
    error('olh:config:legacyRestorationGuard','unknown legacy restorationGuard ''%s''', rg);
end

% ---- runtime -------------------------------------------------------------
put('runtime.maxOuter',     take('maxOuter', 200));
put('runtime.singleThread', take('threads', 1) == 1);
put('runtime.diagnostics',  isfield(flat,'diag') && ~isempty(flat.diag) && logical(flat.diag));
put('runtime.verbose',      isfield(flat,'verbose') && ~isempty(flat.verbose) && logical(flat.verbose));
put('runtime.name',         char(take('name','')));
seen{end+1} = 'diag'; seen{end+1} = 'verbose'; seen{end+1} = 'threads';

% ---- 100% coverage check -------------------------------------------------
% Fields the legacy solver wrote into its own config are recorded, not mapped.
IGNORABLE = {'mmasubPath'};
unmapped = setdiff(fieldnames(flat), [seen(:); IGNORABLE(:)]);
if ~isempty(unmapped)
    error('olh:config:legacyUnmapped', ...
       ['Legacy field(s) with no canonical home:\n    %s\n' ...
        'fromLegacy refuses to drop a field silently.  Extend olh.config.schema ' ...
        'and this mapping.'], strjoin(unmapped(:).', sprintf('\n    ')));
end

cfg.provenance = struct('preset','(from legacy flat config)', ...
                        'overrides',{{}}, 'legacyFields',{fieldnames(flat)});
end
