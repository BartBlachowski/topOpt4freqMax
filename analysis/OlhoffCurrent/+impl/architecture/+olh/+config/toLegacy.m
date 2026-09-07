function flat = toLegacy(cfg)
%TOLEGACY  Render a canonical configuration in the legacy flat form.
%
%   Inverse of olh.config.fromLegacy.  This is the compatibility bridge: the
%   numerical kernels still read the flat names, so nothing about their
%   arithmetic changes while the configuration model is replaced above them.
%
%   Only fields the legacy solver actually consults are emitted, and optional
%   fields are emitted ONLY when active, because the legacy code tests them with
%   isfield/isempty and an inert-but-present field is not always inert.

g = @(p) olh.config.getPath(cfg, p);
flat = struct();

% ---- domain --------------------------------------------------------------
flat.a = g('domain.a');  flat.b = g('domain.b');  flat.t = g('domain.thickness');
flat.nelx = g('domain.mesh.nelx');  flat.nely = g('domain.mesh.nely');
flat.bc      = local_rev(g('domain.boundary.condition'), ...
                  {'simplySupported','clampedSimple','clamped'}, {'a','b','c'});
flat.support = local_rev(g('domain.boundary.support'), ...
                  {'midHeight','corner','face'}, {'mid','corner','face'});
flat.axial   = local_rev(g('domain.boundary.axialRestraint'), ...
                  {'oneEnd','bothEnds'}, {'one','both'});
flat.elemType = g('domain.element.type');
flat.massType = g('domain.element.massMatrix');

% ---- solid ---------------------------------------------------------------
flat.E = g('material.solid.E');  flat.nu = g('material.solid.nu');
flat.rhom = g('material.solid.density');

% ---- SIMP ----------------------------------------------------------------
flat.p = g('material.stiffness.p');
flat.massInterp = local_rev(g('material.mass.model'), ...
                  {'eq2','eq4','eq4a','eq4b'}, {'lin','4','4a','4b'});
flat.rhomin  = g('design.minimum');
flat.rho0    = g('design.initial');
flat.volfrac = g('design.volumeFraction');

% ---- problem -------------------------------------------------------------
flat.n = g('eigen.targetMode');  flat.Nmax = g('eigen.maxCluster');

% ---- filter --------------------------------------------------------------
flat.rminEl   = g('filter.radiusElements');
flat.rminPhys = g('filter.radiusPhysical');
if isempty(flat.rminEl), flat.rminEl = NaN; end
switch g('filter.type')
    case 'none',    flat.filterMode = 'none';
    otherwise
        flat.filterMode = local_rev(g('filter.applyTo'), {'diagonal','all'}, {'diag','all'});
end

% ---- multiplicity --------------------------------------------------------
flat.tolMult  = g('multiplicity.tolerance');
flat.multRule = local_rev(g('multiplicity.method'), ...
                  {'binary','latch','hysteresis','subspace'}, ...
                  {'binary','latch','hyst','subspace'});
flat.tolEnter = g('multiplicity.enterTolerance');
flat.tolExit  = g('multiplicity.exitTolerance');
flat.subN     = g('multiplicity.subspaceSize');
flat.offDiag  = g('multiplicity.offDiagonal');

% ---- move ----------------------------------------------------------------
flat.move        = g('move.initial');
flat.moveFamily  = local_rev(g('move.policy'), ...
                    {'fixed','geometric','ladder','trustRatio'}, {'S0','S1','S2','S3'});
flat.moveMin     = g('move.minimum');
flat.s1Gamma     = g('move.geometric.ratio');
flat.s1AfterCoal = g('move.geometric.afterCoalescence');
flat.s2Levels    = g('move.levels');
flat.s2Window    = g('move.continuation.window');
flat.s2Tol       = g('move.continuation.tolerance');
flat.s2Signal    = local_rev(g('move.continuation.signal'), ...
                    {'boundVariable','designRms'}, {'beta','drms'});
flat.s3Lo = g('move.trust.loRatio');  flat.s3Hi   = g('move.trust.hiRatio');
flat.s3Down = g('move.trust.shrink'); flat.s3Up   = g('move.trust.grow');

% ---- inner ---------------------------------------------------------------
flat.innerSolver = g('optimizer.inner.type');
flat.mmaVariant  = g('optimizer.inner.variant');
flat.innerVar    = local_rev(g('optimizer.inner.variable'), ...
                    {'increment','design'}, {'drho','rho'});
flat.maxInner = g('optimizer.inner.maxIterations');
flat.tolInner = g('optimizer.inner.tolerance');
flat.minInner = g('optimizer.inner.minIterations');

% ---- outer ---------------------------------------------------------------
flat.maxOuter  = g('runtime.maxOuter');
flat.tolOuter  = g('stop.tolerance');
flat.outerNorm = g('stop.norm');
if g('stop.guards.settledMove'), flat.outerGuard = 'settledmove';
else,                            flat.outerGuard = 'none'; end

% ---- numerics ------------------------------------------------------------
flat.solver  = g('eigen.solver');
flat.threads = 1;  if ~g('runtime.singleThread'), flat.threads = maxNumCompThreads('automatic'); end
flat.verbose = g('runtime.verbose');

% ---- optional fields: present ONLY when active ---------------------------
if g('material.stiffness.continuation.enabled')
    flat.pSchedule = g('material.stiffness.continuation.schedule');
    if strcmp(g('material.stiffness.continuation.driver'),'ownCounter')
        flat.pDecouple = true;
    end
end
if g('material.mass.continuation.enabled')
    flat.massLowP = local_rev(g('material.mass.continuation.lowPModel'), ...
                     {'eq2','eq4','eq4a','eq4b'}, {'lin','4','4a','4b'});
end
if g('stop.guards.ladderExhausted'), flat.restorationGuard = 'R1'; end
if g('stop.guards.maxDesignChange'), flat.restorationGuard = 'R2'; end
if g('projection.enabled')
    flat.projection = struct('on',true, ...
        'betaSchedule', g('projection.beta.levels'), 'eta', g('projection.eta'));
end
if g('runtime.diagnostics'), flat.diag = true; end
if ~isempty(g('runtime.name')), flat.name = g('runtime.name'); end
end

% =========================================================================
function out = local_rev(value, canonical, legacy)
i = find(strcmp(char(value), canonical), 1);
if isempty(i)
    error('olh:config:toLegacyUnknown','no legacy spelling for ''%s''', char(value));
end
out = legacy{i};
end
