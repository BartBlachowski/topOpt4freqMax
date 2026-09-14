function nFail = test_config()
%TEST_CONFIG  Defaults, overrides, preset composition, and every refusal.
nFail = 0;
ws = warning('off','olh:config:suspicious');
c = onCleanup(@() warning(ws));

% ---- defaults ------------------------------------------------------------
d = olh.config.defaults();
S = olh.config.schema();
nFail = nFail + local_check('defaults cover every schema field', ...
    isempty(setdiff(S(:,1), olh.config.paths(d))));
nFail = nFail + local_check('defaults validate', local_ok(@() olh.config.validate(d)));

% ---- resolve and overrides -----------------------------------------------
cfg = olh.config.resolve('duOlhoffFrozenM4');
nFail = nFail + local_check('frozen preset resolves', ~isempty(cfg));
nFail = nFail + local_check('provenance records the preset', ...
    strcmp(cfg.provenance.preset,'duOlhoffFrozenM4'));

cfg = olh.config.resolve('duOlhoffFrozenM4','domain.mesh.nelx',320,'domain.mesh.nely',40);
nFail = nFail + local_check('mesh override applied', cfg.domain.mesh.nelx==320);
nFail = nFail + local_check('mesh-scaled tolerance re-derived AFTER the override', ...
    cfg.stop.tolerance == 0.1);
cfg2 = olh.config.resolve('duOlhoffFrozenM4','domain.mesh.nelx',240,'domain.mesh.nely',30);
nFail = nFail + local_check('tolerance at 240x30 matches the stored historical value', ...
    isequal(typecast(cfg2.stop.tolerance,'uint8'), typecast(0.05*sqrt(240*30/3200),'uint8')));

% ---- preset composition is order-independent in effect -------------------
a = olh.config.resolve('pMassCompatible');
nFail = nFail + local_check('preset chain reaches the full formulation', ...
    a.material.mass.continuation.enabled && ...
    strcmp(a.material.stiffness.continuation.driver,'ownCounter') && ...
    a.stop.guards.maxDesignChange);

% ---- refusals ------------------------------------------------------------
R = { ...
 'unknown field',        @() olh.config.validate(setfield(olh.config.defaults(),'nonsense',1)), 'olh:config:unknownField'
 'unknown override',     @() olh.config.resolve('','no.such.field',1),                          'olh:config:unknownOverride'
 'unknown preset',       @() olh.config.resolve('notAPreset'),                                  'olh:presets:unknown'
 'bad enum',             @() olh.config.resolve('','material.mass.model','eq9'),                'olh:config:badEnum'
 'bad type',             @() olh.config.resolve('','projection.enabled',1),                     'olh:config:badType'
 'out of range',         @() olh.config.resolve('','design.volumeFraction',1.5),                'olh:config:outOfRange'
 'projection + sens flt',@() olh.config.resolve('duOlhoffFrozenM4','projection.enabled',true),  'olh:config:projectionNeedsDensityFilter'
 'projection, no levels',@() olh.config.resolve('projected','projection.beta.levels',[]),       'olh:config:projectionNeedsSchedule'
 'non-monotone beta',    @() olh.config.resolve('projected','projection.beta.levels',[4 2 1]),  'olh:config:projectionScheduleNotMonotone'
 'ladder guard, no ladder', @() olh.config.resolve('restorationLadderGuard','move.policy','fixed'), 'olh:config:ladderGuardNeedsLadder'
 'p driver needs ladder',@() olh.config.resolve('pContinuationCoupled','move.policy','fixed'),  'olh:config:pDriverNeedsLadder'
 'mass cont. without p', @() olh.config.resolve('duOlhoffFrozenM4','material.mass.continuation.enabled',true), 'olh:config:massContinuationNeedsP'
 'inert mass schedule',  @() olh.config.resolve('pMassCompatible','material.mass.continuation.lowPModel','eq4b'), 'olh:config:massContinuationIsNoOp'
 'subspace > maxCluster',@() olh.config.resolve('duOlhoffFrozenM4','multiplicity.subspaceSize',9), 'olh:config:subspaceTooLarge'
 'odd nely, mid support',@() olh.config.resolve('duOlhoffFrozenM4','domain.mesh.nely',21),      'olh:config:midSupportNeedsEvenNely'
 'ascending ladder',     @() olh.config.resolve('duOlhoffFrozenM4','move.levels',[0.01 0.05]),  'olh:config:ladderNotDescending'
 'lp + projection',      @() olh.config.resolve('projected','optimizer.inner.type','lp'),       'olh:config:projectionLpUnsupported'
 'no filter radius',     @() olh.config.resolve('duOlhoffFrozenM4','filter.radiusPhysical',[]), 'olh:config:noFilterRadius'
};
for k = 1:size(R,1)
    got = '';
    try
        R{k,2}();
    catch e
        got = e.identifier;
    end
    if strcmp(got, R{k,3})
        fprintf('  ok   refuses %-24s -> %s\n', R{k,1}, got);
    else
        fprintf('  FAIL %-24s expected %s, got ''%s''\n', R{k,1}, R{k,3}, got);  nFail = nFail+1;
    end
end

% ---- warnings, not coercion ----------------------------------------------
warning(ws);
lastwarn('');
w0 = warning('on','olh:config:suspicious');
cfg = olh.config.resolve('duOlhoffFrozenM4','stop.guards.settledMove',false);
[~, wid] = lastwarn;
warning(w0);
nFail = nFail + local_check('ladder without settledMove WARNS', strcmp(wid,'olh:config:suspicious'));
nFail = nFail + local_check('and is NOT coerced back', cfg.stop.guards.settledMove == false);

% ---- every preset validates ---------------------------------------------
T = olh.presets.list();
ws2 = warning('off','olh:config:suspicious');
for k = 1:size(T,1)
    ok = local_ok(@() olh.config.resolve(T{k,1}));
    nFail = nFail + local_check(sprintf('preset %-24s validates', T{k,1}), ok);
end
warning(ws2);
end

function n = local_check(name, cond)
if cond
    fprintf('  ok   %s\n', name); n = 0;
else
    fprintf('  FAIL %s\n', name); n = 1;
end
end
function tf = local_ok(f)
try
    f(); tf = true;
catch
    tf = false;
end
end
