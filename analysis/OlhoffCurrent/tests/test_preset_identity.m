function nFail = test_preset_identity()
%TEST_PRESET_IDENTITY  Named presets keep their scientific identities apart.
%
%   No solve.  Checks, at 160x20 unless stated:
%     1. registry integrity: three canonical names, unique, no name collides with
%        an alias, compatibility and provenance aliases disjoint across presets
%     2. resolution: the compatibility alias resolves ONLY to the historical
%        beta-stall preset (same configuration hash); an unnamed call, a
%        provenance alias and an unknown name are refused
%     3. the resolved formulation of each preset, field by field
%     4. historical values preserved: the pre-migration 81-row configuration hash
%        recomputed from the migrated configuration equals the RECORDED hash
%        (beta-stall 160x20 campaign; three-rung C320; four-rung exhaustion C160)
%        -- configuration resolution only, no mesh is solved
%     5. production is a recorded, eligible, canonical preset; the historical
%        diagnostic is not eligible
%     6. distinct identity: Pedersen and beta-stall differ in material law and
%        controller, and the formulation text says so
%     7. schema has 87 rows

here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(root); addpath(here);
guard = olhoffcurrent_paths(); %#ok<NASGU>

BETA = 'duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered';
EX3  = 'duOlhoffSimpEq4bThreeRungStageExhaustionSensitivityFiltered';
PED  = 'duOlhoffPedersenAdaptiveBoxSensitivityFiltered';
COMPAT = 'duOlhoffFixedPenaltySensitivityFiltered';

nFail = 0;
fprintf('\n%s\nTEST_PRESET_IDENTITY\n%s\n', repmat('=',1,72), repmat('=',1,72));

% ---- 1. registry ----------------------------------------------------------
R = olhoffcurrent_presets();
names = {R.name};
nFail = nFail + chk('three canonical presets, unique', ...
    numel(R) == 3 && numel(unique(names)) == 3 && all(ismember({BETA, EX3, PED}, names)));
allCompat = [R.compatibilityAliases];
allHist = [R.historicalAliases];
nFail = nFail + chk('no alias equals a canonical name', ...
    ~any(ismember(allCompat, names)) && ~any(ismember(allHist, names)));
nFail = nFail + chk('aliases are not shared between presets', ...
    numel(unique(allCompat)) == numel(allCompat) && numel(unique(allHist)) == numel(allHist) && ...
    ~any(ismember(allCompat, allHist)));
nFail = nFail + chk('the only compatibility alias belongs to the beta-stall preset', ...
    isequal(allCompat, {COMPAT}) && isequal(olhoffcurrent_preset(BETA).compatibilityAliases, {COMPAT}));

% ---- 2. resolution --------------------------------------------------------
[cA, iA] = olhoffcurrent_config(160, 20, 'Preset', COMPAT);
cB = olhoffcurrent_config(160, 20, 'Preset', BETA);
nFail = nFail + chk('compatibility alias -> beta-stall preset, identical configuration hash', ...
    strcmp(iA.name, BETA) && strcmp(iA.resolvedVia, 'compatibilityAlias') && ...
    strcmp(olhoffcurrent_config_hash(cA), olhoffcurrent_config_hash(cB)));
nFail = nFail + chk('unnamed configuration is refused', ...
    throwsId(@() olhoffcurrent_config(160, 20), 'olhoffcurrent_config:PresetRequired'));
nFail = nFail + chk('unnamed preset lookup is refused', ...
    throwsId(@() olhoffcurrent_preset(), 'olhoffcurrent_preset:NameRequired'));
nFail = nFail + chk('unnamed run is refused', ...
    throwsId(@() olhoffcurrent_run(160, 20), 'olhoffcurrent_run:PresetRequired'));
nFail = nFail + chk('provenance aliases are refused (M4, S160x20, duOlhoffFrozenM4, duOlhoffAdaptivePedersen)', ...
    all(cellfun(@(n) throwsId(@() olhoffcurrent_preset(n), 'olhoffcurrent_preset:ProvenanceAlias'), ...
        {'M4', 'S160x20', 'duOlhoffFrozenM4', 'duOlhoffAdaptivePedersen', 'TR3_C'})));
nFail = nFail + chk('unknown preset is refused', ...
    throwsId(@() olhoffcurrent_preset('duOlhoffSomethingElse'), 'olhoffcurrent_preset:Unknown'));
nFail = nFail + chk('caveat requires a preset name', ...
    throwsId(@() olhoffcurrent_caveat(), 'olhoffcurrent_caveat:NameRequired'));

% ---- 3. resolved formulations ---------------------------------------------
g = @(c, p) olh.config.getPath(c, p);
cE = olhoffcurrent_config(160, 20, 'Preset', EX3);
cP = olhoffcurrent_config(160, 20, 'Preset', PED);
shared = @(c) (g(c,'material.stiffness.p') == 3 && ...
    ~g(c,'material.stiffness.continuation.enabled') && strcmp(g(c,'filter.type'),'sensitivity') && ...
    strcmp(g(c,'filter.applyTo'),'all') && g(c,'filter.radiusPhysical') == 0.06 && ...
    isempty(g(c,'filter.radiusElements')) && ~g(c,'projection.enabled') && ...
    strcmp(g(c,'multiplicity.method'),'subspace') && g(c,'multiplicity.subspaceSize') == 2 && ...
    g(c,'multiplicity.diagonalOffsets') && g(c,'multiplicity.offDiagonal') && ...
    strcmp(g(c,'optimizer.inner.type'),'mma') && strcmp(g(c,'optimizer.inner.variable'),'increment') && ...
    strcmp(g(c,'optimizer.inner.variant'),'published') && g(c,'optimizer.inner.tolerance') == 0.05 && ...
    g(c,'optimizer.inner.minIterations') == 5 && g(c,'optimizer.inner.maxIterations') == 500 && ...
    strcmp(g(c,'optimizer.inner.asymptoteHistory'),'inner') && g(c,'design.minimum') == 1e-3 && ...
    g(c,'eigen.maxCluster') == 4 && g(c,'stop.tolerance') == 0.05 && g(c,'runtime.singleThread'));
nFail = nFail + chk('shared numerics identical in all three (p, filter R, multiplicity, MMA, rho_min, maxCluster, eps)', ...
    shared(cB) && shared(cE) && shared(cP));
nFail = nFail + chk('beta-stall: SIMP, eq4b, ladder [0.04 0.02 0.01 0.005], beta signal, settled design-change stop, cap 400', ...
    strcmp(g(cB,'material.stiffness.model'),'simp') && strcmp(g(cB,'material.mass.model'),'eq4b') && ...
    strcmp(g(cB,'move.policy'),'ladder') && isequal(g(cB,'move.levels'),[0.04 0.02 0.01 0.005]) && ...
    strcmp(g(cB,'move.continuation.signal'),'boundVariable') && strcmp(g(cB,'stop.rule'),'designChange') && ...
    g(cB,'stop.guards.settledMove') && g(cB,'runtime.maxOuter') == 400);
nFail = nFail + chk('three-rung: SIMP, eq4b, ladder [0.04 0.02 0.01], stage exhaustion move AND stop, cap 1600', ...
    strcmp(g(cE,'material.stiffness.model'),'simp') && strcmp(g(cE,'material.mass.model'),'eq4b') && ...
    strcmp(g(cE,'move.policy'),'ladder') && isequal(g(cE,'move.levels'),[0.04 0.02 0.01]) && ...
    strcmp(g(cE,'move.continuation.signal'),'stageExhaustion') && strcmp(g(cE,'stop.rule'),'stageExhaustion') && ...
    g(cE,'runtime.maxOuter') == 1600);
nFail = nFail + chk('Pedersen: pedersen below 0.1, eq2 mass, adaptive box 0.10/0.002/x1.2/x0.7, no guards, no exhaustion, cap 400', ...
    strcmp(g(cP,'material.stiffness.model'),'pedersen') && g(cP,'material.stiffness.linearBelow') == 0.1 && ...
    strcmp(g(cP,'material.mass.model'),'eq2') && strcmp(g(cP,'move.policy'),'adaptive') && ...
    g(cP,'move.initial') == 0.10 && g(cP,'move.minimum') == 0.002 && g(cP,'move.adaptive.grow') == 1.2 && ...
    g(cP,'move.adaptive.shrink') == 0.7 && ~g(cP,'stop.guards.settledMove') && ...
    g(cP,'stop.guards.boxInactiveFraction') == 0 && ~g(cP,'stop.guards.ladderExhausted') && ...
    ~g(cP,'stop.guards.maxDesignChange') && strcmp(g(cP,'stop.rule'),'designChange') && ...
    ~strcmp(g(cP,'move.continuation.signal'),'stageExhaustion') && g(cP,'runtime.maxOuter') == 400);

% ---- 4. historical values preserved ---------------------------------------
F = jsondecode(fileread(fullfile(here, 'fixtures', 'schema_rows_pre_253069.json')));
old = F.rows;
cX4 = olh.config.resolve(olhoffcurrent_preset(BETA).upstreamPreset, 'domain.mesh.nelx', 160, ...
    'domain.mesh.nely', 20, 'move.continuation.signal', 'stageExhaustion', 'stop.rule', 'stageExhaustion', ...
    'runtime.maxOuter', 1600, 'runtime.singleThread', true, 'runtime.diagnostics', true, ...
    'runtime.verbose', false, 'runtime.name', 'x');
cE320 = olhoffcurrent_config(320, 40, 'Preset', EX3, 'Diagnostics', true);   % resolution only
nFail = nFail + chk('81 pre-migration rows recorded', numel(old) == 81);
nFail = nFail + chk('beta-stall 160x20: pre-migration hash 28756d22... reproduced from the migrated config', ...
    strcmp(oldHash(cB, old), '28756d22aacb59726be9f37583deca89fcfcecc93f9b867d46a49223ed1db697'));
nFail = nFail + chk('three-rung C320x40: pre-migration hash afad9ea4... reproduced (resolution only)', ...
    strcmp(oldHash(cE320, old), 'afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab'));
nFail = nFail + chk('four-rung exhaustion override C160x20: pre-migration hash 31d2ef38... reproduced', ...
    strcmp(oldHash(cX4, old), '31d2ef382746a942a4036d07dd6a1012742432cba0497d2de5ec24b51d2b5904'));

% ---- 5. production ----------------------------------------------------------
[pr, ev] = olhoffcurrent_production_preset();
prov = jsondecode(fileread(fullfile(root, 'PROVENANCE.json')));
evs = prov.production_preset_events;
if iscell(evs), first = evs{1}; else, first = evs(1); end
nFail = nFail + chk('production = Pedersen preset, from the latest recorded event', ...
    strcmp(pr.name, PED) && strcmp(ev.new_preset, PED) && strcmp(ev.old_preset, BETA));
nFail = nFail + chk('the first recorded event is the beta-stall preset (history kept)', ...
    strcmp(first.new_preset, BETA) && numel(evs) >= 2);
nFail = nFail + chk('the historical stage-exhaustion diagnostic is not production-eligible', ...
    ~olhoffcurrent_preset(EX3).productionEligible);

% ---- 6. distinct identity ---------------------------------------------------
dB = olh.config.describe(cB); dP = olh.config.describe(cP);
nFail = nFail + chk('Pedersen and beta-stall differ in stiffness, mass and move policy', ...
    ~strcmp(g(cB,'material.stiffness.model'), g(cP,'material.stiffness.model')) && ...
    ~strcmp(g(cB,'material.mass.model'), g(cP,'material.mass.model')) && ...
    ~strcmp(g(cB,'move.policy'), g(cP,'move.policy')));
nFail = nFail + chk('formulation text names Pedersen only for the Pedersen preset', ...
    contains(dP, 'Pedersen') && ~contains(dB, 'Pedersen'));
nFail = nFail + chk('caveats differ and the Pedersen caveat states the formulation split', ...
    ~strcmp(olhoffcurrent_caveat(PED), olhoffcurrent_caveat(BETA)) && ...
    contains(olhoffcurrent_caveat(PED), 'DISTINCT formulation') && ...
    contains(olhoffcurrent_caveat(PED), 'not a bug'));

% ---- 7. schema ----------------------------------------------------------------
nFail = nFail + chk('configuration schema has 87 rows', size(olh.config.schema(), 1) == 87);

fprintf('%s\n  failures: %d\n\n', repmat('-',1,72), nFail);
end

% =========================================================================
function h = oldHash(cfg, rows)
lines = cell(numel(rows),1);
for k = 1:numel(rows)
    if strcmp(rows{k}, 'runtime.name'); lines{k} = sprintf('%s=<excluded>', rows{k}); continue; end
    lines{k} = sprintf('%s=%s', rows{k}, show(olh.config.getPath(cfg, rows{k})));
end
joined = strjoin(lines, newline);
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(uint8(joined(:)));
d = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(d, 2).', 1, []));
end

function s = show(v)
if ischar(v);            s = v;
elseif isstring(v);      s = char(v);
elseif islogical(v);     s = mat2str(v);
elseif isnumeric(v);     s = mat2str(v, 17);
elseif iscell(v);        s = ['{' strjoin(cellfun(@show, v, 'UniformOutput', false), ',') '}'];
elseif isempty(v);       s = '[]';
else,                    s = class(v);
end
end

function tf = throwsId(fn, id)
try
    fn(); tf = false;
catch ME
    tf = strcmp(ME.identifier, id);
    if ~tf, fprintf('      threw %s (expected %s)\n', ME.identifier, id); end
end
end

function n = chk(label, ok)
if ok, fprintf('  [PASS] %s\n', label); n = 0;
else,  fprintf('  [FAIL] %s\n', label); n = 1; end
end
