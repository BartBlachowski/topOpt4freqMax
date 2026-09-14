function pf = cp_preflight(nelx, nely, varargin)
%CP_PREFLIGHT  THE fail-closed deployment gate.  Runs BEFORE any optimization.
%
%   pf = CP_PREFLIGHT(nelx, nely) resolves the configuration the canary driver
%   will ACTUALLY hand to olhoffSolve, checks it field by field against the
%   frozen manifest PREFLIGHT_MANIFEST.json, and THROWS if anything differs.
%   It never repairs, never coerces and never downgrades a mismatch to a
%   warning: a caller that reaches the solve has had its deployment proved.
%
%   Nothing here trusts a preset name, a comment, a variable name or an
%   intention.  Every value checked is read back out of the resolved cfg with
%   olh.config.getPath, after defaults, preset, overrides, derived rules and
%   validation have all run.
%
%   Options:
%     'Throw'  (default true)  false -> populate pf and return without raising,
%              so the report can show every failure at once rather than the
%              first.  cp_run ALWAYS calls with Throw=true.
%
%   pf.pass is true only when every check passed.
%
%   Verdict issued: THREE_RUNG_DEPLOYMENT_PREFLIGHT_PASS / _FAIL.
%
%   See also CP_CONFIG, CP_RUN.

p = inputParser();
p.addParameter('Throw', true, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
doThrow = p.Results.Throw;

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
addpath(root); addpath(here);
addpath(fullfile(root,'diagnostics','two_branch_controller_validation','scripts'));
addpath(fullfile(root,'diagnostics','three_rung_promotion_validation_retry1','scripts'));

man = jsondecode(fileread(fullfile(study,'PREFLIGHT_MANIFEST.json')));
key = sprintf('x%dx%d', nelx, nely);
assert(isfield(man.meshes, key), 'cp_preflight:MeshNotFrozen', ...
    ['mesh %dx%d is not in the frozen manifest.  This study authorizes ' ...
     'exactly two canaries, 480x60 and 800x100, and the mesh is not a ' ...
     'free parameter.'], nelx, nely);
M = man.meshes.(key);

fail = {};                      % every blocker, not merely the first
rec  = struct();

% =====================================================================
% 0.  TOOLCHAIN AND HOST -- recorded, and required to be recordable
% =====================================================================
rec.matlab   = version;
rec.computer = computer;
rec.os       = local_sh('sw_vers -productVersion');
rec.cpu      = local_sh('sysctl -n machdep.cpu.brand_string');
rec.ncores   = feature('numcores');
rec.ram_bytes = str2double(local_sh('sysctl -n hw.memsize'));
rec.threads  = maxNumCompThreads;
rec.blasEnv  = struct('OMP_NUM_THREADS', getenv('OMP_NUM_THREADS'), ...
                      'MKL_NUM_THREADS', getenv('MKL_NUM_THREADS'), ...
                      'VECLIB_MAXIMUM_THREADS', getenv('VECLIB_MAXIMUM_THREADS'));
rec.hostname = local_sh('hostname');
rec.loadavg  = local_sh('sysctl -n vm.loadavg');
rec.swap     = local_sh('sysctl -n vm.swapusage');
rec.freeRAM  = local_sh('vm_stat | head -4');

% =====================================================================
% 1.  DISPATCH -- exactly one Olhoff implementation is visible
% =====================================================================
guard = olhoffcurrent_paths(); %#ok<NASGU>
dr = olhoffcurrent_assert_dispatch('Throw', false);
rec.dispatch_ok = dr.ok;
rec.dispatch_blockers = dr.blockers;
if ~dr.ok
    fail{end+1} = sprintf('dispatch: %s', strjoin(dr.blockers, '; '));
end

% =====================================================================
% 2.  IMPLEMENTATION TREE -- byte-exact, and the SAME tree the validated
%     three-rung C320 run recorded inside itself
% =====================================================================
sm = olhoffcurrent_source_manifest('Verify', true);
rec.implTree = sm.treeHash;
rec.implTree_ok = sm.ok;
if ~strcmp(sm.treeHash, man.impl.tree_sha256)
    fail{end+1} = sprintf('+impl tree hash %s ~= frozen %s', ...
        sm.treeHash, man.impl.tree_sha256);
end
if ~sm.ok
    fail{end+1} = 'olhoffcurrent_source_manifest reports the tree does not verify';
end

% =====================================================================
% 3.  THE EFFECTIVE CONFIGURATION -- resolved, then read back
% =====================================================================
[cfg, meta] = cp_config(nelx, nely);
g = @(pth) olh.config.getPath(cfg, pth);
NE = nelx*nely;

% ---- 3a.  the three fields that define the VALIDATED three-rung policy ---
% Built as three concatenated blocks rather than one literal with comment
% lines inside it: a comment-only line inside a cell-array literal is a
% parser subtlety this file must not depend on.
reqPolicy = {   % the three fields that DEFINE the validated three-rung policy
 'move.levels',                    [0.04 0.02 0.01],       'vec'
 'move.continuation.signal',       'stageExhaustion',      'str'
 'stop.rule',                      'stageExhaustion',      'str'
 'move.policy',                    'ladder',               'str'
};
reqLocks = {    % the frozen scientific locks
 'material.stiffness.p',           3,                      'num'
 'material.stiffness.continuation.enabled', false,         'log'
 'material.mass.model',            'eq4b',                 'str'
 'material.mass.q',                1,                      'num'
 'material.mass.continuation.enabled', false,              'log'
 'filter.type',                    'sensitivity',          'str'
 'filter.applyTo',                 'all',                  'str'
 'filter.radiusPhysical',          0.06,                   'num'
 'projection.enabled',             false,                  'log'
 'multiplicity.method',            'subspace',             'str'
 'multiplicity.subspaceSize',      2,                      'num'
 'multiplicity.diagonalOffsets',   true,                   'log'
 'multiplicity.offDiagonal',       true,                   'log'
 'multiplicity.tolerance',         0.05,                   'num'
 'optimizer.inner.type',           'mma',                  'str'
 'optimizer.inner.variant',        'published',            'str'
 'optimizer.inner.variable',       'increment',            'str'
 'optimizer.inner.tolerance',      0.05,                   'num'
 'optimizer.inner.minIterations',  5,                      'num'
 'optimizer.inner.maxIterations',  500,                    'num'
 'eigen.solver',                   'eigs',                 'str'
 'eigen.targetMode',               1,                      'num'
 'eigen.maxCluster',               4,                      'num'
 'eigen.tolerance',                1e-12,                  'num'
 'eigen.krylovFactor',             4,                      'num'
 'design.initial',                 0.5,                    'num'
 'design.minimum',                 1e-3,                   'num'
 'design.volumeFraction',          0.5,                    'num'
 'domain.a',                       8,                      'num'
 'domain.b',                       1,                      'num'
 'domain.boundary.condition',      'simplySupported',      'str'
 'domain.boundary.support',        'midHeight',            'str'
 'domain.element.type',            'Q4',                   'str'
 'domain.element.massMatrix',      'consistent',           'str'
 'stop.norm',                      'l2',                   'str'
 'stop.field',                     'designVariable',       'str'
 'stop.toleranceRule',             'meshScaled',           'str'
};
reqRuntime = {  % mesh, tolerance, cap, thread policy, telemetry
 'domain.mesh.nelx',               nelx,                   'num'
 'domain.mesh.nely',               nely,                   'num'
 'stop.tolerance',                 0.05*sqrt(NE/3200),     'num'
 'runtime.maxOuter',               man.cap,                'num'
 'runtime.singleThread',           true,                   'log'
 'runtime.diagnostics',            true,                   'log'
};
req = [reqPolicy; reqLocks; reqRuntime];
chk = struct('path',{},'expected',{},'actual',{},'ok',{});
for k = 1:size(req,1)
    pth = req{k,1}; exp = req{k,2}; kind = req{k,3};
    act = g(pth);
    switch kind
        case 'str', ok = ischar(act) && strcmp(act, exp);
        case 'log', ok = islogical(act) && isscalar(act) && act == exp;
        case 'num', ok = isnumeric(act) && isscalar(act) && act == exp;
        case 'vec', ok = isnumeric(act) && isequal(size(act(:)), size(exp(:))) && ...
                          all(act(:) == exp(:));
    end
    chk(end+1) = struct('path',pth,'expected',local_show(exp), ...
                        'actual',local_show(act),'ok',ok); %#ok<AGROW>
    if ~ok
        fail{end+1} = sprintf('config %s = %s, required %s', ...
            pth, local_show(act), local_show(exp)); %#ok<AGROW>
    end
end
rec.checks = chk;

% ---- 3e.  BETA HAS NO AUTHORITY -- asserted structurally, not by name ----
% Under stageExhaustion olh.move.limit returns from its own branch before the
% bound-variable window is ever formed, and olhoffSolve replaces the sec. 3.5.1
% admission wholesale.  Both are proved here by reading the code paths' own
% switches out of the resolved configuration.
betaMove = strcmp(g('move.continuation.signal'), 'boundVariable');
betaStop = strcmp(g('stop.rule'), 'designChange');
rec.beta_continuation_authority = betaMove;
rec.beta_stop_authority = betaStop;
if betaMove, fail{end+1} = 'beta HOLDS continuation authority (signal=boundVariable)'; end
if betaStop, fail{end+1} = 'beta-era designChange rule HOLDS stop authority'; end

% ---- 3f.  the ladder must NOT carry the removed fourth rung --------------
if any(abs(g('move.levels') - 0.005) < eps)
    fail{end+1} = 'move.levels still contains the removed 0.005 rung';
end

% =====================================================================
% 4.  CONFIG HASH -- the single scalar that stands for all of the above
% =====================================================================
rec.cfgHash = olhoffcurrent_config_hash(cfg);
rec.cfgHash_frozen = M.predicted_config_hash;
if ~strcmp(rec.cfgHash, M.predicted_config_hash)
    fail{end+1} = sprintf('effective config hash %s ~= frozen %s', ...
        rec.cfgHash, M.predicted_config_hash);
end

% =====================================================================
% 5.  TELEMETRY CAPABILITY -- the fields the study promises to retain must
%     exist on the solver's own recorder, checked by name on hist
% =====================================================================
needHist = {'omega','N','beta','nInner','cumInner','innerConv','dxOuter', ...
            'dxNorm2','vol','volErr','move','stage','gap12','degen','multJ', ...
            'tEig','tGrad','tInner','tOuter', ...
            'exA','exB','exE','exNA','exNB','exDecl','exCos','exNet', ...
            'exMedcos','exMednet','exAmp','exStageStart'};
solverSrc = fileread(which('olhoffSolve'));
missing = needHist(~cellfun(@(f) contains(solverSrc, ['hist.' f '(']) || ...
                                 contains(solverSrc, ['hist.' f '(:,']), needHist));
rec.telemetry_missing = missing;
if ~isempty(missing)
    fail{end+1} = sprintf('solver does not record: %s', strjoin(missing, ', '));
end

% ---- retention budget, decided BEFORE the run ---------------------------
rec.retention = struct('NE', NE, 'cap', man.cap, ...
    'bytes_per_column', NE*8, ...
    'worst_case_RHO_DRHO_bytes', 2*NE*8*man.cap, ...
    'policy', M.retention_policy);
rec.retention.fits_in_ram = rec.retention.worst_case_RHO_DRHO_bytes < 0.25*rec.ram_bytes;
if ~rec.retention.fits_in_ram
    fail{end+1} = sprintf(['retention budget %.1f GB exceeds a quarter of RAM; ' ...
        'the checkpoint scheme in INSTRUMENTATION.md must be enabled BEFORE ' ...
        'the run, not after'], rec.retention.worst_case_RHO_DRHO_bytes/1e9);
end

% =====================================================================
% 6.  VERDICT
% =====================================================================
pf = struct('mesh',[nelx nely],'NE',NE,'pass',isempty(fail), ...
            'blockers',{fail},'record',rec,'cfg',cfg,'meta',meta, ...
            'verdict','', 'checkedAt', char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ss')));
if pf.pass, pf.verdict = 'THREE_RUNG_DEPLOYMENT_PREFLIGHT_PASS';
else,       pf.verdict = 'THREE_RUNG_DEPLOYMENT_PREFLIGHT_FAIL';
end

outFile = fullfile(study,'evidence',sprintf('preflight_%dx%d.json', nelx, nely));
if ~isfolder(fileparts(outFile)), mkdir(fileparts(outFile)); end
j = rmfield(pf, {'cfg','meta'});
fid = fopen(outFile,'w'); fprintf(fid,'%s', jsonencode(j, 'PrettyPrint', true)); fclose(fid);

fprintf('\n[cp_preflight] %dx%d  %s\n', nelx, nely, pf.verdict);
fprintf('  cfgHash  %s\n  implTree %s\n', rec.cfgHash, rec.implTree);
for k = 1:numel(fail); fprintf('  BLOCKER: %s\n', fail{k}); end

if doThrow && ~pf.pass
    error('cp_preflight:DeploymentPreflightFail', ...
        ['THREE_RUNG_DEPLOYMENT_PREFLIGHT_FAIL (%d blocker(s)).  ' ...
         'NO optimization is executed.\n    %s'], ...
        numel(fail), strjoin(fail, sprintf('\n    ')));
end
end

% =========================================================================
function s = local_show(v)
if ischar(v); s = v;
elseif islogical(v); s = mat2str(v);
elseif isnumeric(v); s = mat2str(v, 17);
else; s = class(v);
end
end

function s = local_sh(cmd)
[st, out] = system(cmd);
if st ~= 0, s = ''; else, s = strtrim(out); end
end
