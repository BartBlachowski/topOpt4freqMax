function pmg_anchor(which_, outMat)
%PMG_ANCHOR  ONE authorized 160x20 anchor solve in the merged normal checkout.
%   which_ 'historical' : duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered
%          'pedersen'   : duOlhoffPedersenAdaptiveBoxSensitivityFiltered
%   Same protocol as the migration's 'post' side (mig_run_case / mig_case_cfg): the
%   configuration from olhoffcurrent_config(160, 20, 'Preset', name) through the
%   fail-closed path guard, olhoffSolve, single thread, full result saved.
here = fileparts(mfilename('fullpath'));
oc = fileparts(fileparts(fileparts(here)));
repo = fileparts(fileparts(oc));
restoredefaultpath; addpath(here); addpath(oc);
guard = olhoffcurrent_paths(); %#ok<NASGU>
impl = fullfile(oc, '+impl');
assert(strncmp(which('olhoffSolve'), impl, numel(impl)), 'pmg:dispatch', 'target solver not resolved');
assert(strncmp(which('mmasub'), impl, numel(impl)), 'pmg:dispatch', 'target mmasub not resolved');
man = olhoffcurrent_source_manifest('Verify', true);
assert(man.ok, 'pmg:source', '+impl does not verify against SOURCE_MANIFEST.json');
switch which_
    case 'historical', name = 'duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered';
    case 'pedersen',   name = 'duOlhoffPedersenAdaptiveBoxSensitivityFiltered';
    otherwise, error('pmg:anchor', 'unknown anchor %s', which_);
end
nelx = 160; nely = 20;
assert(nelx == 160 && nely == 20, 'pmg:scope', 'only 160x20 is authorized');
maxNumCompThreads(1);
cfg = olhoffcurrent_config(nelx, nely, 'Preset', name);
[~, head] = system(sprintf('git --no-pager -C "%s" rev-parse HEAD', repo));
[~, dirty] = system(sprintf('git --no-pager -C "%s" status --porcelain -- analysis/OlhoffCurrent/+impl analysis/OlhoffCurrent/SOURCE_MANIFEST.json', repo));
meta = struct('anchor', which_, 'preset', name, 'mesh', [nelx nely], 'head', strtrim(head), ...
    'impl_dirty', strtrim(dirty), 'treeHash', man.treeHash, 'config_hash', olhoffcurrent_config_hash(cfg), ...
    'solver', which('olhoffSolve'), 'mmasub', which('mmasub'), 'matlab', version, ...
    'threads', maxNumCompThreads, 'started', char(datetime('now')));
t = tic; res = olhoffSolve(cfg); meta.wall_s = toc(t);
meta.finished = char(datetime('now'));
save(outMat, 'res', 'cfg', 'meta', '-v7.3');
fprintf('PMGANCHOR %-10s status=%s nOuter=%d inner=%d omega1=%.15g wall=%.0fs cfgHash=%s\n', ...
    which_, res.status, numel(res.hist.N), sum(res.hist.nInner), res.omega(1), meta.wall_s, meta.config_hash);
end
