function out = bg_run_arm(armName, nelx, nely, upstreamPreset, overrides, maxOuter, outDir)
%BG_RUN_ARM  Run one bimodality-gap experiment arm through the production solver.
%
%   out = BG_RUN_ARM(armName, nelx, nely, upstreamPreset, overrides, maxOuter, outDir)
%
%   Route (identical to olhoffcurrent_run except that scientific overrides are
%   permitted and RECORDED):
%       olhoffcurrent_paths (fail-closed path gate)
%         -> olh.config.resolve(upstreamPreset, mesh, overrides..., runtime)
%         -> olhoffSolve(cfg)
%   The production tree analysis/OlhoffCurrent/+impl is NOT modified.  Every
%   run records the effective configuration, its SHA-256 (olhoffcurrent_config_hash),
%   the live +impl tree hash and the repository HEAD (olhoffcurrent_provenance).
%
%   Output: <outDir>/BG_<arm>_<nelx>x<nely>.mat (v7) and .json summary.

here     = fileparts(mfilename('fullpath'));            % docs/bimodality_gap/scripts
repoRoot = fileparts(fileparts(fileparts(here)));        % repo root
addpath(fullfile(repoRoot, 'analysis', 'OlhoffCurrent'));
maxNumCompThreads(1);

name = sprintf('BG_%s_%dx%d', armName, nelx, nely);
if nargin < 7 || isempty(outDir), outDir = fullfile(fileparts(here), 'runs'); end
if ~exist(outDir, 'dir'), mkdir(outDir); end

[guard, gate] = olhoffcurrent_paths(); %#ok<ASGLU>
prov = olhoffcurrent_provenance();

cfg = olh.config.resolve(upstreamPreset, ...
    'domain.mesh.nelx', nelx, 'domain.mesh.nely', nely, ...
    overrides{:}, ...
    'runtime.maxOuter', maxOuter, 'runtime.singleThread', true, ...
    'runtime.diagnostics', false, 'runtime.verbose', false, 'runtime.name', name);
cfgHash = olhoffcurrent_config_hash(cfg);
desc    = olh.config.describe(cfg);

% effective filter radius in ELEMENT units, exactly as olhoffSolve derives it
rp = olh.config.getPath(cfg, 'filter.radiusPhysical');
if ~isempty(rp) && rp > 0
    rminEl = rp / (olh.config.getPath(cfg,'domain.b') / nely);
else
    rminEl = olh.config.getPath(cfg, 'filter.radiusElements');
end

fprintf('[%s] cfgHash=%s  rminEl=%.6g  eps=%.6g  maxOuter=%d\n', name, cfgHash, rminEl, ...
    olh.config.getPath(cfg,'stop.tolerance'), maxOuter);
t = tic;
res = olhoffSolve(cfg);
wall = toc(t);

out = struct();
out.name           = name;
out.arm            = armName;
out.mesh           = [nelx nely];
out.upstreamPreset = upstreamPreset;
out.overrides      = {overrides};
out.maxOuter       = maxOuter;
out.cfg            = cfg;
out.cfgHash        = cfgHash;
out.describe       = desc;
out.rminEl         = rminEl;
out.eps            = olh.config.getPath(cfg,'stop.tolerance');
out.rho            = res.rho(:);
out.omega          = res.omega(:);
out.lambda         = res.lambda(:);
out.hist           = res.hist;
out.aux            = res.aux;
out.log            = {res.log};
out.status         = res.status;
out.nOuter         = res.nOuter;
out.wall_s         = wall;
out.solver_wall_s  = res.wallclock;
out.matlab_version = version;
out.threads        = maxNumCompThreads;
out.repo_head      = prov.main_repo_commit;
out.impl_tree_sha256 = prov.live_source_tree_sha256;
out.impl_n_files   = prov.live_source_n_files;
out.resolved_mmasub = which('mmasub');
out.timestamp      = char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ss'));
out.Mnd_final      = mean(4*out.rho.*(1-out.rho));
out.gray_final     = mean(out.rho > 0.1 & out.rho < 0.9);

save(fullfile(outDir, [name '.mat']), '-struct', 'out', '-v7');

J = rmfield(out, {'cfg','rho','omega','lambda','hist','aux','log','describe'});
J.omega1 = out.omega(1); J.omega2 = out.omega(2);
J.gap12  = (out.omega(2)-out.omega(1))/out.omega(1);
J.overrides = overrides;
fid = fopen(fullfile(outDir, [name '.json']), 'w');
fwrite(fid, jsonencode(J, 'PrettyPrint', true)); fclose(fid);
fprintf('[%s] DONE status=%s nOuter=%d omega1=%.6f Mnd=%.5f gray=%.5f wall=%.1fs\n', ...
    name, out.status, out.nOuter, out.omega(1), out.Mnd_final, out.gray_final, wall);
end
