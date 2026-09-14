function out = ms_run(arm, nelx, nely, outDir)
%MS_RUN  Execute one diagnostic arm and derive every preregistered recorder.
%
%   The solver is NOT modified.  Everything the preregistration asks for that
%   res.hist does not already carry is reconstructed from res.diag.drho, which
%   the documented-inert per-iteration recorder provides:
%
%       rho_k = min(1, max(rho_min, rho_{k-1} + drho_k)),   rho_0 = design.initial
%
%   formed exactly as olhoffSolve forms it, and VALIDATED against
%   hist.vol(k) = mean(rho_k) at every iteration.  If that validation fails the
%   reconstruction is wrong and the run is refused.

t0 = tic;
guard = olhoffcurrent_paths(); %#ok<NASGU>          % fail-closed, before anything
gate  = olhoffcurrent_assert_dispatch();
man   = olhoffcurrent_source_manifest();
cur   = olhoffcurrent_currentness('Verbose', false);
assert(man.ok, 'ms_run:Integrity', 'OlhoffCurrent source integrity FAILED');
assert(~strcmp(cur.state,'LOCAL_MODIFIED'), 'ms_run:LocalModified', ...
    'OlhoffCurrent reports %s', cur.state);
maxNumCompThreads(1);

[cfg, meta] = ms_config(arm, nelx, nely);

% The baseline arm must BE production, not merely resemble it.
if strcmp(meta.arm, 'baseline')
    prod = olhoffcurrent_config(nelx, nely, 'Diagnostics', true);
    S = olh.config.schema(); bad = {};
    for k = 1:size(S,1)
        p = S{k,1};
        if strcmp(p, 'runtime.name'); continue; end
        if ~isequaln(olh.config.getPath(cfg,p), olh.config.getPath(prod,p))
            bad{end+1} = p; %#ok<AGROW>
        end
    end
    assert(isempty(bad), 'ms_run:BaselineNotProduction', ...
        'baseline arm differs from the production configuration in: %s', strjoin(bad,', '));
end

fprintf('[ms_run] %-10s %dx%d  preset=%s  cap=%d  diag=on\n', ...
    meta.arm, nelx, nely, meta.preset, meta.maxOuter);

tSolve = tic;
res = olhoffSolve(cfg);
wall = toc(tSolve);

% ---- reconstruct the density history --------------------------------------
NE      = nelx*nely;
rhoMin  = olh.config.getPath(cfg,'design.minimum');
rho0    = olh.config.getPath(cfg,'design.initial');
tolOut  = olh.config.getPath(cfg,'stop.tolerance');
epsRMS  = tolOut/sqrt(NE);

h  = res.hist;
nO = numel(h.N);
assert(isfield(res,'diag') && numel(res.diag.drho) == nO, 'ms_run:NoDiag', ...
    'per-iteration recorder missing or truncated (%d entries for %d iterations)', ...
    numel(res.diag.drho), nO);

rho  = rho0*ones(NE,1);
RHO  = zeros(NE, nO);
volErrMax = 0;
for k = 1:nO
    rho = min(1, max(rhoMin, rho + res.diag.drho{k}));
    RHO(:,k) = rho;
    volErrMax = max(volErrMax, abs(mean(rho) - h.vol(k)));
end
assert(volErrMax < 1e-12, 'ms_run:ReconstructionFailed', ...
    'reconstructed density disagrees with hist.vol by %.3e', volErrMax);
% Projection and density filtering are OFF, so the design variable IS the
% physical FE density.  Asserted, not assumed.
assert(isequaln(RHO(:,end), double(res.rho(:))), 'ms_run:DesignNotPhysical', ...
    'final reconstructed design density differs from res.rho');

% ---- per-iteration metrics -------------------------------------------------
tau = [epsRMS, 1e-4, 1e-3, 1e-2];
P = struct();
P.outer   = (1:nO).';
P.omega1  = h.omega(1,:).';
P.omega2  = h.omega(2,:).';
P.gap12   = h.gap12(:);
P.volume  = h.vol(:);
P.move    = h.move(:);
P.stage   = h.stage(:);
P.beta    = h.beta(:);
P.l2      = h.dxNorm2(:);
P.rms     = h.dxNorm2(:)/sqrt(NE);
P.maxAbs  = h.dxOuter(:);
P.nInner  = h.nInner(:);
P.innerConv = h.innerConv(:);
P.multN   = h.N(:);
P.degen   = h.degen(:);
P.tolOuter = tolOut*ones(nO,1);
P.epsRMS   = epsRMS*ones(nO,1);
P.Mnd   = zeros(nO,1); P.gray = zeros(nO,1); P.mid = zeros(nO,1);
P.nActive = zeros(nO, numel(tau));
for k = 1:nO
    r = RHO(:,k);
    P.Mnd(k)  = 100*mean(4*r.*(1-r));
    P.gray(k) = mean(r > 0.1 & r < 0.9);
    P.mid(k)  = mean(r >= 0.4 & r <= 0.6);
    d = abs(res.diag.drho{k});
    for j = 1:numel(tau); P.nActive(k,j) = sum(d > tau(j)); end
end
% Stopping predicate, decomposed exactly as olhoffSolve applies it.
P.stopRaw = P.l2 < tolOut;                       % the (P4) test itself
P.settled = [false; P.move(2:end) == P.move(1:end-1)];  % settledMove state
P.stopAdmitted = P.stopRaw & P.settled;          % what the guard actually admits
P.descent = [false; P.move(2:end) < P.move(1:end-1)];   % move-descent events

converged = any(contains(res.log,'converged at outer'));
if converged;              status = 'NATIVE_CONVERGED';
elseif nO >= meta.maxOuter; status = 'CAP_HIT';
else;                       status = 'UNRECOGNIZED_STOP'; end

out = struct('arm', meta.arm, 'label', meta.label, 'mesh', [nelx nely], 'NE', NE, ...
    'status', status, 'converged', converged, 'nOuter', nO, ...
    'innerTotal', sum(h.nInner), 'wall_s', wall, 'total_s', toc(t0), ...
    'omega', double(res.omega(:)), 'rhoFinal', RHO(:,end), ...
    'Mnd_final', P.Mnd(end), 'gray_final', P.gray(end), 'mid_final', P.mid(end), ...
    'volume_final', P.volume(end), 'tolOuter', tolOut, 'epsRMS', epsRMS, ...
    'tau', tau, 'per', P, 'log', {res.log}, ...
    'cfgHash', olhoffcurrent_config_hash(cfg), 'meta', meta, ...
    'sourceTree', man.treeHash, 'currentness', cur.state, ...
    'nResolved', numel(gate.resolved), 'volReconErrMax', volErrMax);

if nargin >= 4 && ~isempty(outDir)
    if ~isfolder(outDir); mkdir(outDir); end
    f = fullfile(outDir, sprintf('%s_%dx%d.mat', meta.arm, nelx, nely));
    save(f, 'out', 'cfg', 'RHO', '-v7.3');
    fprintf('[ms_run] saved %s\n', f);
end
fprintf(['[ms_run] %-10s %dx%d  %s  outer=%d inner=%d  omega1=%.12g  ' ...
         'M_nd=%.3f%%  gray=%.4f  wall=%.1fs\n'], meta.arm, nelx, nely, status, ...
    nO, sum(h.nInner), out.omega(1), out.Mnd_final, out.gray_final, wall);
end
