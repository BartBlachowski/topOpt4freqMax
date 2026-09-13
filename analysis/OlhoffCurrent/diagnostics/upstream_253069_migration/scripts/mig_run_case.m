function mig_run_case(side, caseName, outMat)
%MIG_RUN_CASE  One 160x20 solve of the migration protocol (MIGRATION_PREREGISTRATION §4).
%   side 'pre' | 'post' : the worktree's analysis/OlhoffCurrent through its own
%                         fail-closed gate (olhoffcurrent_paths)
%   side 'up'           : the read-only 253069 snapshot (mig_use_snapshot)
%   caseName BETA | EX3 | EX4 | PED | A6_pdecoupled160 | A7_massp160
%   The full solver result is saved; nothing is summarized away.
P = mig_paths();
switch side
    case {'pre', 'post'}
        restoredefaultpath; addpath(P.scripts); addpath(P.oc);
        guard = olhoffcurrent_paths(); %#ok<NASGU>
        impl = fullfile(P.oc, '+impl');
        assert(strncmp(which('olhoffSolve'), impl, numel(impl)), 'mig:dispatch', 'target solver not resolved');
        treeHash = olhoffcurrent_source_manifest('Verify', false).treeHash;
    case 'up'
        mig_use_snapshot(P.up);
        treeHash = 'git-archive 253069 (snapshot)';
    otherwise
        error('mig:side', 'side %s', side);
end
maxNumCompThreads(1);

if startsWith(caseName, 'A')
    % upstream behavioural anchor, legacy flat route (olhoffOpt), config from the
    % snapshot's frozen TMA configurations exactly as anchorCfg builds it
    [cfg, meta] = mig_anchor_cfg(caseName, P.up);
    t = tic; res = olhoffOpt(cfg); wall = toc(t);
else
    cfg = mig_case_cfg(side, caseName);
    meta = struct('label', caseName);
    t = tic; res = olhoffSolve(cfg); wall = toc(t);
end

meta.side = side; meta.case = caseName; meta.wall_s = wall;
meta.solver = which('olhoffSolve'); meta.mmasub = which('mmasub');
meta.treeHash = treeHash; meta.matlab = version; meta.when = char(datetime('now'));
meta.threads = maxNumCompThreads;
save(outMat, 'res', 'cfg', 'meta', '-v7.3');
fprintf('MIGCASE %-5s %-18s status=%s nOuter=%d omega1=%.15g wall=%.0fs solver=%s\n', ...
    side, caseName, res.status, numel(res.hist.N), res.omega(1), wall, meta.solver);
end
