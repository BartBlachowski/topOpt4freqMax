function out = fi_state()
%FI_STATE  Part 1: authoritative 480 state identity.  Read-only.
S = fi_setup();
study = fileparts(fileparts(mfilename('fullpath')));

expect = struct( ...
    'rho_sha256','0a498a7d6ab0565b29c15ff9364060d937d4df10aa661fc90d02c038ce6e4a60', ...
    'cfgHash','03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e', ...
    'implTree','edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb', ...
    'nOuter',386,'stage',3,'move',0.01,'N',2,'multJ',0);

o = struct();
o.rho_sha256 = local_vecHash(S.rho386);
o.cfgHash    = olhoffcurrent_config_hash(S.cfg);
o.cfgHash_recorded = S.cfgHash;
sm = olhoffcurrent_source_manifest('Verify',true);
o.implTree = sm.treeHash;  o.implTree_ok = sm.ok;
o.nOuter = S.nOuter;
o.stage  = S.hist.stage(end);
o.move   = S.hist.move(end);
o.N      = S.hist.N(end);
o.multJ  = S.hist.multJ(end);
o.nInner_final = S.hist.nInner(end);
o.innerConv_final = S.hist.innerConv(end);
o.omega_at_rho385 = S.hist.omega(:,end).';          % pre-update, iteration 386
o.gap12_at_rho385 = S.hist.gap12(end);
o.beta_at_386 = S.hist.beta(end);
o.volume_386  = mean(S.rho386);
o.volErr_386  = mean(S.rho386) - S.volfrac;
o.Mnd_386     = 100*mean(4*S.rho386.*(1-S.rho386));
o.gray_386    = mean(S.rho386>0.1 & S.rho386<0.9);
o.rho385_sha256 = local_vecHash(S.rho385);
o.drho386_sha256 = local_vecHash(S.drho386);
o.rho386_minmax = [min(S.rho386) max(S.rho386)];
o.rho385_minmax = [min(S.rho385) max(S.rho385)];

% post-update spectrum at the authoritative endpoint
E = fi_eval(S, S.rho386, 'full');
o.omega_at_rho386 = E.omega(:).';
o.gap12_at_rho386 = E.gap12;
o.N_detected_386  = E.N;
o.dOff_386        = E.dOff(:).';

% filter configuration
o.filter = struct('type',S.g('filter.type'),'applyTo',S.g('filter.applyTo'), ...
    'radiusPhysical',S.g('filter.radiusPhysical'),'rminEl',S.rminEl, ...
    'nnzH',nnz(S.flt.H),'Hsymmetric',issymmetric(S.flt.H), ...
    'HsMin',full(min(S.flt.Hs)),'HsMax',full(max(S.flt.Hs)));
o.multiplicity = struct('method',S.g('multiplicity.method'), ...
    'subspaceSize',S.g('multiplicity.subspaceSize'), ...
    'tolerance',S.g('multiplicity.tolerance'), ...
    'diagonalOffsets',S.g('multiplicity.diagonalOffsets'), ...
    'offDiagonal',S.g('multiplicity.offDiagonal'));
o.mma = struct('variant',S.g('optimizer.inner.variant'), ...
    'variable',S.g('optimizer.inner.variable'), ...
    'tolInner',S.g('optimizer.inner.tolerance'), ...
    'minInner',S.g('optimizer.inner.minIterations'), ...
    'maxInner',S.g('optimizer.inner.maxIterations'), ...
    'a0',1,'a',0,'c',1000,'d',0);
o.controller = struct('signal',S.g('move.continuation.signal'), ...
    'stopRule',S.g('stop.rule'),'levels',S.g('move.levels'), ...
    'terminalBranch',S.exh.terminalBranch,'terminalDeclIter',S.exh.terminalDeclIter);

fail = {};
chk = @(nm,a,b) local_chk(nm,a,b);
fail = [fail, chk('rho_sha256', o.rho_sha256, expect.rho_sha256)];
fail = [fail, chk('cfgHash',    o.cfgHash,    expect.cfgHash)];
fail = [fail, chk('implTree',   o.implTree,   expect.implTree)];
if o.nOuter ~= expect.nOuter, fail{end+1} = 'nOuter'; end
if o.stage  ~= expect.stage,  fail{end+1} = 'stage';  end
if o.move   ~= expect.move,   fail{end+1} = 'move';   end
if o.N      ~= expect.N,      fail{end+1} = 'N';      end
if o.multJ  ~= expect.multJ,  fail{end+1} = 'multJ';  end
if ~o.implTree_ok,            fail{end+1} = 'source manifest verify'; end

o.blockers = fail;
o.pass = isempty(fail);
o.verdict = 'FROZEN_480_STATE_IDENTITY_FAIL';
if o.pass, o.verdict = 'FROZEN_480_STATE_IDENTITY_PASS'; end

f = fullfile(study,'evaluations','state_identity.json');
fid = fopen(f,'w'); fprintf(fid,'%s',jsonencode(o,'PrettyPrint',true)); fclose(fid);
fprintf('[fi_state] %s\n', o.verdict);
fprintf('  rho386 %s\n  cfg    %s\n  impl   %s\n', o.rho_sha256, o.cfgHash, o.implTree);
fprintf('  outer=%d stage=%d move=%.3g N=%d multJ=%d nInner=%d\n', ...
    o.nOuter,o.stage,o.move,o.N,o.multJ,o.nInner_final);
fprintf('  gap12 @rho385=%.6g  @rho386=%.6g   Mnd=%.4f%%\n', ...
    o.gap12_at_rho385, o.gap12_at_rho386, o.Mnd_386);
for k=1:numel(fail), fprintf('  BLOCKER: %s\n', fail{k}); end
out = o;
end

function c = local_chk(nm,a,b)
if strcmp(a,b), c = {}; else, c = {sprintf('%s: %s ~= %s', nm, a, b)}; end
end

function h = local_vecHash(v)
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(typecast(double(v(:)),'uint8'));
d = typecast(md.digest(),'uint8');
h = lower(reshape(dec2hex(d,2).',1,[]));
end
