function sd_m1_post()
%SD_M1_POST  Reconstruct and verify the M1 trajectory; preregistered P-prefix test.
%
%   rho_k = min(1, max(rhomin, rho_{k-1} + drho_k)) is replayed with the solver's
%   own statement from the recorded res.diag.drho, and must end bitwise at
%   res.rho.  The per-element adaptive box is replayed with the snapshot's own
%   olh.move.limit on that sequence and must reproduce hist.move (max) and
%   aux.moveMean (mean) bitwise.  Then M1 hist/aux are compared bitwise with the
%   retained S480x60 record for every iteration whose START state has all
%   elements > 0.1 (k < k*).
P = sd_use_source();
L = load(fullfile(P.m1dir, 'M1_480x60_res.mat')); res = L.res; cfg = L.cfg;
S = load(P.s480, 'res'); s = S.res;
NE = res.mdl.nele; n = res.nOuter;
rhomin = cfg.design.minimum;
RHO = zeros(NE, n); DRHO = zeros(NE, n); DVEC = zeros(NE, n+1);
rho = cfg.design.initial*ones(NE,1);
mvState = []; hstub = struct('N', [], 'beta', [], 'omega', []);
startMin = zeros(1,n);
for k = 1:n
    startMin(k) = min(rho);
    [mv, mvState] = olh.move.limit(cfg, k, hstub, mvState, rho);
    DVEC(:,k) = mv;
    d = res.diag.drho{k};
    rho = min(1, max(rhomin, rho + d));
    RHO(:,k) = rho; DRHO(:,k) = d;
end
[mv, ~] = olh.move.limit(cfg, n+1, hstub, mvState, rho); DVEC(:,n+1) = mv;   % the box a 65th iteration would use
chk.final_rho_bitwise = isequal(RHO(:,n), res.rho);
chk.move_max_bitwise  = isequal(max(DVEC(:,1:n),[],1), res.hist.move);
mm = zeros(1,n); md = zeros(1,n);
for k = 1:n, mm(k) = mean(DVEC(:,k)); r = RHO(:,k); md(k) = 4*mean(r.*(1-r)); end   % solver's per-vector form
chk.move_mean_bitwise = isequal(mm, res.aux.moveMean);
chk.Mnd_bitwise = isequal(md, res.aux.Mnd);
kstar = find(startMin <= 0.1, 1);
chk.kstar = kstar; chk.startMin = startMin;

% ---- P-prefix: bitwise equality with retained S480 for k < k* ------------
f = {'omega','N','beta','nInner','dxOuter','vol','degen','multJ','innerConv','cumInner', ...
     'dxNorm2','move','gap12','volErr','dBeta','stage'};
firstDiff = struct();
for i = 1:numel(f)
    a = res.hist.(f{i}); b = s.hist.(f{i});
    m = min(size(a,2), size(b,2));
    dd = find(any(a(:,1:m) ~= b(:,1:m) & ~(isnan(a(:,1:m)) & isnan(b(:,1:m))), 1), 1);
    if isempty(dd), dd = NaN; end
    firstDiff.(f{i}) = dd;
end
a = res.aux.Mnd; b = s.aux.Mnd; m = min(numel(a), numel(b));
dd = find(a(1:m) ~= b(1:m), 1); if isempty(dd), dd = NaN; end; firstDiff.auxMnd = dd;
a = res.aux.moveMean; b = s.aux.moveMean; dd = find(a(1:m) ~= b(1:m), 1); if isempty(dd), dd = NaN; end; firstDiff.auxMoveMean = dd;
fd = struct2array(firstDiff);
chk.firstDiff = firstDiff;
chk.first_any_diff = min(fd(~isnan(fd)));
if isempty(chk.first_any_diff), chk.first_any_diff = NaN; end
% hist(k) depends on the start state of k (FE) and on drho_k; aux.Mnd(k) on rho_k.
% Preregistered test: every entry with k < k* must be bitwise equal.
chk.prefix_pass = isnan(chk.first_any_diff) || chk.first_any_diff >= kstar;
chk.prefix_verdict = 'PREFIX_FAIL'; if chk.prefix_pass, chk.prefix_verdict = 'PREFIX_BITWISE_PASS'; end

% ---- preregistered M1 same-state set ------------------------------------
sel = [kstar-1, kstar+5, n];
names = arrayfun(@(k) sprintf('M1_k%03d', k), sel, 'UniformOutput', false);
rhos = RHO(:, sel);
boxes = DVEC(:, sel+1);          % box of the iteration that STARTS from that state
save(fullfile(P.eval, 'm1_states.mat'), 'names', 'rhos', 'boxes', 'sel', 'kstar');
hist = res.hist; aux = res.aux; omegaFinal = res.omega; log = res.log; %#ok<NASGU>
dlamPred = res.diag.dlamPred; lamDiag = res.diag.lam; betaDiag = res.diag.beta; %#ok<NASGU>
save(fullfile(P.m1dir, 'M1_480x60_trajectory.mat'), 'RHO', 'DRHO', 'DVEC', 'hist', 'aux', ...
    'omegaFinal', 'log', 'dlamPred', 'lamDiag', 'betaDiag', 'cfg', '-v7.3');
fid = fopen(fullfile(P.m1dir, 'M1_verification.json'), 'w');
fprintf(fid, '%s\n', jsonencode(chk, 'PrettyPrint', true)); fclose(fid);
fprintf('final_rho_bitwise=%d move_max=%d move_mean=%d Mnd=%d kstar=%d first_any_diff=%g prefix=%s\n', ...
    chk.final_rho_bitwise, chk.move_max_bitwise, chk.move_mean_bitwise, chk.Mnd_bitwise, kstar, chk.first_any_diff, chk.prefix_verdict);
disp(firstDiff);
fprintf('startMin(1:8) = %s\n', mat2str(startMin(1:8), 6));
end
