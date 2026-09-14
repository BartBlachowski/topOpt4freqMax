function nFail = cv_tests()
%CV_TESTS  Software/mechanics tests for the frozen two-branch stage-exhaustion
%   controller.  Preregistration section 8, tests 1-16.
%
%   THESE ARE SOFTWARE TESTS, NOT SCIENTIFIC EVIDENCE.  Sub-160x20 meshes are
%   used only where a real solve is needed to exercise the solver's own wiring.
%
%   Tests 1-11 drive olh.move.limit and olh.move.exhaustion through EXACTLY the
%   call sequence olhoffSolve uses (local_replay below mirrors it line for line),
%   so the controller logic is exercised deterministically and instantly.  The
%   solver's own wiring is additionally proved by a real candidate solve
%   (test 12), by the production bitwise reproduction (test 14) and, in the
%   scientific runs, by the 400x50 prefix-equivalence check.

repo  = local_repo();
root  = fullfile(repo,'analysis','OlhoffCurrent');
addpath(root);
addpath(fullfile(root,'diagnostics','dynamical_regime','scripts'));
addpath(fullfile(root,'diagnostics','two_branch_maturity_240','scripts'));
guard = olhoffcurrent_paths(); %#ok<NASGU>

nFail = 0; R = {};
fprintf('\n%s\nCV_TESTS  two-branch stage-exhaustion controller\n%s\n', ...
    repmat('=',1,72), repmat('=',1,72));

NE = 400; tol = 0.05*sqrt(NE/3200);   % a small synthetic design space
rho0 = 0.5;
v = zeros(NE,1); v(1:NE) = (1:NE)'/NE; v = v/norm(v);   % a fixed unit direction

% ---------------------------------------------------------------- 1,2,3
% A true / B false;  A false / B true;  A false / B false.
big   = 2.0*tol;      % amplitude clearly >= tol
small = 0.20*tol;     % amplitude clearly <  tol

[exA,~] = local_synth(NE, tol, rho0, v, 80, 'alternate', big);
[nFail,R] = chk(nFail,R,'1  A true, B false -> E true', ...
    all(exA.A(40:80)) && ~any(exA.B(40:80)) && all(exA.E(40:80)));

[exB,~] = local_synth(NE, tol, rho0, v, 80, 'coherent', small);
[nFail,R] = chk(nFail,R,'2  A false, B true -> E true', ...
    ~any(exB.A(40:80)) && all(exB.B(40:80)) && all(exB.E(40:80)));

[exN,~] = local_synth(NE, tol, rho0, v, 80, 'coherent', big);
[nFail,R] = chk(nFail,R,'3  A false, B false -> E false', ...
    ~any(exN.A(20:80)) && ~any(exN.B(20:80)) && ~any(exN.E(20:80)));

% the DOCUMENTED hole: low-amplitude cancellation satisfies neither branch
[exH,~] = local_synth(NE, tol, rho0, v, 80, 'alternate', small);
[nFail,R] = chk(nFail,R,'3b documented hole: low-amplitude cancellation -> E false', ...
    ~any(exH.A(20:80)) && ~any(exH.B(20:80)));

% ---------------------------------------------------------------- 4
% Persistence boundary.  Earliest possible declaration is stageStart+38: the
% medians need a full 20-window (first defined at s+19) and the counter then
% needs 20 consecutive.  That is exactly tb_branches at s=1 (window begins 20,
% closes 39).
k19 = find(exB.nB >= 19, 1);  k20 = find(exB.nB >= 20, 1);
[nFail,R] = chk(nFail,R,'4  persistence boundary: 19 no declaration, 20 declares', ...
    k19 == 38 && k20 == 39 && exB.declIter == 39 && exB.declBegin == 20 && ...
    strcmp(exB.declBranch,'B'));

% ---------------------------------------------------------------- 5,6,7
% Full replay through olh.move.limit: window reset, one rung per transition,
% and the 0.005 floor.  A coherent small-step design exhausts every stage.
cfg = local_candCfg(repo, 160, 20);
rep = local_replay(cfg, NE, tol, rho0, v, 400, 'coherent', small);

resetOk = true;
for i = 2:numel(rep.stageStarts)
    s = rep.stageStarts(i);
    % nothing stage-local may be defined before s+19, and the first declaration
    % of a stage may not precede s+38
    resetOk = resetOk && all(isnan(rep.ex.medcos(s:min(s+18,end)))) && ...
              all(rep.ex.nA(s:min(s+18,end))==0) && all(rep.ex.nB(s:min(s+18,end))==0);
end
firstDeclOfStage = rep.declIters;
earliestOk = all(firstDeclOfStage(:) >= rep.stageStarts(:) + 38);
[nFail,R] = chk(nFail,R,'5  window reset after descent (no pre-transition data, no decl before s+38)', ...
    resetOk && earliestOk);

steps = diff(rep.stageSeq);
oneRung = all(ismember(steps,[0 1])) && ...
          numel(rep.descentIters) == numel(unique(rep.descentIters)) && ...
          numel(rep.descentIters) == numel(rep.stageStarts)-1 && ...
          isequal(rep.stageSeq(rep.descentIters).', 2:(numel(rep.descentIters)+1)) && ...
          isequal(rep.descentIters(:), rep.stageStarts(2:end).') && ...
          numel(rep.declIters) == numel(rep.descentIters) + double(rep.ex.declared) && ...
          all(rep.descentIters(:) == rep.declIters(1:numel(rep.descentIters)).' + 1);
[nFail,R] = chk(nFail,R,'6  exactly one rung per accepted transition, applied the iteration after declaration', ...
    oneRung);

[nFail,R] = chk(nFail,R,'7  move never drops below 0.005', ...
    min(rep.move) >= 0.005 && all(ismember(rep.move,[0.04 0.02 0.01 0.005])));

% ---------------------------------------------------------------- 8,9
% beta stall alone cannot descend the move, and cannot admit convergence.
% Feed a beta history that stalls hard from the first iteration while the design
% keeps making large coherent steps (E never true).
repBeta = local_replay(cfg, NE, tol, rho0, v, 200, 'coherent', big, 'stalledBeta', true);
[nFail,R] = chk(nFail,R,'8  beta stall alone cannot descend the move', ...
    all(repBeta.move == 0.04) && all(repBeta.stageSeq == 1) && ...
    repBeta.betaWouldFire && isempty(repBeta.descentIters));
[nFail,R] = chk(nFail,R,'9  beta stall alone cannot admit terminal convergence', ...
    ~any(repBeta.conv));

% ---------------------------------------------------------------- 10,11
% move_min with E false cannot converge; move_min with persistent E does, at the
% declaring iteration.
[nFail,R] = chk(nFail,R,'10 move_min + E false cannot converge', ...
    local_terminalCase(cfg, NE, tol, rho0, v, big, false));
[nFail,R] = chk(nFail,R,'11 move_min + persistent E true converges at the declaring iteration', ...
    local_terminalCase(cfg, NE, tol, rho0, v, small, true));

% ---------------------------------------------------------------- 12,13
% Real solve.  CAP_HIT must remain CAP_HIT, and the status precedence that keeps
% a solver failure a failure must be untouched.
c80 = local_candCfg(repo, 80, 10); c80 = olh.config.assign(c80,'runtime.maxOuter',45);
r80 = olhoffSolve(c80);
[nFail,R] = chk(nFail,R,'12 CAP_HIT remains CAP_HIT under the candidate', ...
    strcmp(r80.status,'CAP_HIT') && r80.nOuter == 45 && ...
    ~any(contains(r80.log,'converged at outer iteration')));

% The status function decides SOLVER_FAILURE > CAP_HIT > CONVERGED.  Extract it
% from the file on disk and from the task-start commit and require the text to
% be identical: the candidate may not relabel a failure or a cap as convergence.
nowTxt  = fileread(fullfile(root,'+impl','architecture','olhoffSolve.m'));
baseTxt = local_git(repo, ['show b6014ba8bca41f85671d79ab4c8bdee7419880bb:' ...
    'analysis/OlhoffCurrent/+impl/architecture/olhoffSolve.m']);
[nFail,R] = chk(nFail,R,'13 failure remains failure: status precedence byte-identical to task start', ...
    ~isempty(baseTxt) && strcmp(local_fun(nowTxt,'local_status'), local_fun(baseTxt,'local_status')) && ...
    contains(local_fun(nowTxt,'local_status'),'SOLVER_FAILURE'));

% ---------------------------------------------------------------- 14
% Candidate OFF reproduces production.  The production configuration must be
% field-for-field what it was, and the solver must take none of the new paths.
cp = olhoffcurrent_config(160,20);
[nFail,R] = chk(nFail,R,'14a production config still selects the production controller', ...
    strcmp(cp.move.continuation.signal,'boundVariable') && ...
    strcmp(cp.stop.rule,'designChange') && ...
    isequal(cp.move.levels,[0.04 0.02 0.01 0.005]));
% the real bitwise proof is test_preset_equivalence, run separately by cv_gate.

% ---------------------------------------------------------------- 15
% Telemetry is inert: diagnostics on/off must give the same trajectory.
cA = local_candCfg(repo,80,10); cA = olh.config.assign(cA,'runtime.maxOuter',30,'runtime.diagnostics',false);
cB = olh.config.assign(cA,'runtime.diagnostics',true);
rA = olhoffSolve(cA); rB = olhoffSolve(cB);
[nFail,R] = chk(nFail,R,'15 telemetry does not alter numerical results', ...
    isequal(rA.rho,rB.rho) && isequal(rA.hist.omega,rB.hist.omega) && ...
    isequal(rA.hist.dxNorm2,rB.hist.dxNorm2) && isequal(rA.hist.exA,rB.hist.exA));

% ---------------------------------------------------------------- 16
% Online/offline equivalence against the FROZEN tb_branches, element by element.
ok16 = true; detail16 = {};
% (a) the surviving 400x50 fixed-move arm
f = fullfile(root,'evidence','move_activity_400','F400_400x50_trajectory.mat');
if isfile(f)
    S = load(f);
    [okF, dF] = local_equiv(S.RHO, S.hist.dxNorm2(:), S.hist.move(:), ...
                            S.cfg.design.initial, 20000);
    ok16 = ok16 && okF; detail16{end+1} = sprintf('F400:%s', dF);
else
    ok16 = false; detail16{end+1} = 'F400:MISSING';
end
% (b) synthetic trajectories exercising A, B, neither, and A-then-B
for nm = {'alternate','coherent','none','mixed'}
    [RHO, amp, mv] = local_traj(NE, tol, rho0, v, 200, nm{1}, big, small);   %#ok<ASGLU>
    [okS, dS] = local_equiv(RHO, amp, mv, rho0, NE);
    ok16 = ok16 && okS; detail16{end+1} = sprintf('%s:%s', nm{1}, dS); %#ok<AGROW>
end
[nFail,R] = chk(nFail,R,sprintf('16 online == frozen tb_branches  [%s]', ...
    strjoin(detail16,' ')), ok16);

fprintf('%s\n', repmat('-',1,72));
fprintf('CV_TESTS: %d test(s), %d failure(s)\n', numel(R), nFail);
outF = fullfile(fileparts(fileparts(mfilename('fullpath'))),'evidence','software_tests.json');
T = struct('generated', char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z''')), ...
           'nTests', numel(R), 'nFail', nFail, 'results', {R});
fid=fopen(outF,'w'); fwrite(fid, jsonencode(T,'PrettyPrint',true)); fclose(fid);
fprintf('wrote %s\n', outF);
end

% =========================================================================
function [n,R] = chk(n, R, name, pass)
if pass, tag='PASS'; else, tag='FAIL'; n=n+1; end
fprintf('  [%s] %s\n', tag, name);
R{end+1} = struct('name',name,'pass',logical(pass));
end

function repo = local_repo()
here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(fileparts(fileparts(here)))));
end

function s = local_git(repo, args)
% --no-pager and GIT_PAGER=cat: under system() git's stdout can look like a
% terminal, and the pager then blocks forever waiting for a key.
[st,s] = system(sprintf('GIT_PAGER=cat git --no-pager -C "%s" %s 2>&1', repo, args));
if st ~= 0, s = ''; end
end

function cfg = local_candCfg(repo, nelx, nely) %#ok<INUSL>
info = olhoffcurrent_preset();
cfg = olh.config.resolve(info.upstreamPreset, ...
    'domain.mesh.nelx', nelx, 'domain.mesh.nely', nely, ...
    'move.continuation.signal','stageExhaustion', 'stop.rule','stageExhaustion', ...
    'runtime.maxOuter', 400, 'runtime.singleThread', true, ...
    'runtime.diagnostics', false, 'runtime.verbose', false);
end

% ---- synthetic design trajectories --------------------------------------
function [RHO, amp, mv] = local_traj(NE, tol, rho0, v, n, kind, big, small, switchAt, switchKind)
%LOCAL_TRAJ  A synthetic design trajectory with a prescribed dynamical regime.
%   'alternate'  cancelling, amplitude `big`     -> Branch A
%   'coherent'   converging, amplitude `small`   -> Branch B
%   'none'       coherent,   amplitude `big`     -> neither
%   'mixed'      alternate for 100 iterations, then coherent
%   switchAt/switchKind optionally change the regime from that iteration on.
if nargin < 7 || isempty(big),   big   = 2.0*tol;  end
if nargin < 8 || isempty(small), small = 0.20*tol; end
if nargin < 9,  switchAt   = Inf; end
if nargin < 10, switchKind = ''; end
RHO = zeros(NE,n); amp = zeros(n,1); mv = 0.04*ones(n,1);
r = rho0*ones(NE,1);
for k = 1:n
    kd = kind;
    if k >= switchAt && ~isempty(switchKind), kd = switchKind; end
    switch kd
        case 'alternate', a = big;   sgn = (-1)^k;
        case 'coherent',  a = small; sgn = 1;
        case 'none',      a = big;   sgn = 1;
        case 'mixed'
            if k <= 100, a = big; sgn = (-1)^k; else, a = small; sgn = 1; end
        otherwise, error('local_traj:kind','unknown kind %s', kd);
    end
    d = sgn*a*v;
    r = r + d; RHO(:,k) = r; amp(k) = norm(d);
end
end

function [ex, RHO] = local_synth(NE, tol, rho0, v, n, kind, a)
[RHO, amp, ~] = local_traj(NE, tol, rho0, v, n, kind, a, a);
%#ok<*AGROW>
ex = [];
for k = 1:n
    ex = olh.move.exhaustion(ex, NE, tol, k, RHO(:,k), local_unit(NE, amp(k)), rho0);
end
end

function d = local_unit(NE, a)
d = zeros(NE,1); d(1) = a;    % any vector whose 2-norm is a: amp reads norm(drho)
end

% ---- a faithful replay of olhoffSolve's controller wiring ---------------
function out = local_replay(cfg, NE, tol, rho0, v, n, kind, a, varargin)
p = inputParser(); p.addParameter('stalledBeta', false);
p.addParameter('switchAt', Inf); p.addParameter('switchKind', '');
p.addParameter('altAmp', []); p.parse(varargin{:});
% `a` is the amplitude of `kind`; altAmp is the amplitude the switched-to regime
% needs when the two regimes sit on opposite sides of tol.
alt = p.Results.altAmp; if isempty(alt), alt = a; end
[RHO, amp, ~] = local_traj(NE, tol, rho0, v, n, kind, alt, a, ...
                           p.Results.switchAt, p.Results.switchKind);
hist = struct('N',[],'beta',[],'dxNorm2',[],'move',[]);
mvState = []; mv = zeros(n,1); stageSeq = zeros(n,1); conv = false(n,1);
descentIters = []; declIters = [];
for k = 1:n
    [mv(k), mvState] = olh.move.limit(cfg, k, hist, mvState);          % TOP
    if k>1 && mvState.stage > stageSeq(k-1)
        descentIters(end+1) = k; %#ok<AGROW>
    end
    stageSeq(k) = mvState.stage;
    hist.N(k) = 2;  hist.move(k) = mv(k);  hist.dxNorm2(k) = amp(k);
    if p.Results.stalledBeta, hist.beta(k) = 1e4; else, hist.beta(k) = 1e4*(1+0.05*k); end
    wasDecl = ~isempty(mvState.ex) && mvState.ex.declared;
    mvState.ex = olh.move.exhaustion(mvState.ex, NE, tol, k, RHO(:,k), ...
                                     local_unit(NE, amp(k)), rho0);     % BOTTOM
    if ~wasDecl && mvState.ex.declared, declIters(end+1) = k; end %#ok<AGROW>
    conv(k) = mvState.ex.declared && stageSeq(k) >= numel(cfg.move.levels);
    if conv(k), break; end
end
% would production's beta stall have fired at all?
W = cfg.move.continuation.window; betaFire = false;
b = hist.beta;
if numel(b) >= 2*W
    w2 = mean(b(end-W+1:end)); w1 = mean(b(end-2*W+1:end-W));
    betaFire = (w2-w1)/max(abs(w1),eps) < cfg.move.continuation.tolerance;
end
ci = find(conv(1:k),1);
out = struct('move',mv(1:k),'stageSeq',stageSeq(1:k),'conv',conv(1:k), ...
             'ex',mvState.ex,'stageStarts',mvState.stageStarts, ...
             'descentIters',descentIters,'declIters',declIters, ...
             'betaWouldFire',betaFire,'convIter',ci,'n',k);
end

function ok = local_terminalCase(cfg, NE, tol, rho0, v, a, expectConv)
%LOCAL_TERMINALCASE  Reach the last rung on a Branch-B regime, then hold the
%   requested regime there and ask whether the run may stop.
small = 0.20*tol;
% pass 1: where does the last rung begin?
rep = local_replay(cfg, NE, tol, rho0, v, 900, 'coherent', small);
if numel(rep.stageStarts) < numel(cfg.move.levels), ok = false; return; end
s4 = rep.stageStarts(end);
% pass 2: from s4 on, either keep E true (coherent, small) or make it false
if expectConv
    rep2 = local_replay(cfg, NE, tol, rho0, v, 900, 'coherent', small);
    ok = ~isempty(rep2.convIter) && rep2.stageSeq(end) == numel(cfg.move.levels) && ...
         rep2.move(end) == 0.005 && rep2.convIter == rep2.ex.declIter && ...
         rep2.convIter >= s4 + 38;
else
    rep2 = local_replay(cfg, NE, tol, rho0, v, 900, 'coherent', small, ...
                        'switchAt', s4, 'switchKind', 'none', 'altAmp', a);
    ok = isempty(rep2.convIter) && rep2.stageSeq(end) == numel(cfg.move.levels) && ...
         rep2.move(end) == 0.005 && rep2.n == 900;
end
end

% ---- online vs frozen offline -------------------------------------------
function [ok, detail] = local_equiv(RHO, amp, mv, rho0, NE)
tol = 0.05*sqrt(NE/3200);
n = size(RHO,2);
ex = [];
for k = 1:n
    ex = olh.move.exhaustion(ex, NE, tol, k, RHO(:,k), local_unit(NE, amp(k)), rho0);
end
dyn = dr_dyn(RHO, mv, rho0);
per = struct('move', mv, 'l2', amp);
B   = tb_branches(per, dyn, NE);
okA = isequaln(logical(ex.A(:)), logical(B.Apred(:)));
okB = isequaln(logical(ex.B(:)), logical(B.Bpred(:)));
if isnan(B.event)
    okE = ~ex.declared;
    detail = sprintf('noEvent(decl=%d)', ex.declared);
else
    okE = ex.declared && ex.declIter == B.event + B.P - 1 && ...
          ex.declBegin == B.event && strcmp(ex.declBranch, B.branch);
    detail = sprintf('%s@%d==%d+%d', B.branch, ex.declIter, B.event, B.P-1);
end
ok = okA && okB && okE;
detail = sprintf('%s%s', detail, local_flag(okA,okB,okE));
end

function t = local_fun(txt, name)
%LOCAL_FUN  The text of a local function, from its header to the next one.
i = strfind(txt, ['function s = ' name]);
if isempty(i), t = ''; return; end
rest = txt(i(1):end);
j = strfind(rest(2:end), sprintf('\nfunction '));
if isempty(j), t = rest; else, t = rest(1:j(1)); end
end

function s = local_flag(a,b,e)
s = ''; if ~a, s=[s ' A!']; end, if ~b, s=[s ' B!']; end, if ~e, s=[s ' E!']; end
end
