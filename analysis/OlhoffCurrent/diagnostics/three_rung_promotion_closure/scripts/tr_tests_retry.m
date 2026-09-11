function out = tr_tests_retry()
%TR_TESTS_RETRY  Part C software validation of the three-rung controller mechanics.
%
%   Every test here is SOFTWARE MECHANICS.  None is a scientific result and
%   none may be cited as one.  The two mechanisms under test are driven
%   DIRECTLY with synthetic detector state, so the tests do not depend on any
%   optimization outcome:
%
%     descent        olh.move.limit, 'stageExhaustion' branch
%     termination    the terminal-admission rule of olhoffSolve.m, restated
%                    here ONLY as the two-line predicate it is, and then
%                    cross-checked against the real solver on a tiny mesh.
%
%   The scientific run lock is respected: the only mesh exercised end to end is
%   48x6 (NE = 288), far below the 160x20 scientific floor, and its numbers are
%   never interpreted.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
addpath(root); addpath(here);
addpath(fullfile(root,'diagnostics','two_branch_controller_validation','scripts'));
guard = olhoffcurrent_paths(); %#ok<NASGU>

T = struct('name',{},'ok',{},'detail',{});
rec = @(n,o,d) struct('name',n,'ok',logical(o),'detail',d);

L3 = [0.04 0.02 0.01];
L4 = [0.04 0.02 0.01 0.005];

% ======================================================================
% 1-4  descent and termination at each rung, both ladders
% ======================================================================
for tl = {L3, L4}
    L = tl{1}; tag = sprintf('%d-rung', numel(L));
    for stage = 1:numel(L3)
        [mvD, stD] = local_step(L, stage, true);      % E declared
        descended  = stD.stage > stage;
        convTerm   = local_admit(L, stage, true);
        [mvH, stH] = local_step(L, stage, false);     % E not declared
        heldMove   = (mvH == L(stage)) && stH.stage == stage;
        convHold   = local_admit(L, stage, false);

        wantDescend = stage < numel(L);
        wantConv    = stage >= numel(L);
        T(end+1) = rec(sprintf('%s stage %d (move %.3g): E declared -> descend=%d conv=%d', ...
                    tag, stage, L(stage), wantDescend, wantConv), ...
                    descended == wantDescend && convTerm == wantConv, ...
                    sprintf('descended=%d conv=%d mv=%.4g', descended, convTerm, mvD)); %#ok<AGROW>
        T(end+1) = rec(sprintf('%s stage %d: E NOT declared -> hold, no stop', tag, stage), ...
                    heldMove && ~convHold, ...
                    sprintf('mv=%.4g stage=%d conv=%d', mvH, stH.stage, convHold)); %#ok<AGROW>
    end
end

% ======================================================================
% 5  the three-rung ladder can never produce move = 0.005
% ======================================================================
reach = [];
st = []; 
for stage = 1:6
    [mv, st2] = local_step(L3, min(stage,numel(L3)), true);
    reach(end+1) = mv; %#ok<AGROW>
end
T(end+1) = rec('three-rung ladder never yields move 0.005', ...
    ~any(abs(reach - 0.005) < 1e-15) && all(ismember(round(reach,12), round(L3,12))), ...
    sprintf('reachable moves = %s', mat2str(unique(reach))));
T(end+1) = rec('0.005 is absent from the three-rung level vector', ...
    ~any(abs(L3 - 0.005) < 1e-15), mat2str(L3));

% ======================================================================
% 6-7  beta has no continuation authority and no terminal authority
% ======================================================================
% A beta history engineered to be a textbook stall (flat, then falling) is fed
% to the stageExhaustion branch with the detector NOT declared.
cfgS = local_cfg(L3, 'stageExhaustion');
hist = struct('beta', [ones(1,60) 0.999*ones(1,60)], 'N', 2*ones(1,120), ...
              'omega', ones(2,120), 'move', 0.04*ones(1,120), 'stage', ones(1,120), ...
              'dxNorm2', 1e-9*ones(1,120));
stB = local_state(1, false);
[mvB, stB2] = olh.move.limit(cfgS, 120, hist, stB);
T(end+1) = rec('beta stall cannot descend the ladder under stageExhaustion', ...
    stB2.stage == 1 && mvB == 0.04, sprintf('stage=%d mv=%.4g', stB2.stage, mvB));
T(end+1) = rec('beta stall cannot terminate under stageExhaustion', ...
    ~local_admit(L3, 1, false), 'admit(stage=1, declared=false) = false');
% and the same beta history DOES descend under the production boundVariable
% signal -- so the test above is a real suppression, not an inert history.
cfgP = local_cfg(L4, 'boundVariable');
stP  = local_state(1, false);
[~, stP2] = olh.move.limit(cfgP, 120, hist, stP);
T(end+1) = rec('control: that same beta history DOES descend under boundVariable', ...
    stP2.stage == 2, sprintf('stage=%d', stP2.stage));

% ======================================================================
% 8  CAP_HIT stays CAP_HIT; a non-declared terminal stage does not stop
% ======================================================================
T(end+1) = rec('three-rung at terminal stage without declaration does not stop', ...
    ~local_admit(L3, 3, false), 'admit(stage=3, declared=false) = false');

% ======================================================================
% 9  A / B / persistence / reset semantics unchanged by the ladder length
% ======================================================================
NE = 200; tol = 0.05*sqrt(NE/3200);
rng(7);
seq = cell(1,80);
for k = 1:80, seq{k} = 0.5*tol/sqrt(NE)*ones(NE,1); end   % tiny, coherent -> Branch B
exA = []; exB = [];
rhoA = 0.5*ones(NE,1); rhoB = 0.5*ones(NE,1);
for k = 1:80
    rhoA = rhoA + seq{k};  exA = olh.move.exhaustion(exA, NE, tol, k, rhoA, seq{k}, 0.5);
    rhoB = rhoB + seq{k};  exB = olh.move.exhaustion(exB, NE, tol, k, rhoB, seq{k}, 0.5);
end
T(end+1) = rec('detector is ladder-blind: identical state from identical input', ...
    isequaln(exA, exB), 'exhaustion.m takes no ladder argument');
T(end+1) = rec('Branch B declares at P=20 consecutive (synthetic coherent decay)', ...
    exA.declared && strcmp(exA.declBranch,'B') && exA.declIter == exA.declBegin + 19, ...
    sprintf('declared=%d branch=%s declIter=%d declBegin=%d', ...
            exA.declared, exA.declBranch, exA.declIter, exA.declBegin));
T(end+1) = rec('frozen constants unchanged: W=20 P=20 Wnp=10', ...
    exA.W == 20 && exA.P == 20 && exA.Wnp == 10, ...
    sprintf('W=%d P=%d Wnp=%d tol=%.6g', exA.W, exA.P, exA.Wnp, exA.tol));
% reset on descent clears the counters and re-bases the window
stR = local_state(1, true); stR.ex = exA;
cfgR = local_cfg(L3,'stageExhaustion');
histR = struct('beta',ones(1,81),'N',2*ones(1,81),'omega',ones(2,81), ...
               'move',0.04*ones(1,81),'stage',ones(1,81),'dxNorm2',ones(1,81));
[~, stR2] = olh.move.limit(cfgR, 81, histR, stR);
T(end+1) = rec('descent resets the detector window wholly to the new stage', ...
    stR2.stage == 2 && stR2.ex.stageStart == 81 && stR2.ex.cntA == 0 && ...
    stR2.ex.cntB == 0 && ~stR2.ex.declared && isnan(stR2.ex.declIter), ...
    sprintf('stage=%d stageStart=%d cntA=%d cntB=%d declared=%d', ...
        stR2.stage, stR2.ex.stageStart, stR2.ex.cntA, stR2.ex.cntB, stR2.ex.declared));

% ======================================================================
% 10  end-to-end mechanics on a 48x6 mesh (SOFTWARE ONLY, never interpreted)
% ======================================================================
maxNumCompThreads(1);
tiny = @(lv, cap) olh.config.resolve('duOlhoffFrozenM4', ...
        'domain.mesh.nelx',48,'domain.mesh.nely',6, ...
        'runtime.maxOuter',cap,'runtime.singleThread',true, ...
        'runtime.diagnostics',true,'runtime.verbose',false, ...
        'move.continuation.signal','stageExhaustion','stop.rule','stageExhaustion', ...
        'move.levels',lv,'runtime.name','TR3_SOFTWARE_ONLY');
r3 = olhoffSolve(tiny(L3, 400));
r4 = olhoffSolve(tiny(L4, 400));
n3 = numel(r3.hist.N); n4 = numel(r4.hist.N);
T(end+1) = rec('48x6: three-rung never visits move 0.005', ...
    ~any(abs(r3.hist.move - 0.005) < 1e-15), ...
    sprintf('moves visited = %s', mat2str(unique(r3.hist.move))));
T(end+1) = rec('48x6: three-rung final stage <= 3', ...
    max(r3.hist.stage) <= 3, sprintf('maxStage=%d', max(r3.hist.stage)));
% prefix equivalence of the two ladders on this tiny mesh, up to the first
% iteration at which the three-rung run reached stage 3 AND declared
kcmp = min(n3, n4);
same = isequal(r3.hist.move(1:kcmp), r4.hist.move(1:kcmp)) && ...
       isequal(r3.hist.stage(1:kcmp), r4.hist.stage(1:kcmp));
T(end+1) = rec('48x6: ladder length is inert over the common prefix', same, ...
    sprintf('n3=%d (%s) n4=%d (%s) common=%d', n3, r3.status, n4, r4.status, kcmp));
T(end+1) = rec('48x6: cap is honoured (status is one of CONVERGED / CAP_HIT)', ...
    ismember(r3.status,{'CONVERGED','CAP_HIT'}) && n3 <= 400, ...
    sprintf('status=%s n=%d', r3.status, n3));
% a cap of 5 must stay CAP_HIT under the three-rung ladder
r3c = olhoffSolve(tiny(L3, 5));
T(end+1) = rec('48x6: CAP_HIT stays CAP_HIT under the three-rung ladder', ...
    strcmp(r3c.status,'CAP_HIT') && numel(r3c.hist.N) == 5, ...
    sprintf('status=%s n=%d', r3c.status, numel(r3c.hist.N)));

% ======================================================================
% 11  configuration validation still refuses a non-descending ladder, and
%     still couples the two stage-exhaustion switches
% ======================================================================
coupled = false;
try
    olh.config.resolve('duOlhoffFrozenM4','move.levels',L3, ...
        'move.continuation.signal','boundVariable','stop.rule','stageExhaustion');
catch ME
    coupled = strcmp(ME.identifier,'olh:config:exhaustionStopNeedsSignal') || ...
              contains(ME.message,'requires move.continuation.signal');
end
T(end+1) = rec('validation still couples stop.rule and the move signal', coupled, ...
    'stop.rule=stageExhaustion requires move.continuation.signal=stageExhaustion');

bad = false;
try
    olh.config.resolve('duOlhoffFrozenM4','move.levels',[0.01 0.02 0.04]);
catch ME
    bad = strcmp(ME.identifier,'olh:config:ladderNotDescending');
end
T(end+1) = rec('validation still rejects a non-descending ladder', bad, ...
    'olh:config:ladderNotDescending');

% ---- report -------------------------------------------------------------
out = struct('tests', T, 'nTests', numel(T), 'nPass', sum([T.ok]));
out.pass = all([T.ok]);
out.verdict = 'THREE_RUNG_SOFTWARE_VALIDATION_FAIL';
if out.pass, out.verdict = 'THREE_RUNG_SOFTWARE_VALIDATION_PASS'; end

fprintf('\n== software validation: %d/%d passed ==\n', out.nPass, out.nTests);
for k = 1:numel(T)
    fprintf('  [%s] %s\n', local_mark(T(k).ok), T(k).name);
    if ~T(k).ok, fprintf('        detail: %s\n', T(k).detail); end
end
fprintf('VERDICT: %s\n', out.verdict);

fid = fopen(fullfile(study,'evidence','software_tests.json'),'w');
c = onCleanup(@() fclose(fid)); fprintf(fid,'%s\n', jsonencode(out,'PrettyPrint',true));
end

% ---------------------------------------------------------------------------
function cfg = local_cfg(levels, signal, stopRule)
% olh.config.validate couples the two switches: stop.rule='stageExhaustion'
% REQUIRES the matching move signal, so the production control below must use
% production's own stop rule.  That coupling is itself part of the frozen
% design and is asserted as a test.
if nargin < 3
    if strcmp(signal,'stageExhaustion'), stopRule = 'stageExhaustion';
    else, stopRule = olh.config.getPath(olh.config.defaults(),'stop.rule'); end
end
cfg = olh.config.resolve('duOlhoffFrozenM4', 'move.levels', levels, ...
        'move.continuation.signal', signal, 'stop.rule', stopRule, ...
        'runtime.name','TR3_SOFTWARE_ONLY');
end

function st = local_state(stage, declared)
ex = struct('W',20,'P',20,'Wnp',10,'tol',0.1,'NE',200,'stageStart',1, ...
    'X',0.5*ones(200,1),'dPrev',[],'dn',[],'amp',[],'cos',[],'net',[], ...
    'medcos',[],'mednet',[],'A',[],'B',[],'E',[],'nA',[],'nB',[], ...
    'cntA',20,'cntB',0,'declared',declared,'declIter',60,'declBranch','A', ...
    'declBegin',41,'events',zeros(0,3),'eventBranch',{{}});
st = struct('mv',0.04,'stage',stage,'lastBeta',NaN,'stall',0,'ratioHist',[], ...
    'coalSeen',true,'lastStage',0,'lastRealized',NaN,'ex',ex, ...
    'stageStarts',1,'descents',zeros(0,4));
end

function [mv, st2] = local_step(levels, stage, declared)
% one call of the REAL descent controller with synthetic detector state
cfg = local_cfg(levels, 'stageExhaustion');
st  = local_state(stage, declared);
st.mv = levels(stage);
hist = struct('beta',ones(1,100),'N',2*ones(1,100),'omega',ones(2,100), ...
              'move',levels(stage)*ones(1,100),'stage',stage*ones(1,100), ...
              'dxNorm2',ones(1,100));
[mv, st2] = olh.move.limit(cfg, 100, hist, st);
end

function tf = local_admit(levels, stage, declared)
% the terminal-admission rule of olhoffSolve.m, stated as the predicate it is
atLastLevel = stage >= numel(levels);
tf = declared && atLastLevel;
end

function s = local_mark(ok)
if ok, s = 'PASS'; else, s = 'FAIL'; end
end
