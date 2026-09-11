function out = tr_prefix()
%TR_PREFIX  Parts E and F: exact prefix equivalence and terminal validation.
%
%   Compares the three-rung candidate against the frozen four-rung oracle over
%   iterations 1..S3 (= 352), on:
%
%     * the raw anchors  RHO[:,1:352]  and  omega(1:2,1:352)     (hashes)
%     * every numeric field of res.hist                          (bitwise)
%     * every telemetry column of the per-iteration CSV          (bitwise)
%     * the event structure S1/S2/S3 and their branches
%
%   The oracle CSV is taken from **git HEAD**, not the working tree, so the
%   pre-existing tOuter-only drift in the working copy cannot contaminate the
%   comparison.  Exclusions are exactly those declared in PREREGISTRATION.md
%   sec. 6 and nothing else.

S3 = 352;

% ---- the preregistered exclusion list, VERBATIM --------------------------
% PREREGISTRATION.md sec. 6 named exactly these three and said "no other
% exclusion is permitted".  It is NOT amended here.  It was enumerated against
% the 55-column telemetry CSV, in which tOuter is the only wall-clock column.
EXCLUDE_PREREG = {'tOuter', 'prodStageShadow', 'prodMoveShadow'};

% ---- wall-clock timing telemetry, by CATEGORY ----------------------------
% res.hist -- which the CSV does not carry in full -- holds three further
% timers that the preregistered list did not name:
%
%   hist.tEig   = toc(te)   olhoffSolve.m:215,405
%   hist.tGrad  = toc(tg)   olhoffSolve.m:287,406
%   hist.tInner = toc(ti)   olhoffSolve.m:351,407
%
% They are `toc` values and nothing in the solver reads them back (the only
% other mention of any of them is the hist initialization at line 62).  They
% are members of the category the task brief excludes by name --
% "tOuter / nondeterministic timing telemetry" -- and of the rule "do not use
% wall-clock timing as a bitwise reproducibility requirement".
%
% Note that tInner is the WALL TIME of the inner solve, not inner WORK.  Inner
% work is nInner / cumInner, which are compared and must match exactly.
%
% BOTH readings are computed and BOTH are reported.  Nothing is silently
% reclassified: see C320_PREFIX_EQUIVALENCE.md sec. 4.
TIMING = {'tOuter', 'tEig', 'tGrad', 'tInner'};
EXCLUDE = union(EXCLUDE_PREREG, TIMING);

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
addpath(root); addpath(here);
guard = olhoffcurrent_paths(); %#ok<NASGU>

newTraj = fullfile(root,'evidence','three_rung_promotion_validation_retry1', ...
                   'C320x40_three_rung_trajectory.mat');
oldTraj = fullfile(root,'evidence','two_branch_controller_validation', ...
                   'C320x40_trajectory.mat');
newCsv  = fullfile(study,'runs','C320x40_three_rung_iterations.csv');
oldCsv  = fullfile(study,'evidence','oracle_C320x40_iterations_HEAD.csv');

A = load(newTraj,'RHO','DRHO','hist','exh','meta');   % candidate, three-rung
B = load(oldTraj,'RHO','DRHO','hist','exh','meta');   % oracle,    four-rung

out = struct();
out.nOuterNew = size(A.RHO,2);
out.nOuterOld = size(B.RHO,2);
out.implTreeNew = A.meta.implTree;
out.implTreeOld = B.meta.implTree;
out.implTreeSame = strcmp(A.meta.implTree, B.meta.implTree);

% ======================================================================
% E1  the frozen raw anchors
% ======================================================================
EXP = struct('rho','b8c0f18d887c7f5a81ece452f0a4097a0cc1a67710a137d66e677fcfd78b5ba3', ...
             'omega','fd63608399edaede2c4f928a25d5478eacc3711f6d626ae770ad107e7f4c488d');
out.anchors = struct();
if out.nOuterNew >= S3
    out.anchors.rho   = local_vecHash(A.RHO(:,1:S3));
    out.anchors.omega = local_vecHash(A.hist.omega(1:2,1:S3));
else
    out.anchors.rho = '<run shorter than S3>'; out.anchors.omega = out.anchors.rho;
end
out.anchors.rhoExpected   = EXP.rho;
out.anchors.omegaExpected = EXP.omega;
out.anchors.rhoMatch      = strcmp(out.anchors.rho,   EXP.rho);
out.anchors.omegaMatch    = strcmp(out.anchors.omega, EXP.omega);
% and, independently, straight against the oracle container
out.anchors.rhoVsOracle   = isequal(A.RHO(:,1:min(S3,out.nOuterNew)), B.RHO(:,1:min(S3,out.nOuterNew)));
out.anchors.drhoVsOracle  = isequal(A.DRHO(:,1:min(S3,out.nOuterNew)), B.DRHO(:,1:min(S3,out.nOuterNew)));

% ======================================================================
% E2  every numeric field of hist, bitwise over 1..S3
% ======================================================================
fn = intersect(fieldnames(A.hist), fieldnames(B.hist));
out.hist = struct('field',{},'ok',{},'nDiff',{},'maxAbsDev',{},'excluded',{});
for k = 1:numel(fn)
    f = fn{k};
    a = A.hist.(f); b = B.hist.(f);
    if ~isnumeric(a) && ~islogical(a), continue; end
    if isempty(a) || isempty(b), continue; end
    if isvector(a), a = a(:).'; b = b(:).'; na = numel(a); nb = numel(b);
    else,           na = size(a,2); nb = size(b,2); end
    m = min([S3, na, nb]);
    if isrow(a) || iscolumn(a), av = a(1:m); bv = b(1:m);
    else,                       av = a(:,1:m); bv = b(:,1:m); end
    ex = any(strcmp(f, EXCLUDE));
    dv = double(av(:)) - double(bv(:));
    nd = sum(~(double(av(:)) == double(bv(:)) | (isnan(double(av(:))) & isnan(double(bv(:))))));
    out.hist(end+1) = struct('field',f,'ok',nd == 0,'nDiff',nd, ...
        'maxAbsDev', max([0; abs(dv(~isnan(dv)))]), 'excluded', ex); %#ok<AGROW>
end

% ======================================================================
% E3  every telemetry column, bitwise over 1..S3
% ======================================================================
Tn = readtable(newCsv); To = readtable(oldCsv);
cols = intersect(Tn.Properties.VariableNames, To.Properties.VariableNames, 'stable');
out.csv = struct('col',{},'ok',{},'nDiff',{},'maxAbsDev',{},'excluded',{});
for k = 1:numel(cols)
    c = cols{k};
    av = Tn.(c); bv = To.(c);
    m = min([S3, numel(av), numel(bv)]);
    av = double(av(1:m)); bv = double(bv(1:m));
    same = (av == bv) | (isnan(av) & isnan(bv));
    dv = av - bv;
    out.csv(end+1) = struct('col',c,'ok',all(same),'nDiff',sum(~same), ...
        'maxAbsDev', max([0; abs(dv(~isnan(dv)))]), ...
        'excluded', any(strcmp(c, EXCLUDE))); %#ok<AGROW>
end

incl    = out.csv(~[out.csv.excluded]);
inclH   = out.hist(~[out.hist.excluded]);
out.csvIncludedOk  = all([incl.ok]);
out.histIncludedOk = all([inclH.ok]);
out.excludedReport = out.csv([out.csv.excluded]);

% ======================================================================
% E4  event reproduction
% ======================================================================
out.descentsNew   = A.exh.descents;
out.descentsOld   = B.exh.descents;
out.branchesNew   = A.exh.eventBranch;
out.branchesOld   = B.exh.eventBranch;
out.stageStartsNew= A.exh.stageStarts(:).';
out.stageStartsOld= B.exh.stageStarts(:).';
out.declIterNew   = A.exh.terminalDeclIter;
out.declBranchNew = A.exh.terminalBranch;
out.declBeginNew  = A.exh.terminalDeclBegin;
out.declaredNew   = A.exh.terminalDeclared;

% S1/S2 come from the applied descents; S3 is the terminal declaration, which
% the three-rung run consumes as CONVERGENCE and therefore never records as a
% descent.  Both are checked.
d = A.exh.descents;
out.S1 = struct('declIter', NaN, 'branch', '', 'ok', false);
out.S2 = out.S1; out.S3 = out.S1;
if size(d,1) >= 1
    out.S1 = struct('declIter', d(1,3), 'branch', A.exh.eventBranch{1}, ...
                    'ok', d(1,3) == 274 && strcmp(A.exh.eventBranch{1},'A'));
end
if size(d,1) >= 2
    out.S2 = struct('declIter', d(2,3), 'branch', A.exh.eventBranch{2}, ...
                    'ok', d(2,3) == 313 && strcmp(A.exh.eventBranch{2},'B'));
end
% S3 is the TERMINAL declaration.  The three-rung run consumes it as
% convergence, so it is never recorded as a descent -- it appears in the
% solver's terminal* fields instead.  That asymmetry is the whole point of the
% candidate and is asserted, not worked around.
out.S3 = struct('declIter', A.exh.terminalDeclIter, 'branch', A.exh.terminalBranch, ...
                'ok', logical(A.exh.terminalDeclared) && A.exh.terminalDeclIter == 352 && ...
                      strcmp(A.exh.terminalBranch,'B') && A.exh.terminalDeclBegin == 333);
out.eventsOk = out.S1.ok && out.S2.ok && out.S3.ok && size(d,1) == 2;

% ======================================================================
% E5  verdict
% ======================================================================
anchorsOk = out.anchors.rhoMatch && out.anchors.omegaMatch && ...
            out.anchors.rhoVsOracle && out.anchors.drhoVsOracle;

% (a) the LITERAL reading: only the three fields PREREGISTRATION.md sec. 6 named
litH = out.hist(~ismember({out.hist.field}, EXCLUDE_PREREG));
litC = out.csv( ~ismember({out.csv.col},    EXCLUDE_PREREG));
out.prefixPassLiteral = anchorsOk && all([litH.ok]) && all([litC.ok]) && out.eventsOk;
out.literalFailures = [ {litH(~[litH.ok]).field}, {litC(~[litC.ok]).col} ];

% (b) the CATEGORY reading: wall-clock timing telemetry excluded as a class
out.prefixPassCategory = anchorsOk && out.histIncludedOk && out.csvIncludedOk && out.eventsOk;
out.categoryFailures = [ {out.hist(~[out.hist.ok] & ~[out.hist.excluded]).field}, ...
                         {out.csv( ~[out.csv.ok]  & ~[out.csv.excluded]).col} ];

% The brief's controlling rule is the category one.  It is applied here, and the
% divergence between the two readings is reported in full rather than resolved
% quietly.
out.prefixPass = out.prefixPassCategory;
out.timingOnlyDivergence = out.prefixPassCategory && ~out.prefixPassLiteral && ...
    all(ismember(out.literalFailures, TIMING));
if out.prefixPass, out.prefixVerdict = 'C320_THREE_RUNG_PREFIX_EQUIVALENCE_PASS';
else,              out.prefixVerdict = 'C320_THREE_RUNG_PREFIX_EQUIVALENCE_FAIL'; end

% ======================================================================
% F  terminal validation
% ======================================================================
rec = jsondecode(fileread(fullfile(study,'runs','C320x40_three_rung_record.json')));
h = A.hist; n = out.nOuterNew;
ORACLE = struct('omega1',166.42726927757769,'omega2',203.58101247948915, ...
   'gap12',0.22324312213489572,'volume',0.49999874883108736, ...
   'Mnd',12.940093529981478,'gray',0.15296874999999999,'mid',0.030156249999999999, ...
   'move',0.01,'stage',3,'multN',2,'nInner',19,'innerConv',1,'cumInner',6498);
got = struct('omega1',h.omega(1,n),'omega2',h.omega(2,n), ...
   'gap12',(h.omega(2,n)-h.omega(1,n))/h.omega(1,n), ...
   'volume',mean(A.RHO(:,n)), ...
   'Mnd',Tn.Mnd(n),'gray',Tn.gray(n),'mid',Tn.mid(n), ...
   'move',h.move(n),'stage',h.stage(n),'multN',h.N(n), ...
   'nInner',h.nInner(n),'innerConv',double(h.innerConv(n)),'cumInner',sum(h.nInner(1:n)));
f = fieldnames(ORACLE);
out.terminal = struct('field',{},'oracle',{},'got',{},'ok',{},'absDev',{});
for k = 1:numel(f)
    a = ORACLE.(f{k}); b = got.(f{k});
    out.terminal(end+1) = struct('field',f{k},'oracle',a,'got',b, ...
        'ok', a == b, 'absDev', abs(a-b)); %#ok<AGROW>
end
out.terminalStateOk = all([out.terminal.ok]);
out.status      = rec.status;
out.nOuter      = rec.nOuter;
out.innerNonConv= rec.innerNonConv;
out.capHit      = strcmp(rec.status,'CAP_HIT');
out.visitedMoves= unique(h.move(:)).';
out.touched0005 = any(abs(h.move - 0.005) < 1e-15);
out.maxStage    = max(h.stage);

out.terminationPass = strcmp(rec.status,'CONVERGED') && out.nOuter == S3 && ...
    ~out.touched0005 && out.maxStage == 3 && out.terminalStateOk && ...
    out.innerNonConv == 0 && ~out.capHit;
if out.terminationPass, out.terminationVerdict = 'C320_THREE_RUNG_TERMINATION_PASS';
else,                   out.terminationVerdict = 'C320_THREE_RUNG_TERMINATION_FAIL'; end

% ---- cost, against the oracle -------------------------------------------
out.cost = struct('outerFour', size(B.RHO,2), 'outerThree', out.nOuter, ...
    'innerFour', sum(B.hist.nInner), 'innerThree', rec.innerTotal);
out.cost.outerSaved = out.cost.outerFour - out.cost.outerThree;
out.cost.outerSavedPct = 100*out.cost.outerSaved/out.cost.outerFour;
out.cost.innerSaved = out.cost.innerFour - out.cost.innerThree;
out.cost.innerSavedPct = 100*out.cost.innerSaved/out.cost.innerFour;

fid = fopen(fullfile(study,'evidence','prefix_equivalence.json'),'w');
c = onCleanup(@() fclose(fid)); fprintf(fid,'%s\n', jsonencode(out,'PrettyPrint',true));

% ---- report -------------------------------------------------------------
fprintf('\n======== PART E : PREFIX EQUIVALENCE (1..%d) ========\n', S3);
fprintf('implTree same         : %d  (%s)\n', out.implTreeSame, out.implTreeNew);
fprintf('RHO[:,1:352] hash     : %d  %s\n', out.anchors.rhoMatch, out.anchors.rho);
fprintf('omega(1:2,1:352) hash : %d  %s\n', out.anchors.omegaMatch, out.anchors.omega);
fprintf('RHO  vs oracle bitwise: %d\n', out.anchors.rhoVsOracle);
fprintf('DRHO vs oracle bitwise: %d\n', out.anchors.drhoVsOracle);
fprintf('\nhist fields: %d compared, %d differing (excluded shown separately)\n', ...
    numel(inclH), sum(~[inclH.ok]));
for k=1:numel(out.hist)
    if ~out.hist(k).ok
        fprintf('   %-14s %-8s nDiff=%d maxAbsDev=%.3g\n', out.hist(k).field, ...
            local_tag(out.hist(k).excluded), out.hist(k).nDiff, out.hist(k).maxAbsDev);
    end
end
fprintf('\nCSV columns: %d compared, %d differing\n', numel(incl), sum(~[incl.ok]));
for k=1:numel(out.csv)
    if ~out.csv(k).ok
        fprintf('   %-18s %-8s nDiff=%d maxAbsDev=%.3g\n', out.csv(k).col, ...
            local_tag(out.csv(k).excluded), out.csv(k).nDiff, out.csv(k).maxAbsDev);
    end
end
fprintf('\nevents: S1 declIter=%d/%s ok=%d | S2 declIter=%d/%s ok=%d | S3 declIter=%d/%s ok=%d\n', ...
    out.S1.declIter, out.S1.branch, out.S1.ok, out.S2.declIter, out.S2.branch, out.S2.ok, ...
    out.S3.declIter, out.S3.branch, out.S3.ok);
fprintf('\n  (a) literal preregistered exclusion list : %d   failures: %s\n', ...
    out.prefixPassLiteral, strjoin(out.literalFailures, ' '));
fprintf('  (b) wall-clock timing excluded by class : %d   failures: %s\n', ...
    out.prefixPassCategory, strjoin(out.categoryFailures, ' '));
fprintf('  divergence is timing-only               : %d\n', out.timingOnlyDivergence);
fprintf('VERDICT: %s\n', out.prefixVerdict);

fprintf('\n======== PART F : TERMINATION ========\n');
fprintf('status=%s  nOuter=%d  maxStage=%d  innerNonConv=%d\n', ...
    out.status, out.nOuter, out.maxStage, out.innerNonConv);
fprintf('moves visited = %s   entered 0.005 = %d\n', mat2str(out.visitedMoves), out.touched0005);
for k=1:numel(out.terminal)
    t = out.terminal(k);
    fprintf('  %-10s oracle=%.17g  got=%.17g  ok=%d\n', t.field, t.oracle, t.got, t.ok);
end
fprintf('cost: outer %d -> %d (-%.2f%%)   inner %d -> %d (-%.2f%%)\n', ...
    out.cost.outerFour, out.cost.outerThree, out.cost.outerSavedPct, ...
    out.cost.innerFour, out.cost.innerThree, out.cost.innerSavedPct);
fprintf('VERDICT: %s\n', out.terminationVerdict);
end

function s = local_tag(ex)
if ex, s = '[EXCL]'; else, s = '[INCL]'; end
end

function h = local_vecHash(v)
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(typecast(double(v(:)),'uint8'));
d = typecast(md.digest(),'uint8');
h = lower(reshape(dec2hex(d,2).',1,[]));
end
