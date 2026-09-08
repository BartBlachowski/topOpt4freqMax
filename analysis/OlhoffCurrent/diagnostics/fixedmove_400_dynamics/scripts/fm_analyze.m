function A = fm_analyze()
%FM_ANALYZE  Phases 7-17.  Purely post hoc; no solver is invoked.
%
%   Reuses the preceding dynamical_regime analysis object (hash-valid) so the
%   160x20 / 320x40 / 400x50-production quantities are literally the same
%   numbers that study reported, and adds the new 400x50 fixed-move arm.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
DIAG  = fileparts(study);
DR    = fullfile(DIAG,'dynamical_regime');

prev = load(fullfile(DR,'evidence','dr_analysis.mat'));   % A.T = f160,p160,p320,f320,p400
C    = load(fullfile(study,'runs','runC_400x50.mat'));    % the new arm
A    = prev.A;
A.T.f400 = struct('label','400x50 fixed move 0.04 (RUN C)', ...
                  'per',C.out.per,'dyn',C.out.per,'RHO',C.RHO);
A.T.f400.C  = dr_classify(A.T.f400.dyn, 20);
p = A.T.f400.per;
d = find([false; p.move(2:end) < p.move(1:end-1)]);
A.T.f400.ev = struct('nOuter',numel(p.move), ...
    'firstBetaStall',first(p.betaStallFires),'moveDescents',d(:).', ...
    'firstDescent',NaN,'period2Onset',A.T.f400.C.onset.PERIOD2, ...
    'coherentOnset',A.T.f400.C.onset.COHERENT,'finalLabel',A.T.f400.C.labelFinal);
A.runC = struct('status',C.out.status,'nOuter',C.out.nOuter,'wall_s',C.out.wall_s, ...
    'innerTotal',C.out.innerTotal,'omega1',C.out.omega(1),'Mnd_final',C.out.Mnd_final, ...
    'volume_final',C.out.volume_final,'cfgHash',C.out.cfgHash, ...
    'prodTol',C.out.prodTol,'nativeStopIter',C.out.nativeStopIter);

% ---------- Phase 10: common prefix vs the retained production arm -------
PA = load(fullfile(DR,'runs','runA_400x50.mat'));
nP = min(size(PA.RHO,2), size(C.RHO,2));
dmax = 0; firstDiff = NaN;
for k = 1:nP
    dd = max(abs(C.RHO(:,k)-PA.RHO(:,k)));
    if dd > 0 && isnan(firstDiff), firstDiff = k; end
    dmax = max(dmax, dd);
end
n137 = 137;
d137 = 0;
for k = 1:n137, d137 = max(d137, max(abs(C.RHO(:,k)-PA.RHO(:,k)))); end
q = PA.out.per;
A.prefix = struct('reference','dynamical_regime/runs/runA_400x50.mat', ...
    'bitwise_1_137', d137==0, 'maxAbsRhoDiff_1_137', d137, ...
    'firstDifferingIter', firstDiff, ...
    'expectedFirstDifferingIter', 138, ...
    'firstDiffAsExpected', isequaln(firstDiff,138), ...
    'maxAbsOmega1Diff_1_137', max(abs(p.omega1(1:n137)-q.omega1(1:n137))), ...
    'maxAbsOmega2Diff_1_137', max(abs(p.omega2(1:n137)-q.omega2(1:n137))), ...
    'maxAbsMndDiff_1_137',    max(abs(p.Mnd(1:n137)-q.Mnd(1:n137))), ...
    'maxAbsVolDiff_1_137',    max(abs(p.volume(1:n137)-q.volume(1:n137))), ...
    'maxAbsBetaDiff_1_137',   max(abs(p.beta(1:n137)-q.beta(1:n137))), ...
    'prodMoveAt137', q.move(137), 'prodMoveAt138', q.move(138), ...
    'fixedMoveAt138', p.move(138));

% ---------- Phase 11: what production abandoned by descending at 138 -----
k138 = 138; kOn = A.T.f400.C.onset.PERIOD2;
if ~isnan(kOn)
    A.abandoned = struct('fromIter',k138,'toIter',kOn, ...
        'Mnd_from',p.Mnd(k138),'Mnd_to',p.Mnd(kOn),'dMnd',p.Mnd(kOn)-p.Mnd(k138), ...
        'gray_from',p.gray(k138),'gray_to',p.gray(kOn), ...
        'mid_from',p.mid(k138),'mid_to',p.mid(kOn), ...
        'omega1_from',p.omega1(k138),'omega1_to',p.omega1(kOn), ...
        'omega1_relpct',100*(p.omega1(kOn)/p.omega1(k138)-1), ...
        'rhoL1Distance',sum(abs(C.RHO(:,kOn)-C.RHO(:,k138)))/size(C.RHO,1), ...
        'rhoL2Distance',norm(C.RHO(:,kOn)-C.RHO(:,k138))/sqrt(size(C.RHO,1)), ...
        'prodFinalMnd',PA.out.Mnd_final,'prodFinalOmega1',PA.out.omega(1), ...
        'fixedFinalMnd',C.out.Mnd_final,'fixedFinalOmega1',C.out.omega(1));
else
    A.abandoned = [];
end

% ---------- Phase 9/12: useful evolution + confirmation tail -------------
A.useful = struct();
keys = {'f160','f320','f400'};
for i = 1:numel(keys)
    t = A.T.(keys{i}); pp = t.per; kk = t.C.onset.PERIOD2; n = numel(pp.Mnd);
    s = struct('onset',kk,'nOuter',n);
    if ~isnan(kk)
        s.tail = n - kk;
        s.Mnd_onset = pp.Mnd(kk); s.Mnd_end = pp.Mnd(end);
        s.remaining_Mnd_pct = 100*(pp.Mnd(kk)-pp.Mnd(end))/(pp.Mnd(1)-pp.Mnd(end));
        s.remaining_L1_pct  = 100*sum(abs(t.RHO(:,end)-t.RHO(:,kk)))/sum(abs(t.RHO(:,end)-t.RHO(:,1)));
        s.Mnd_change_after_onset      = pp.Mnd(end)-pp.Mnd(kk);
        s.Mnd_relchange_after_onset_pct = 100*(pp.Mnd(end)-pp.Mnd(kk))/pp.Mnd(kk);
        s.Mnd_best_after_onset        = min(pp.Mnd(kk:end));
        s.Mnd_best_improve_rel_pct    = 100*(pp.Mnd(kk)-min(pp.Mnd(kk:end)))/pp.Mnd(kk);
        s.gray_onset = pp.gray(kk); s.gray_end = pp.gray(end);
        s.mid_onset  = pp.mid(kk);  s.mid_end  = pp.mid(end);
        s.omega1_onset = pp.omega1(kk); s.omega1_end = pp.omega1(end);
        s.omega1_after_onset_relpct = 100*(pp.omega1(end)/pp.omega1(kk)-1);
        s.volume_onset = pp.volume(kk);
        s.boundFrac_onset = t.dyn.boundFrac(kk);
        s.cos_raw_onset = t.dyn.cosT(kk);  s.cos_unsat_onset = t.dyn.cosT_unsat(kk);
        s.net_raw_onset = t.dyn.net_ratio(kk); s.net_unsat_onset = t.dyn.net_ratio_unsat(kk);
        w = kk:min(n, kk+20);
        s.medcos_raw_onsetwin   = median(t.dyn.cosT(w),'omitnan');
        s.medcos_unsat_onsetwin = median(t.dyn.cosT_unsat(w),'omitnan');
        s.medboundFrac_onsetwin = median(t.dyn.boundFrac(w),'omitnan');
    end
    A.useful.(keys{i}) = s;
end

save(fullfile(study,'evidence','fm_analysis.mat'),'A','-v7.3');
fprintf('[fm_analyze] prefix 1..137 bitwise=%d (maxdiff %.3g); first diff at %s (expected 138)\n', ...
    A.prefix.bitwise_1_137, A.prefix.maxAbsRhoDiff_1_137, mat2str(A.prefix.firstDifferingIter));
fprintf('[fm_analyze] 400x50 fixed: onset=%s status=%s nativeStopWouldBe=%s\n', ...
    mat2str(A.T.f400.C.onset.PERIOD2), C.out.status, mat2str(C.out.nativeStopIter));
fprintf('[fm_analyze] saved evidence/fm_analysis.mat\n');
end

function k = first(v), k = find(v,1); if isempty(k), k = NaN; end, end
