function A = tb_analyze()
%TB_ANALYZE  Phases 9-14.  Post hoc only; no solver is invoked.
here  = fileparts(mfilename('fullpath')); study = fileparts(here);
DIAG  = fileparts(study);
prev  = load(fullfile(DIAG,'fixedmove_400_dynamics','evidence','fm_analysis.mat'));
D     = load(fullfile(study,'runs','runD_240x30.mat'));
A = prev.A;

A.T.f240 = struct('label','240x30 fixed move 0.04 (WITHHELD)', ...
                  'per',D.out.per,'dyn',D.out.per,'RHO',D.RHO);
A.T.f240.C = dr_classify(A.T.f240.dyn, 20);
A.runD = struct('status',D.out.status,'nOuter',D.out.nOuter,'wall_s',D.out.wall_s, ...
    'innerTotal',D.out.innerTotal,'omega1',D.out.omega(1),'Mnd_final',D.out.Mnd_final, ...
    'volume_final',D.out.volume_final,'cfgHash',D.out.cfgHash, ...
    'prodTol',D.out.prodTol,'nativeStopIter',D.out.nativeStopIter,'NE',D.out.NE);

% ---- apply the FROZEN predicates to all four fixed-move arms -----------
keys = {'f160','f240','f320','f400'}; NEs = [3200 7200 12800 20000];
A.branch = struct();
for i = 1:numel(keys)
    t = A.T.(keys{i});
    B = tb_branches(t.per, t.dyn, NEs(i));
    p = t.per; d = t.dyn; n = numel(p.move); k = B.event;
    s = struct('mesh',keys{i},'NE',NEs(i),'tol',B.tol,'kA',B.kA,'kB',B.kB, ...
               'event',k,'branch',B.branch,'bothFired',B.bothFired,'nOuter',n);
    % native stop replay on this arm
    kN = NaN;
    for j = 2:n, if p.l2(j) < B.tol && p.move(j)==p.move(j-1), kN = j; break; end, end
    s.nativeStop = kN;
    s.betaStallFirst = local_first(p.betaStallFires);
    if ~isnan(k)
        s.tail = n - k;
        s.Mnd = p.Mnd(k); s.omega1 = p.omega1(k); s.gray = p.gray(k);
        s.mid = p.mid(k); s.volume = p.volume(k);
        s.cosT = d.cosT(k); s.cosT_unsat = d.cosT_unsat(k);
        s.net_path = d.net_ratio(k); s.net_path_unsat = d.net_ratio_unsat(k);
        s.medcos = B.medcos(k); s.mednet = B.mednet(k);
        s.amp_l2 = p.l2(k); s.amp_rms = p.l2(k)/sqrt(NEs(i));
        s.maxAbs_over_move = p.ratio(k); s.boundFrac = d.boundFrac(k);
        s.remUseful   = 100*(p.Mnd(k)-p.Mnd(end))/(p.Mnd(1)-p.Mnd(end));
        s.postRelImp  = 100*(p.Mnd(k)-min(p.Mnd(k:end)))/p.Mnd(k);
        s.Mnd_best_after = min(p.Mnd(k:end));
        s.Mnd_best_at    = k-1+find(p.Mnd(k:end)==min(p.Mnd(k:end)),1);
        s.Mnd_end = p.Mnd(end);
        s.omega1_end = p.omega1(end);
        s.omega1_best_after = max(p.omega1(k:end));
        s.omega1_relpct_after = 100*(max(p.omega1(k:end))/p.omega1(k)-1);
        s.gray_end = p.gray(end); s.mid_end = p.mid(end);
        s.rhoL1_after = sum(abs(t.RHO(:,end)-t.RHO(:,k)))/NEs(i);
        % P5: is the event carried by a saturated minority?
        w = k:min(n,k+B.P-1);
        s.medcos_unsat_window = median(d.cosT_unsat(w),'omitnan');
        s.medboundFrac_window = median(d.boundFrac(w),'omitnan');
    end
    % terminal state (P6)
    w = max(2,n-100):n-1;
    s.term_maxAbs_over_move = median(p.ratio(w));
    s.term_l2 = median(p.l2(w));
    s.term_cosT = median(d.cosT(w),'omitnan');
    s.term_net_path = median(d.net_ratio(w),'omitnan');
    s.term_boundFrac = median(d.boundFrac(w),'omitnan');
    s.term_isCancelling = s.term_cosT < 0;
    s.term_isConverged  = (s.term_l2 < B.tol) && (s.term_cosT > 0);
    s.term_thirdRegime  = ~s.term_isCancelling && ~s.term_isConverged;
    A.branch.(keys{i}) = s;
    A.T.(keys{i}).B = B;
end

save(fullfile(study,'evidence','tb_analysis.mat'),'A','-v7.3');
b = A.branch.f240;
fprintf('[tb_analyze] 240x30: kA=%s kB=%s -> event=%s branch=%s (nativeStop=%s, status=%s)\n', ...
    mat2str(b.kA), mat2str(b.kB), mat2str(b.event), b.branch, mat2str(b.nativeStop), A.runD.status);
if ~isnan(b.event)
    fprintf('             Mnd=%.4f remUseful=%.3f%% postRelImp=%.2f%% tail=%d boundFrac=%.5f\n', ...
        b.Mnd, b.remUseful, b.postRelImp, b.tail, b.boundFrac);
end
fprintf('             terminal: maxAbs/mv=%.4f cos=%.3f net=%.3f cancelling=%d converged=%d third=%d\n', ...
    b.term_maxAbs_over_move, b.term_cosT, b.term_net_path, b.term_isCancelling, b.term_isConverged, b.term_thirdRegime);
fprintf('[tb_analyze] saved evidence/tb_analysis.mat\n');
end
function k = local_first(v), k = find(v,1); if isempty(k), k = NaN; end, end
