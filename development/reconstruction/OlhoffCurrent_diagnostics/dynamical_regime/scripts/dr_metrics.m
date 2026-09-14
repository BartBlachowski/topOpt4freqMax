function dr_metrics(A)
%DR_METRICS  Emit METRICS.json (Phases J, K, L, M).
here = fileparts(mfilename('fullpath')); study = fileparts(here);
T = A.T; keys = fieldnames(T);
M = struct();
M.study = 'dynamical_regime';
M.repo_head_at_task_start = '7154d8201e9defb06d0d758da866c3769c07179a';
M.impl_tree_sha256 = 'c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c';
M.matlab = '25.2.0.3042426 (R2025b) Update 1';
M.threads = 1;
M.classifier = struct('persistence_P',20,'period2_rule','median q2 < 1 AND median cos < 0', ...
                      'coherent_rule','median q2 > 1.5 AND median cos > 0.25','W',10);
M.runA = A.runA; M.runB = A.runB;
M.prefix = A.prefix;

for i = 1:numel(keys)
    t = T.(keys{i}); p = t.per; d = t.dyn; n = numel(p.move);
    s = struct();
    s.label = t.label; s.nOuter = n;
    s.events = t.ev;
    s.final = struct('Mnd',p.Mnd(end),'gray',p.gray(end),'mid',p.mid(end), ...
                     'omega1',p.omega1(end),'volume',p.volume(end),'move',p.move(end));
    s.regime = struct('finalLabel',t.C.labelFinal,'fracPERIOD2',t.C.fracPERIOD2, ...
                      'fracCOHERENT',t.C.fracCOHERENT, ...
                      'period2Onset',t.C.onset.PERIOD2,'coherentOnset',t.C.onset.COHERENT);
    s.undefined = struct('q2',sum(d.undefQ2),'cosTheta',sum(d.undefCos));
    % state at every move descent (Phase L)
    dd = t.ev.moveDescents; tab = [];
    for k = dd
        tab(end+1,:) = [k, p.move(max(k-1,1)), p.move(k), p.betaStallRel(k), ...
            p.Mnd(k), p.gray(k), p.mid(k), p.omega1(k), d.d1(k), d.d2(k), ...
            d.q2(k), d.cosT(k), d.net_ratio(k), d.boundFrac(k), d.revFrac(k)]; %#ok<AGROW>
    end
    s.descentTableCols = {'iter','moveBefore','moveAfter','betaStallRel','Mnd','gray','mid', ...
                          'omega1','d1','d2','q2','cosTheta','net_ratio','boundFrac','revFrac'};
    s.descentTable = tab;
    % state at period-2 onset (Phase K4 / M)
    k2 = t.C.onset.PERIOD2;
    if ~isnan(k2)
        s.atPeriod2Onset = struct('iter',k2,'Mnd',p.Mnd(k2),'gray',p.gray(k2), ...
            'mid',p.mid(k2),'omega1',p.omega1(k2),'move',p.move(k2), ...
            'q2',d.q2(k2),'cosTheta',d.cosT(k2),'net_ratio',d.net_ratio(k2), ...
            'remaining_Mnd_pct',100*(p.Mnd(k2)-p.Mnd(end))/(p.Mnd(1)-p.Mnd(end)), ...
            'remaining_L1_pct',100*sum(abs(t.RHO(:,end)-t.RHO(:,k2)))/sum(abs(t.RHO(:,end)-t.RHO(:,1))));
    else
        s.atPeriod2Onset = [];
    end
    % robustness: with saturated elements removed, at the first descent
    kf = t.ev.firstDescent;
    if ~isnan(kf)
        s.atFirstDescent = struct('iter',kf,'q2',d.q2(kf),'q2_unsat',d.q2_unsat(kf), ...
            'cosTheta',d.cosT(kf),'cosTheta_unsat',d.cosT_unsat(kf), ...
            'net_ratio',d.net_ratio(kf),'net_ratio_unsat',d.net_ratio_unsat(kf), ...
            'boundFrac',d.boundFrac(kf),'revFrac',d.revFrac(kf), ...
            'Mnd',p.Mnd(kf),'omega1',p.omega1(kf),'move',p.move(kf), ...
            'q2_L1norm',d.q2_1(kf),'net_ratio_L1norm',d.net_ratio1(kf), ...
            'regimeLabel',char(t.C.label(kf)));
    else
        s.atFirstDescent = [];
    end
    M.(keys{i}) = s;
end
txt = jsonencode(M,'PrettyPrint',true);
fid = fopen(fullfile(study,'METRICS.json'),'w'); fwrite(fid,txt); fclose(fid);
fprintf('[dr_metrics] METRICS.json written (%d bytes)\n', numel(txt));
end
