function tm_phaseA()
%TM_PHASEA  Phase A -- offline topology-maturity signal design.
%
%   Operates EXCLUSIVELY on retained, hash-valid raw density trajectories from
%   analysis/OlhoffCurrent/diagnostics/move_transition.  No solver is executed,
%   no configuration is resolved, nothing in +impl/ is touched.
%
%   ARM U is the fixed-move counterfactual (move held at ladder stage 1 = 0.04
%   until its own transition rule fires).  It is the trajectory that reveals how
%   much design evolution production discarded when it descended at kP.
%
%   Writes: evidence/phaseA_stats.mat, METRICS.json (partial), figures/.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
D     = '/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffCurrent/diagnostics';
W     = 10;                       % preregistered window, unchanged (A2)

meshes = {'160x20','320x40'};
kP     = [79 130];                % production first descent (= first ARM P/U divergence)
kM     = [600 214];               % mature fixed-move reference (ARM U own descent; 160 never descends)

A = struct();
for i = 1:numel(meshes)
    m = meshes{i};
    U = load(fullfile(D,'move_transition/runs',['armU_' m '.mat']));
    P = load(fullfile(D,'move_transition/runs',['armP_' m '.mat']));
    T = readmatrix(fullfile(D,'move_transition/runs',['armU_' m '_iterations.csv']));
    rho = U.RHO; NE = size(rho,1); nK = size(rho,2);
    mv  = T(:,9); Mnd = T(:,6); om1 = T(:,2); vol = T(:,5); stage = T(:,10);

    % --- verify the arms share a bitwise prefix ending exactly at kP ---------
    dcol = arrayfun(@(k) max(abs(P.RHO(:,k)-U.RHO(:,k))), 1:min(size(P.RHO,2),nK));
    kdiv = find(dcol>0,1);

    dstep = abs(diff(rho,1,2));

    % --- ground-truth remaining evolution -----------------------------------
    den  = sum(abs(rho(:,kM(i))-rho(:,1)));
    rem  = arrayfun(@(k) sum(abs(rho(:,kM(i))-rho(:,k)))/den, 1:nK);
    cum  = cumsum(sum(dstep,1));
    remPath = (cum(kM(i)-1)-cum(kP(i)-1))/cum(kM(i)-1);

    % --- regime diagnosis: 1-step vs 2-step displacement --------------------
    d1 = sum(abs(rho(:,3:end)  -rho(:,2:end-1)),1)/NE;
    d2 = sum(abs(rho(:,3:end)  -rho(:,1:end-2)),1)/NE;
    coherence2 = [NaN NaN d2./max(d1,eps)];      % ~2 coherent, <1 period-2 cycle

    % --- candidate family (A8) ----------------------------------------------
    C = struct();
    f0 = @() nan(nK,1);
    names = {'inst_max','inst_rms','DW_l1','DW_l2','CW_l1','CW_max','CW_q90', ...
             'RW','RW_end','coher','frac_half','med_DW','trim_DW','DW_unsat','fracNet25'};
    for n = 1:numel(names), C.(names{n}) = f0(); end
    C.inst_max(2:end) = max(dstep,[],1)';
    C.inst_rms(2:end) = (sqrt(sum(dstep.^2,1)/NE))';
    for k = (W+1):nK
        dd = abs(rho(:,k)-rho(:,k-W));                 % net (endpoint) motion
        cc = sum(dstep(:,(k-W):(k-1)),2);              % accumulated path motion
        sat = cc > 0.9*W*mv(k);
        C.DW_l1(k)=sum(dd)/NE;   C.DW_l2(k)=norm(dd)/sqrt(NE);
        C.CW_l1(k)=sum(cc)/NE;   C.CW_max(k)=max(cc);  C.CW_q90(k)=quantile(cc,0.90);
        C.RW(k)=C.CW_l1(k)/(W*mv(k));
        C.RW_end(k)=C.DW_l1(k)/(W*mv(k));
        C.coher(k)=C.DW_l1(k)/max(C.CW_l1(k),eps);
        C.frac_half(k)=mean(cc>0.5*W*mv(k));
        C.med_DW(k)=median(dd);  C.trim_DW(k)=trimmean(dd,20);
        C.fracNet25(k)=mean(dd>0.25*W*mv(k));
        if any(~sat), C.DW_unsat(k)=sum(dd(~sat))/sum(~sat); else, C.DW_unsat(k)=0; end
    end

    A.(['m' m]) = struct('mesh',m,'NE',NE,'nK',nK,'kP',kP(i),'kM',kM(i), ...
        'firstDivergence',kdiv,'rem',rem,'remAtkP',rem(kP(i)),'remPathAtkP',remPath, ...
        'move',mv,'Mnd',Mnd,'omega1',om1,'volume',vol,'stage',stage, ...
        'coherence2',coherence2,'C',C,'names',{names}, ...
        'Mnd_kP',Mnd(kP(i)),'Mnd_kM',Mnd(kM(i)), ...
        'om1_kP',om1(kP(i)),'om1_kM',om1(kM(i)));
end

save(fullfile(study,'evidence','phaseA_stats.mat'),'A','-v7.3');
tm_figures(A, fullfile(study,'figures'));
fprintf('Phase A analysis complete.\n');
end
