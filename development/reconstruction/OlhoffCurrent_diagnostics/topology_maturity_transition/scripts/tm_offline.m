function R = tm_offline()
%TM_OFFLINE  Phase A offline topology-maturity signal analysis.
%   Operates ONLY on retained, hash-valid trajectories from the
%   move_transition study (ARM P = production, ARM U = fixed move at stage 1).
%   No solver is run.  No configuration is changed.

D = '/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffCurrent/diagnostics';
meshes = {'160x20','320x40'};
kP     = [79 130];        % production first descent (verified: first P/U divergence)
R = struct();

for mi = 1:numel(meshes)
    m = meshes{mi};
    U = load(fullfile(D,'move_transition/runs',['armU_' m '.mat']));
    P = load(fullfile(D,'move_transition/runs',['armP_' m '.mat']));
    rhoU = U.RHO;  NE = size(rhoU,1);  nK = size(rhoU,2);

    % ---- ARM U move / stage from its telemetry CSV -------------------------
    T = readmatrix(fullfile(D,'move_transition/runs',['armU_' m '_iterations.csv']));
    moveU  = T(:,9);  stageU = T(:,10);  MndU = T(:,6);  om1U = T(:,2);
    kU = find(stageU > 1, 1);            % ARM U's own first descent
    if isempty(kU), kU = NaN; end

    % ---- maturity reference: the mature fixed-move state -------------------
    if isnan(kU), kM = nK; else, kM = kU; end

    % ---- remaining topology evolution at production first descent ----------
    dstep = abs(diff(rhoU,1,2));                 % NE x (nK-1)
    stepL1 = sum(dstep,1);                       % per-iteration L1 motion
    cum = cumsum(stepL1);
    accTot  = cum(kM-1);
    accAtkP = cum(kP(mi)-1);
    rem_path = (accTot-accAtkP)/accTot;
    ep_tot   = sum(abs(rhoU(:,kM)-rhoU(:,1)));
    ep_rem   = sum(abs(rhoU(:,kM)-rhoU(:,kP(mi))));
    rem_end  = ep_rem/ep_tot;

    fprintf('\n===== %s =====\n', m);
    fprintf('NE=%d  production first descent kP=%d  ARM U first descent kU=%s  kM=%d\n', ...
        NE, kP(mi), mat2str(kU), kM);
    fprintf('REMAINING evolution at kP:  path-length %.4f%%   endpoint %.4f%%\n', ...
        100*rem_path, 100*rem_end);
    fprintf('M_nd at kP = %.4f%%   M_nd at kM = %.4f%%\n', MndU(kP(mi)), MndU(kM));
    fprintf('omega1 at kP = %.6f   at kM = %.6f  (%+.3f%%)\n', ...
        om1U(kP(mi)), om1U(kM), 100*(om1U(kM)/om1U(kP(mi))-1));

    % ---- candidate statistics along ARM U ---------------------------------
    W = 10;
    S = struct();
    S.k     = (1:nK)';
    S.move  = moveU;
    S.Mnd   = MndU;
    S.inst_max = [NaN; max(dstep,[],1)'];
    S.inst_rms = [NaN; (sqrt(sum(dstep.^2,1)/NE))'];
    S.inst_l1n = [NaN; (sum(dstep,1)/NE)'];

    DW_l1 = nan(nK,1); DW_l2 = nan(nK,1); CW_l1 = nan(nK,1); CW_max = nan(nK,1);
    frac_half = nan(nK,1); q90 = nan(nK,1);
    for k = (W+1):nK
        dd = abs(rhoU(:,k)-rhoU(:,k-W));
        DW_l1(k) = sum(dd)/NE;
        DW_l2(k) = norm(dd)/sqrt(NE);
        cc = sum(dstep(:,(k-W):(k-1)),2);        % accumulated per element
        CW_l1(k) = sum(cc)/NE;
        CW_max(k) = max(cc);
        q90(k) = quantile(cc,0.90);
        frac_half(k) = mean(cc > 0.5*W*moveU(k));
    end
    S.DW_l1 = DW_l1;  S.DW_l2 = DW_l2;  S.CW_l1 = CW_l1;
    S.CW_max = CW_max; S.CW_q90 = q90; S.frac_half = frac_half;

    % move-normalized (dimensionless mean utilization of the step bound)
    S.RW      = CW_l1 ./ (W*moveU);       % mean elementwise utilization, in [0,1]
    S.RW_end  = DW_l1 ./ (W*moveU);       % coherent (net) utilization
    S.coher   = DW_l1 ./ max(CW_l1,eps);  % directedness in [0,1]

    R.(['m' m]) = struct('NE',NE,'kP',kP(mi),'kU',kU,'kM',kM, ...
        'rem_path',rem_path,'rem_end',rem_end,'S',S, ...
        'Mnd_kP',MndU(kP(mi)),'Mnd_kM',MndU(kM), ...
        'om1_kP',om1U(kP(mi)),'om1_kM',om1U(kM));

    fprintf('\n  candidate values AT kP=%d (ARM U, still at move=%.3f):\n', kP(mi), moveU(kP(mi)));
    f = {'inst_max','inst_rms','DW_l1','DW_l2','CW_l1','CW_max','CW_q90','RW','RW_end','coher','frac_half'};
    for i=1:numel(f)
        fprintf('    %-10s = %.6g\n', f{i}, S.(f{i})(kP(mi)));
    end
    if ~isnan(kU)
        fprintf('\n  candidate values AT kU=%d (ARM U mature descent):\n', kU);
        for i=1:numel(f)
            fprintf('    %-10s = %.6g\n', f{i}, S.(f{i})(kU));
        end
    end
end
end
