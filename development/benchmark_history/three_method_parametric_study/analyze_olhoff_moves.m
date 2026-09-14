function analyze_olhoff_moves()
%ANALYZE_OLHOFF_MOVES  WP6/WP7/WP11 analysis of the Olhoff move sweep.
%
%   Observer-only.  Every trajectory in raw/stage_a was run with the native
%   outer stop SUPPRESSED, so the full 1200-iteration record exists for every
%   move level and no detector decision ever altered a trajectory.  Detectors
%   are therefore evaluated retrospectively but computed future-blind: each
%   candidate condition at iteration k uses only history up to k.
%
%   Label definitions (localLabel/localCondition/localTopologyTurnover) are
%   carried over verbatim from analysis/olhoff_native_convergence so that the
%   present move sweep is measured on exactly the same yardstick as the frozen
%   move=0.005 negative result.

repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
study = fullfile(repo,'analysis','three_method_parametric_study');
addpath(study);
raw = fullfile(study,'raw','stage_a');
out = fullfile(study,'results');
if ~exist(out,'dir'), mkdir(out); end

files = dir(fullfile(raw,'olhoff_move_*.mat'));
[~,ord] = sort({files.name}); files = files(ord);

trajRows = {}; detRows = {}; fpRows = {};

for fi = 1:numel(files)
    S = load(fullfile(raw,files(fi).name)); r = S.record;
    if ~strcmp(r.status,'COMPLETED_OBSERVER')
        trajRows{end+1,1} = struct('run_id',r.run_id,'move',NaN,'status',r.status); %#ok<AGROW>
        continue;
    end
    n = r.n_iter;
    w = r.hist.omega(1,1:n);                 % native omega_1 trajectory
    R = r.telemetry.rho_snapshots;           % 7200 x (n+1), column j = state after iter j-1
    modeEvent = r.telemetry.mode_order_changed(1:n) | r.telemetry.N_changed(1:n);
    health = r.hist.innerConv(1:n) & r.telemetry.lp_flag(1:n)==1 & ...
             r.telemetry.eig_ok(1:n) & r.telemetry.finite_ok(1:n) & ...
             ~r.telemetry.eig_warning(1:n);

    % ---- phase metrics at lag 1 and lag 2 -------------------------------
    % Lag 2 is the period-two phase pairing that the move=0.005 trajectory
    % requires.  Lag 1 is the ordinary consecutive comparison, which is the
    % right pairing if a larger move genuinely settles rather than cycling.
    [objP1,rhoP1,topoP1] = phaseMetrics(w,R,1,n);
    [objP2,rhoP2,topoP2] = phaseMetrics(w,R,2,n);

    % ---- trajectory-level summary --------------------------------------
    row = struct();
    row.run_id = r.run_id;  row.method = 'Olhoff';
    row.nelx = r.nelx; row.nely = r.nely; row.move = r.move;
    row.n_iter_observed = n;
    row.wall_time_s = r.wall_time;
    row.t_iter_s = r.wall_time/n;
    row.eig_time_s = sum(r.hist.tEig(1:n));
    row.grad_time_s = sum(r.hist.tGrad(1:n));
    row.inner_time_s = sum(r.hist.tInner(1:n));
    row.eig_share = row.eig_time_s / max(sum(r.hist.tEig(1:n)+r.hist.tGrad(1:n)+r.hist.tInner(1:n)),eps);
    row.omega1_native = r.omega_native(1);
    row.omega2_native = r.omega_native(2);
    row.omega3_native = r.omega_native(3);
    row.gap12_rel_final = r.telemetry.gaps_rel(1,n);
    row.N_final = r.hist.N(n);
    row.N_frac_2_last200 = mean(r.hist.N(max(1,n-199):n)==2);
    row.max_gap12_last200 = max(r.telemetry.gaps_rel(1,max(1,n-199):n));
    row.last_modal_event = lastTrue(modeEvent);
    row.n_solver_failures = sum(~health);
    row.vol_final = r.hist.vol(n);
    row.vol_resid_rel = abs(r.hist.vol(n)-r.cfg.volfrac)/r.cfg.volfrac;
    % Native outer stop test max|drho| < tolOuter (WP6 salvageability check)
    row.dx_outer_min = min(r.hist.dxOuter(1:n));
    row.dx_outer_final = r.hist.dxOuter(n);
    row.tol_outer = r.cfg.tolOuter;
    row.tol_outer_ever_fires = any(r.hist.dxOuter(1:n) < r.cfg.tolOuter);
    row.tol_outer_first_fire = firstTrue(r.hist.dxOuter(1:n) < r.cfg.tolOuter);
    row.d_rms_final = r.telemetry.d_rms(n);
    row.moving_frac_final = r.telemetry.moving_fraction(3,n);   % 0.2*move threshold
    row.move_bound_frac_final = r.telemetry.move_bound_fraction(n);
    row.rho_phase_rms_lag1_last200 = max(rhoP1(max(1,n-199):n));
    row.rho_phase_rms_lag2_last200 = max(rhoP2(max(1,n-199):n));
    row.obj_phase_lag1_last200 = max(objP1(max(1,n-199):n));
    row.obj_phase_lag2_last200 = max(objP2(max(1,n-199):n));
    % WP7 validity classification of the terminal state.  Two bimodality
    % standards are reported: the strict 1% eigengap used by the frozen
    % move=0.005 audit (primary), and the method's own tolMult multiplicity
    % tolerance (disclosed sensitivity, looser).
    row.bimodal_strict_1pct = row.gap12_rel_final <= 0.01;
    row.bimodal_tolmult = row.gap12_rel_final <= r.cfg.tolMult;
    row.tol_mult = r.cfg.tolMult;
    con = connectivity4(double(R(:,n+1))>=0.5, r.cfg.nely, r.cfg.nelx);
    row.connected_final = con.left_right_connected;
    row.largest_component_fraction = con.largest_component_fraction;
    row.n_components_final = con.n_components;
    if row.n_solver_failures > 0
        row.validity = 'SOLVER_FAILURE';
    elseif ~row.connected_final
        row.validity = 'CONNECTIVITY_FAILURE';
    elseif ~row.bimodal_strict_1pct
        row.validity = 'STATIONARY_NOT_BIMODAL';
    else
        row.validity = 'BIMODAL_VALID';
    end
    row.break_reason = r.telemetry.break_reason;
    trajRows{end+1,1} = row; %#ok<AGROW>

    % ---- detector families ---------------------------------------------
    % Small transparent grid.  Families: objective-only, design-only, hybrid.
    % Lag 1 and lag 2.  Three tolerance levels each.  All observer-only.
    fams = {'objective','design','hybrid'};
    levels = struct( ...
        'name',       {'loose','mid','strict'}, ...
        'blockTol',   {1e-3,   3e-4,  1e-4}, ...
        'phaseTol',   {1e-3,   3e-4,  1e-4}, ...
        'rhoTol',     {5e-3,   2.5e-3,1.25e-3}, ...
        'topoTol',    {3e-3,   1.5e-3,7e-4}, ...
        'persist',    {10,     15,    20});
    B = 20; W = 40; MW = 40; gapTol = 1e-2;

    for lag = [1 2]
        if lag==1, objP=objP1; rhoP=rhoP1; topoP=topoP1;
        else,      objP=objP2; rhoP=rhoP2; topoP=topoP2; end
        for fa = 1:numel(fams)
            for lv = 1:numel(levels)
                L = levels(lv);
                cond = false(1,n);
                for k = 1:n
                    cond(k) = localCondition(k,fams{fa},B,W,L.blockTol,L.phaseTol, ...
                        L.rhoTol,L.topoTol,MW,gapTol,w,objP,rhoP,topoP,r,modeEvent,health);
                end
                fire = firstPersistent(cond,L.persist);
                d = struct();
                d.run_id = r.run_id; d.move = r.move;
                d.family = fams{fa}; d.lag = lag; d.level = L.name;
                d.persistence = L.persist;
                d.block_tol = L.blockTol; d.phase_tol = L.phaseTol;
                d.rho_tol = L.rhoTol; d.topo_tol = L.topoTol;
                d.eligible_fraction = mean(cond);
                d.fire_iter = fire;
                if isnan(fire)
                    d.classification = 'NEVER_FIRES';
                    d.H50 = 'NA'; d.H100 = 'NA'; d.H200 = 'NA';
                    d.omega1_at_fire = NaN; d.terminal_obj_loss = NaN;
                    d.gap12_at_fire = NaN; d.N_at_fire = NaN;
                    d.wall_time_at_fire_s = NaN;
                else
                    d.H50  = localLabel(fire,50, w,R,r,modeEvent,health);
                    d.H100 = localLabel(fire,100,w,R,r,modeEvent,health);
                    d.H200 = localLabel(fire,200,w,R,r,modeEvent,health);
                    d.classification = classifyFire(d.H50,d.H100,d.H200);
                    d.omega1_at_fire = w(fire);
                    d.terminal_obj_loss = localTerminalObjectiveLoss(fire,w);
                    d.gap12_at_fire = r.telemetry.gaps_rel(1,fire);
                    d.N_at_fire = r.hist.N(fire);
                    d.wall_time_at_fire_s = r.wall_time*fire/n;
                end
                detRows{end+1,1} = d; %#ok<AGROW>

                if ~isnan(fire)
                    for H = [50 100 200]
                        f = struct('run_id',r.run_id,'move',r.move,'family',fams{fa}, ...
                            'lag',lag,'level',L.name,'fire_iter',fire,'horizon',H, ...
                            'label_status',localLabel(fire,H,w,R,r,modeEvent,health), ...
                            'obj_dev_pct',100*futureObjDeviation(fire,H,w), ...
                            'topo_turnover_pct',100*localTopologyTurnover(fire,min(n,fire+H),R), ...
                            'modal_event_after',any(modeEvent(fire+1:min(n,fire+H))));
                        fpRows{end+1,1} = f; %#ok<AGROW>
                    end
                end
            end
        end
    end
    fprintf('analyzed %s (move=%g)\n',r.run_id,r.move);
end

writetable(struct2table(cell2mat(trajRows)), fullfile(out,'olhoff_trajectory_summary.csv'));
writetable(struct2table(cell2mat(detRows)),  fullfile(out,'olhoff_detector_grid.csv'));
writetable(struct2table(cell2mat(fpRows)),   fullfile(out,'false_convergence_events.csv'));
fprintf('wrote Olhoff analysis CSVs to %s\n',out);
end

% ---------------------------------------------------------------------------
function [objP,rhoP,topoP] = phaseMetrics(w,R,lag,n)
% Same-phase recurrence at the requested lag.  Column j+1 of R is the density
% state after iteration j, so state(k) is R(:,k+1).
objP = NaN(1,n); rhoP = NaN(1,n); topoP = NaN(1,n);
for k = lag+1:n
    objP(k) = abs(w(k)-w(k-lag))/max(abs(w(k)),eps);
    a = double(R(:,k+1)); b = double(R(:,k+1-lag));
    rhoP(k) = sqrt(mean((a-b).^2));
    topoP(k) = mean((a>=0.5)~=(b>=0.5));
end
objP(1:lag) = Inf; rhoP(1:lag) = Inf; topoP(1:lag) = Inf;
end

function tf=localCondition(k,family,B,W,blockTol,phaseTol,rhoTol,topoTol,MW,gapTol, ...
        w,objPhase,rhoPhase,topoPhase,r,modeEvent,health)
need=max([W MW 2*B],[],'omitnan');
if k<need, tf=false; return; end
ixW=k-W+1:k; ixM=k-MW+1:k;
common=all(r.hist.N(ixM)==2)&all(r.telemetry.gaps_rel(1,ixM)<=gapTol)& ...
    ~any(modeEvent(ixM))&all(health(ixM))& ...
    abs(r.hist.vol(k)-r.cfg.volfrac)/r.cfg.volfrac<=1e-8;
objective=true; design=true;
if strcmp(family,'objective') || strcmp(family,'hybrid')
    newMean=mean(w(k-B+1:k)); oldMean=mean(w(k-2*B+1:k-B));
    objective=abs(newMean-oldMean)/max(abs(newMean),eps)<=blockTol & ...
        max(objPhase(ixW))<=phaseTol;
end
if strcmp(family,'design') || strcmp(family,'hybrid')
    design=max(rhoPhase(ixW))<=rhoTol & max(topoPhase(ixW))<=topoTol;
end
tf=common&objective&design;
end

function st=localLabel(k,H,w,R,r,modeEvent,health)
% Returns 'PASS', 'FAIL' or 'CENSORED'.  CENSORED means the observation window
% ended before the horizon closed, so the look-ahead test could not be run --
% it is NOT evidence against the candidate and must never be reported as a
% false positive.
n=numel(w); q=min(n,k+H);
if q-k<H, st='CENSORED'; return; end
if mod(q-k,2)==1, q=q-1; end
B=20; if k<2*B, st='CENSORED'; return; end
startMean=mean(w(k-B+1:k)); terminalMean=mean(w(end-B+1:end));
objTerminal=abs(startMean-terminalMean)/terminalMean;
centres=k:B:q; blockMeans=NaN(size(centres));
for i=1:numel(centres), blockMeans(i)=mean(w(centres(i)-B+1:centres(i))); end
futureObj=max(abs(blockMeans-startMean))/terminalMean;
topo=localTopologyTurnover(k,q,R);
future=(k+1):q;
pass=objTerminal<=1e-3 & futureObj<=1e-3 & topo<=5e-3 & ...
    ~any(modeEvent(future)) & all(health(future)) & all(r.hist.N(future)==2) & ...
    all(r.telemetry.gaps_rel(1,future)<=1e-2);
if pass, st='PASS'; else, st='FAIL'; end
end

function x=futureObjDeviation(k,H,w)
n=numel(w); q=min(n,k+H); B=20;
if k<B, x=NaN; return; end
startMean=mean(w(k-B+1:k)); terminalMean=mean(w(end-B+1:end));
centres=k:B:q; bm=NaN(size(centres));
for i=1:numel(centres), bm(i)=mean(w(centres(i)-B+1:centres(i))); end
x=max(abs(bm-startMean))/terminalMean;
end

function x=localTerminalObjectiveLoss(k,w)
B=20; terminalMean=mean(w(end-B+1:end));
x=(terminalMean-mean(w(k-B+1:k)))/terminalMean;
end

function x=localTopologyTurnover(k,q,R)
if mod(q-k,2)==1, q=q-1; end
x=mean((R(:,k+1)>=0.5)~=(R(:,q+1)>=0.5));
end

function c=connectivity4(B,nely,nelx)
B=reshape(B,nely,nelx); visited=false(size(B)); labels=zeros(size(B));
sizes=[]; component=0;
for rr=1:nely
    for cc=1:nelx
        if ~B(rr,cc)||visited(rr,cc), continue; end
        component=component+1; qr=zeros(nnz(B),1); qc=qr;
        head=1; tail=1; qr(1)=rr; qc(1)=cc; visited(rr,cc)=true; count=0;
        while head<=tail
            r0=qr(head); c0=qc(head); head=head+1; count=count+1; labels(r0,c0)=component;
            nb=[r0-1 c0;r0+1 c0;r0 c0-1;r0 c0+1];
            for k=1:4
                r2=nb(k,1); c2=nb(k,2);
                if r2>=1&&r2<=nely&&c2>=1&&c2<=nelx&&B(r2,c2)&&~visited(r2,c2)
                    tail=tail+1; qr(tail)=r2; qc(tail)=c2; visited(r2,c2)=true;
                end
            end
        end
        sizes(component)=count; %#ok<AGROW>
    end
end
left=unique(labels(:,1)); left(left==0)=[];
right=unique(labels(:,end)); right(right==0)=[];
c=struct('n_components',component,'left_right_connected',~isempty(intersect(left,right)), ...
    'largest_component_fraction',max([0 sizes])/max(1,nnz(B)));
end

function k=firstPersistent(cond,p)
n=numel(cond); k=NaN;
run=0;
for i=1:n
    if cond(i), run=run+1; else, run=0; end
    if run>=p, k=i; return; end
end
end

function c=classifyFire(h50,h100,h200)
% A candidate is only TRUE if every horizon it could actually be tested at
% passed, and at least one horizon closed.  If no horizon closed the verdict
% is CENSORED, not a pass and not a false positive.
st={h50,h100,h200};
tested=st(~strcmp(st,'CENSORED'));
if isempty(tested), c='CENSORED_NO_CLOSED_HORIZON'; return; end
if any(strcmp(tested,'FAIL'))
    if strcmp(h50,'PASS'), c='FALSE_POSITIVE_DELAYED';
    else,                  c='FALSE_POSITIVE_IMMEDIATE'; end
    return;
end
if strcmp(h200,'PASS'), c='TRUE_ON_TRAJECTORY';
else,                   c='TRUE_BUT_HORIZON_LIMITED'; end
end

function k=firstTrue(v), k=find(v,1,'first'); if isempty(k), k=NaN; end, end
function k=lastTrue(v),  k=find(v,1,'last');  if isempty(k), k=NaN; end, end
