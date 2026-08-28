function analyze_native_development()
%ANALYZE_NATIVE_DEVELOPMENT Develop detector families on 240x30 only.
here=fileparts(mfilename('fullpath'));
resultDir=fullfile(here,'results');
s=load(fullfile(resultDir,'development_240x30.mat'),'res','identity');
if ~s.identity.passed
    error('analyze_native_development:Identity','Frozen identity gate did not pass.');
end
r=s.res; n=r.nOuter; w=r.hist.omega(1,:); R=double(r.telemetry.rho_snapshots);

rhoPhase=NaN(1,n); topoPhase=NaN(1,n);
for k=2:n
    d=R(:,k+1)-R(:,k-1);
    rhoPhase(k)=sqrt(mean(d.^2));
    topoPhase(k)=mean((R(:,k+1)>=0.5)~=(R(:,k-1)>=0.5));
end
objPhase=NaN(1,n);
objPhase(3:n)=abs(w(3:n)-w(1:n-2))./max(abs(w(3:n)),eps);
modeEvent=r.telemetry.mode_order_changed|r.telemetry.N_changed;
health=r.hist.innerConv&(r.telemetry.lp_flag==1)&r.telemetry.eig_ok& ...
    r.telemetry.finite_ok&~r.telemetry.eig_warning;

% Interpretable development grid.  NaN means that family does not use the
% corresponding stationarity signal; all families retain common safety guards.
name={'O_loose';'O_mid';'O_strict';'D_loose';'D_mid';'D_strict'; ...
      'H_balanced';'H_conservative'};
family={'objective';'objective';'objective';'design';'design';'design'; ...
        'hybrid';'hybrid'};
block=[20;20;30;NaN;NaN;NaN;20;30];
window=[20;40;60;20;40;60;40;60];
persistence=[10;15;20;10;15;20;20;25];
blockTol=[2e-4;1e-4;5e-5;NaN;NaN;NaN;1e-4;5e-5];
phaseTol=[2e-4;1e-4;5e-5;NaN;NaN;NaN;1e-4;5e-5];
rhoTol=[NaN;NaN;NaN;1.5e-3;1.25e-3;7.5e-4;1.25e-3;7.5e-4];
topoTol=[NaN;NaN;NaN;1e-3;7e-4;3e-4;7e-4;3e-4];
modalWindow=[40;40;60;40;40;60;40;60];
gapTol=1e-2*ones(size(window));

nC=numel(name); firstHit=NaN(nC,1); terminalObjLoss=NaN(nC,1);
terminalTopoTurnover=NaN(nC,1); passH50=false(nC,1);
passH100=false(nC,1); passH200=false(nC,1);
omega1Fire=NaN(nC,1); omega2Fire=NaN(nC,1); omega3Fire=NaN(nC,1);
gap12Fire=NaN(nC,1); NFire=NaN(nC,1); rmsFire=NaN(nC,1);
movingFractionFire=NaN(nC,1); densityL1=NaN(nC,1); densityL2=NaN(nC,1);
classification=repmat("NEVER_FIRES",nC,1);
condition=false(nC,n);
for c=1:nC
    raw=false(1,n);
    for k=1:n
        raw(k)=localCondition(k,char(family{c}),block(c),window(c), ...
            blockTol(c),phaseTol(c),rhoTol(c),topoTol(c),modalWindow(c), ...
            gapTol(c),w,objPhase,rhoPhase,topoPhase,r,modeEvent,health);
        if k>=persistence(c)
            condition(c,k)=all(raw(k-persistence(c)+1:k));
        end
    end
    q=find(condition(c,:),1,'first');
    if ~isempty(q)
        firstHit(c)=q;
        terminalObjLoss(c)=localTerminalObjectiveLoss(q,w);
        terminalTopoTurnover(c)=localTopologyTurnover(q,n,R);
        passH50(c)=localLabel(q,50,w,R,r,modeEvent,health);
        passH100(c)=localLabel(q,100,w,R,r,modeEvent,health);
        passH200(c)=localLabel(q,200,w,R,r,modeEvent,health);
        post=min(q+1,n); omega1Fire(c)=r.hist.omega(1,post);
        omega2Fire(c)=r.hist.omega(2,post); omega3Fire(c)=r.hist.omega(3,post);
        gap12Fire(c)=abs(omega2Fire(c)-omega1Fire(c))/omega1Fire(c);
        NFire(c)=r.hist.N(post); rmsFire(c)=r.telemetry.d_rms(q);
        movingFractionFire(c)=r.telemetry.moving_fraction(3,q);
        delta=R(:,q+1)-double(r.rho); densityL1(c)=mean(abs(delta));
        densityL2(c)=sqrt(mean(delta.^2));
        if ~(passH50(c)&&passH100(c)&&passH200(c)), classification(c)="FALSE_POSITIVE";
        elseif q>1400, classification(c)="TOO_LATE";
        else, classification(c)="TRUE_POSITIVE";
        end
    end
end

candidates=table(name,family,block,window,persistence,blockTol,phaseTol, ...
    rhoTol,topoTol,modalWindow,gapTol,firstHit,terminalObjLoss, ...
    terminalTopoTurnover,omega1Fire,omega2Fire,omega3Fire,gap12Fire,NFire, ...
    rmsFire,movingFractionFire,densityL1,densityL2,passH50,passH100,passH200, ...
    classification);
writetable(candidates,fullfile(resultDir,'native_convergence_candidates.csv'));

% Retrospective labels at common and detector-selected stop points.  These are
% deliberately unavailable to the online detector.
checkpoints=unique([200 300 400 500 600 800 1000 1200 1400 firstHit(isfinite(firstHit))']);
source=repmat("common_checkpoint",numel(checkpoints),1);
for i=1:numel(checkpoints)
    hitNames=name(firstHit==checkpoints(i));
    if ~isempty(hitNames), source(i)="candidate:"+strjoin(string(hitNames),'+'); end
end
falsePos=table(); row=0;
for i=1:numel(checkpoints)
    for H=[50 100 200]
        row=row+1; k=checkpoints(i);
        q=min(n,k+H); if mod(q-k,2)==1, q=q-1; end
        falsePos.iteration(row,1)=k;
        falsePos.source(row,1)=source(i);
        falsePos.horizon(row,1)=H;
        falsePos.label_pass(row,1)=localLabel(k,H,w,R,r,modeEvent,health);
        falsePos.terminal_objective_loss(row,1)=localTerminalObjectiveLoss(k,w);
        falsePos.future_topology_turnover(row,1)=localTopologyTurnover(k,q,R);
        falsePos.future_mode_event(row,1)=any(modeEvent(k+1:min(n,k+H)));
        falsePos.future_solver_failure(row,1)=any(~health(k+1:min(n,k+H)));
        loss=abs(falsePos.terminal_objective_loss(row));
        falsePos.pass_objective_1pct(row,1)=loss<=.01;
        falsePos.pass_objective_0p5pct(row,1)=loss<=.005;
        falsePos.pass_objective_0p25pct(row,1)=loss<=.0025;
        falsePos.pass_objective_0p1pct(row,1)=loss<=.001;
    end
end
writetable(falsePos,fullfile(resultDir,'native_convergence_false_positives.csv'));

features=table((1:n)',w',objPhase',rhoPhase',topoPhase',r.telemetry.d_rms', ...
    r.telemetry.moving_fraction(3,:)',r.telemetry.gaps_rel(1,:)',r.hist.N', ...
    modeEvent',health','VariableNames',{'iteration','omega1','objective_phase_recurrence', ...
    'rho_phase_rms','topology_phase_turnover','step_rms','moving_fraction_gt_1e3', ...
    'gap12_rel','multiplicity_N','mode_event','solver_health'});
writetable(features,fullfile(resultDir,'development_240x30_features.csv'));

fig=figure('Visible','off','Color','w','Position',[100 100 1100 800]);
tiledlayout(3,1,'TileSpacing','compact');
nexttile; plot(1:n,w,'k-'); ylabel('\omega_1'); grid on; hold on
for c=1:nC
    if isfinite(firstHit(c)), xline(firstHit(c),'--',name{c}); end
end
title('240x30 detector development (all candidates retrospective)');
nexttile; semilogy(1:n,objPhase,'Color',[0 .35 .8]); ylabel('|\omega_k-\omega_{k-2}|/\omega_k'); grid on
nexttile; semilogy(1:n,rhoPhase,'Color',[.8 .2 0]); hold on
semilogy(1:n,max(topoPhase,1e-8),'Color',[.2 .55 .2]);
ylabel('period-two design metrics'); xlabel('outer iteration'); grid on
legend('density RMS','binary turnover','Location','best');
exportgraphics(fig,fullfile(resultDir,'development_240x30_detector_features.png'),'Resolution',180);
close(fig);
disp(candidates);
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

function tf=localLabel(k,H,w,R,r,modeEvent,health)
n=numel(w); q=min(n,k+H);
if q-k<H, tf=false; return; end
if mod(q-k,2)==1, q=q-1; end
B=20; if k<2*B, tf=false; return; end
startMean=mean(w(k-B+1:k)); terminalMean=mean(w(end-B+1:end));
objTerminal=abs(startMean-terminalMean)/terminalMean;
centres=k:B:q; blockMeans=NaN(size(centres));
for i=1:numel(centres), blockMeans(i)=mean(w(centres(i)-B+1:centres(i))); end
futureObj=max(abs(blockMeans-startMean))/terminalMean;
topo=localTopologyTurnover(k,q,R);
future=(k+1):q;
tf=objTerminal<=1e-3 & futureObj<=1e-3 & topo<=5e-3 & ...
    ~any(modeEvent(future)) & all(health(future)) & all(r.hist.N(future)==2) & ...
    all(r.telemetry.gaps_rel(1,future)<=1e-2);
end

function x=localTerminalObjectiveLoss(k,w)
B=20; terminalMean=mean(w(end-B+1:end));
x=(terminalMean-mean(w(k-B+1:k)))/terminalMean;
end

function x=localTopologyTurnover(k,q,R)
if mod(q-k,2)==1, q=q-1; end
x=mean((R(:,k+1)>=0.5)~=(R(:,q+1)>=0.5));
end
