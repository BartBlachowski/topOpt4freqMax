function analyze_baseline_dynamics()
%ANALYZE_BASELINE_DYNAMICS Offline diagnosis of frozen k=0:1600 controls.
maxNumCompThreads(1);
here=fileparts(mfilename('fullpath')); repo=fileparts(fileparts(here));
Q=readtable(fullfile(repo,'analysis','olhoff_fixed_budget_audit','quality_budget_curves.csv'),'TextType','string');
meshes=[160 20;240 30;320 40;400 50]; bands=[0.025 0.01 0.005];
eventRows={}; summaryRows={};
for im=1:4
    nelx=meshes(im,1); nely=meshes(im,2); meshId=string(sprintf('%dx%d',nelx,nely));
    T=Q(Q.mesh==meshId,:); loss=T.common_raw_E1_loss_to_k1600;
    src=fullfile(repo,'analysis','olhoff_native_convergence','results',sprintf('development_%dx%d.mat',nelx,nely));
    S=load(src,'res'); res=S.res; X=double(res.telemetry.rho_snapshots);
    moveFrac=[NaN res.telemetry.move_bound_fraction]'; dRms=[NaN res.telemetry.d_rms]';
    movingStrong=[NaN res.telemetry.moving_fraction(end,:)]';
    lp=[NaN res.telemetry.lp_flag]'; finite=[true res.telemetry.finite_ok]';
    gap=T.native_gap12; N=T.native_N; omega=T.native_omega1;
    firsts=zeros(1,3); persists=zeros(1,3); exitsCount=zeros(1,3);
    for ib=1:3
        c=loss<=bands(ib); [firsts(ib),persists(ib)]=crossings(c);
        exits=find(diff(c)==-1); exitsCount(ib)=numel(exits);
        for j=1:numel(exits)
            k=exits(j); re=find(c(k+2:end),1,'first');
            if isempty(re), reentry=NaN; duration=1601-k;
            else, reentry=k+re; duration=reentry-k; end
            stop=1600; if isfinite(reentry), stop=reentry-1; end
            peak=max(loss(k+1:stop+1));
            [cosRev,signRev,activeTurn,binaryPrev,binaryTerminal,rawConn,binConn,rawComp,binComp,volRes] = ...
                state_event_metrics(X,k,res.cfg.rhomin);
            qModal=min(k+1,1600);
            eventRows(end+1,:)={meshId,bands(ib),k,reentry,duration,peak, ... %#ok<AGROW>
                gap(k+1),N(k+1),logical(res.telemetry.mode_order_changed(qModal)), ...
                moveFrac(k+1),movingStrong(k+1),dRms(k+1),cosRev,signRev,activeTurn, ...
                abs(omega(k+1)-omega(k))/max(abs(omega(k)),eps),binaryPrev,binaryTerminal, ...
                rawConn,binConn,rawComp,binComp,volRes,lp(k+1),finite(k+1)};
        end
    end
    c1=loss<=0.01; exits1=find(diff(c1)==-1);
    mature=(0:1600)'>=firsts(2); control=mature & c1; control(exits1+1)=false;
    exitIx=exits1+1;
    revCos=NaN(1601,1); revSign=revCos; active=revCos; binPrev=revCos;
    for k=max(2,firsts(2)):1600
        [revCos(k+1),revSign(k+1),active(k+1),binPrev(k+1)] = update_metrics(X,k,res.cfg.rhomin);
    end
    gapCondition=N==2 & gap<=0.01; [gapFirst,gapPersist]=crossings(gapCondition);
    gapReopen=nnz(diff(gapCondition)==-1); nChanges=nnz(diff(N(max(1,firsts(2)+1):end))~=0);
    lag2=sqrt(mean((X(:,3:end)-X(:,1:end-2)).^2,1));
    allConnected=true; allBinaryConnected=true;
    for k=exits1(:)'
        [~,~,~,~,~,rc,bc]=state_event_metrics(X,k,res.cfg.rhomin);
        allConnected=allConnected&&rc; allBinaryConnected=allBinaryConnected&&bc;
    end
    summaryRows(end+1,:)={meshId,firsts(1),persists(1),firsts(2),persists(2), ... %#ok<AGROW>
        firsts(3),persists(3),exitsCount(1),exitsCount(2),exitsCount(3), ...
        gapFirst,gapPersist,gapReopen,nChanges,mean(moveFrac(exitIx),'omitnan'), ...
        mean(moveFrac(control),'omitnan'),mean(movingStrong(exitIx),'omitnan'), ...
        mean(movingStrong(control),'omitnan'),mean(dRms(exitIx),'omitnan'), ...
        mean(dRms(control),'omitnan'),mean(revCos(exitIx),'omitnan'), ...
        mean(revCos(control),'omitnan'),mean(revSign(exitIx),'omitnan'), ...
        mean(revSign(control),'omitnan'),mean(active(exitIx),'omitnan'), ...
        mean(active(control),'omitnan'),mean(binPrev(exitIx),'omitnan'), ...
        mean(binPrev(control),'omitnan'),mean(lag2(end-199:end)),max(lag2(end-199:end)), ...
        nnz(res.telemetry.lp_flag~=1),nnz(~res.telemetry.finite_ok),allConnected,allBinaryConnected, ...
        "PENDING_SCIENTIFIC_REVIEW","PENDING"};
end

E=cell2table(eventRows,'VariableNames',{'mesh','quality_band','exit_iteration','reentry_iteration', ...
    'excursion_duration','peak_loss','gap12_at_exit','N_at_exit','mode_order_changed', ...
    'move_bound_fraction','strongly_moving_fraction','density_update_rms','successive_update_cosine', ...
    'sign_reversal_fraction','active_set_turnover','native_objective_relative_change','binary_turnover_from_previous', ...
    'binary_turnover_to_baseline_terminal','raw_connected','binary_connected','raw_components', ...
    'binary_components','volume_residual','lp_flag','finite_ok'});
writetable(E,fullfile(here,'raw','baseline_excursions.csv'));
S=cell2table(summaryRows,'VariableNames',{'mesh','first_2p5','persistent_2p5','first_1p0', ...
    'persistent_1p0','first_0p5','persistent_0p5','exits_2p5','exits_1p0','exits_0p5', ...
    'first_bimodal_gap1','persistent_bimodal_gap1','gap1_reopenings','N_changes_after_first_1p0', ...
    'mean_move_bound_fraction_at_1p0_exit','mean_move_bound_fraction_good_control', ...
    'mean_strongly_moving_fraction_at_exit','mean_strongly_moving_fraction_good_control', ...
    'mean_density_rms_at_exit','mean_density_rms_good_control','mean_update_cosine_at_exit', ...
    'mean_update_cosine_good_control','mean_sign_reversal_at_exit','mean_sign_reversal_good_control', ...
    'mean_active_set_turnover_at_exit','mean_active_set_turnover_good_control', ...
    'mean_binary_turnover_at_exit','mean_binary_turnover_good_control','late200_mean_lag2_rms', ...
    'late200_max_lag2_rms','lp_failures','nonfinite_iterations','raw_connected_at_all_exits', ...
    'binary_connected_at_all_exits','stabilization_hypothesis','diagnosed_mechanism'});
for i=1:height(S)
    switch S.mesh(i)
        case "160x20"
            S.stabilization_hypothesis(i)="STABILIZATION_HYPOTHESIS_PLAUSIBLE";
            S.diagnosed_mechanism(i)="shallow same-basin excursions with persistent move saturation; exit association is modest";
        case "240x30"
            S.stabilization_hypothesis(i)="STABILIZATION_HYPOTHESIS_PLAUSIBLE";
            S.diagnosed_mechanism(i)="quality already persistent but late updates are move-saturated and near period-two with strong direction reversal";
        case "320x40"
            S.stabilization_hypothesis(i)="STABILIZATION_HYPOTHESIS_SUPPORTED";
            S.diagnosed_mechanism(i)="same-basin coherent move-saturated bursts including a large late quality spike; no solver or connectivity event";
        case "400x50"
            S.stabilization_hypothesis(i)="STABILIZATION_HYPOTHESIS_SUPPORTED";
            S.diagnosed_mechanism(i)="one-step same-basin quality spikes coincide with 41-51 percent move saturation; no persistent modal or topology failure";
    end
end
writetable(S,fullfile(here,'baseline_late_dynamics.csv'));
fprintf('Wrote %d excursion events and %d mesh summaries.\n',height(E),height(S));
end

function [first,persist]=crossings(c)
ix=find(c,1,'first'); if isempty(ix),first=NaN;else,first=ix-1;end
bad=find(~c); lastBad=max([-1;bad(:)-1]); if lastBad>=numel(c)-1,persist=NaN;else,persist=lastBad+1;end
end

function [cosRev,signRev,activeTurn,binaryPrev,binaryTerminal,rawConn,binConn,rawComp,binComp,volRes]=state_event_metrics(X,k,rhomin)
x=X(:,k+1); prev=X(:,max(1,k)); terminal=X(:,end); d=x-prev;
if k>=2
    dprev=prev-X(:,k-1); cosRev=(d'*dprev)/max(norm(d)*norm(dprev),eps);
    both=abs(d)>1e-4 & abs(dprev)>1e-4; signRev=mean(sign(d(both))~=sign(dprev(both)),'omitnan');
else, cosRev=NaN; signRev=NaN; end
a=(x<=rhomin+0.01)|(x>=0.99); ap=(prev<=rhomin+0.01)|(prev>=0.99); activeTurn=mean(a~=ap);
xb=volume_binary(x); xbp=volume_binary(prev); xbt=volume_binary(terminal);
binaryPrev=mean(xb~=xbp); binaryTerminal=mean(xb~=xbt);
[rawConn,rawComp]=connectivity(x>=0.5); [binConn,binComp]=connectivity(xb);
volRes=mean(x)-0.5;
end

function [cosRev,signRev,activeTurn,binaryPrev]=update_metrics(X,k,rhomin)
x=X(:,k+1); prev=X(:,k); d=x-prev;
if k>=2
    dprev=prev-X(:,k-1); cosRev=(d'*dprev)/max(norm(d)*norm(dprev),eps);
    both=abs(d)>1e-4 & abs(dprev)>1e-4; signRev=mean(sign(d(both))~=sign(dprev(both)),'omitnan');
else, cosRev=NaN; signRev=NaN; end
a=(x<=rhomin+0.01)|(x>=0.99); ap=(prev<=rhomin+0.01)|(prev>=0.99); activeTurn=mean(a~=ap);
binaryPrev=mean(volume_binary(x)~=volume_binary(prev));
end

function xb=volume_binary(x)
n=round(0.5*numel(x)); [~,o]=sortrows([-x(:),(1:numel(x))'],[1 2]); xb=false(size(x)); xb(o(1:n))=true;
end

function [connected,nComp]=connectivity(v)
% Infer nely from the 8:1 aspect meshes used in this audit.
n=numel(v); nely=round(sqrt(n/8)); nelx=round(n/nely); B=reshape(v,nely,nelx); seen=false(size(B)); lab=zeros(size(B)); nComp=0;
for r=1:nely
 for c=1:nelx
  if ~B(r,c)||seen(r,c),continue,end
  nComp=nComp+1; qr=zeros(nnz(B),1); qc=qr; h=1;t=1;qr(1)=r;qc(1)=c;seen(r,c)=true;
  while h<=t
   rr=qr(h);cc=qc(h);h=h+1;lab(rr,cc)=nComp;
   nb=[rr-1 cc;rr+1 cc;rr cc-1;rr cc+1];
   for j=1:4
    r2=nb(j,1);c2=nb(j,2);
    if r2>=1&&r2<=nely&&c2>=1&&c2<=nelx&&B(r2,c2)&&~seen(r2,c2),t=t+1;qr(t)=r2;qc(t)=c2;seen(r2,c2)=true;end
   end
  end
 end
end
L=unique(lab(:,1));R=unique(lab(:,end));L(L==0)=[];R(R==0)=[];connected=~isempty(intersect(L,R));
end
