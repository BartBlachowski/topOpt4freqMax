function build_outputs()
%BUILD_OUTPUTS Derive classifications, curves, tables, and figures from evidence.
maxNumCompThreads(1);
here=fileparts(mfilename('fullpath')); repo=fileparts(fileparts(here));
pre=jsondecode(fileread(fullfile(here,'study_preregistration.json')));
C=readtable(fullfile(here,'checkpoint_metrics.csv'),'TextType','string');
M=readtable(fullfile(here,'modal_establishment.csv'),'TextType','string');
meshes=[[pre.meshes.nelx]' [pre.meshes.nely]'];

curveParts=cell(size(meshes,1),1); minRows={}; phaseRows={};
for im=1:size(meshes,1)
    nelx=meshes(im,1); nely=meshes(im,2); meshId=string(sprintf('%dx%d',nelx,nely));
    E=readtable(fullfile(here,'raw',sprintf('e1_raw_trajectory_%dx%d.csv',nelx,nely)), ...
        'TextType','string');
    src=fullfile(repo,'analysis','olhoff_native_convergence','results', ...
        sprintf('development_%dx%d.mat',nelx,nely));
    S=load(src,'res'); res=S.res; X=double(res.telemetry.rho_snapshots);
    native=aligned_native(res); gap=(native(:,2)-native(:,1))./native(:,1);
    N=[res.hist.N detect_n(res.omega,res.cfg)]';
    loop=res.hist.tEig+res.hist.tGrad+res.hist.tInner;
    cumulative=[0 cumsum(loop)]'; cumulativeEig=[0 cumsum(res.hist.tEig)]';
    ref=X(:,end); refOdd=X(:,end-1); xbRef=volume_binary(ref); xbRefOdd=volume_binary(refOdd);
    rmsTerminal=NaN(1601,1); turnoverTerminal=rmsTerminal;
    rmsSamePhase=rmsTerminal; turnoverSamePhase=rmsTerminal;
    for q=1:1601
        x=X(:,q); xb=volume_binary(x);
        rmsTerminal(q)=sqrt(mean((x-ref).^2)); turnoverTerminal(q)=mean(xb~=xbRef);
        if mod(q-1,2)==0
            pr=ref; pbr=xbRef;
        else
            pr=refOdd; pbr=xbRefOdd;
        end
        rmsSamePhase(q)=sqrt(mean((x-pr).^2)); turnoverSamePhase(q)=mean(xb~=pbr);
    end
    refE1=E.common_raw_E1_omega1(end);
    ratio=E.common_raw_E1_omega1/refE1; loss=(refE1-E.common_raw_E1_omega1)/refE1;
    trailing=min(50,(0:1600)'); windowTime=NaN(1601,1); eigShare=NaN(1601,1);
    for q=2:1601
        lo=max(1,q-1-49); windowTime(q)=mean(loop(lo:q-1));
        eigShare(q)=cumulativeEig(q)/cumulative(q);
    end
    curveParts{im}=table(repmat(meshId,1601,1),(0:1600)',ratio,loss, ...
        native(:,1),native(:,2),native(:,3),gap,N,rmsTerminal,turnoverTerminal, ...
        rmsSamePhase,turnoverSamePhase,cumulative,cumulativeEig,eigShare,windowTime, ...
        'VariableNames',{'mesh','iteration','common_raw_E1_ratio_to_k1600', ...
        'common_raw_E1_loss_to_k1600','native_omega1','native_omega2','native_omega3', ...
        'native_gap12','native_N','density_rms_to_k1600','binary_turnover_to_k1600', ...
        'density_rms_to_same_phase_late_state','binary_turnover_to_same_phase_late_state', ...
        'cumulative_loop_time_s','cumulative_eigensolve_time_s','cumulative_eigensolve_share', ...
        'window50_loop_time_per_iteration_s'});

    bands=[0.025 0.01 0.005];
    for b=bands
        [first,persist]=crossings(loss<=b);
        minRows(end+1,:)={meshId,"COMMON_RAW_E1_LOSS",b,first,persist}; %#ok<AGROW>
    end
    [first,persist]=crossings(gap<=0.01);
    minRows(end+1,:)={meshId,"NATIVE_GAP12",0.01,first,persist}; %#ok<AGROW>
    [first,persist]=crossings(N==2 & gap<=0.01);
    minRows(end+1,:)={meshId,"BIMODAL_N2_AND_GAP12",0.01,first,persist}; %#ok<AGROW>

    last100=1502:1601; lag2=sqrt(mean((X(:,3:end)-X(:,1:end-2)).^2,1));
    phaseRows(end+1,:)={meshId,mean(lag2(end-199:end)),max(lag2(end-199:end)), ...
        mean(native(last100,1)),min(native(last100,1)),max(native(last100,1)), ...
        mean(E.common_raw_E1_omega1(last100)),min(E.common_raw_E1_omega1(last100)), ...
        max(E.common_raw_E1_omega1(last100)),E.common_raw_E1_omega1(end)}; %#ok<AGROW>
end

Q=vertcat(curveParts{:}); writetable(Q,fullfile(here,'quality_budget_curves.csv'));
Min=cell2table(minRows,'VariableNames',{'mesh','metric','threshold_fraction','first_crossing','persistent_crossing'});
writetable(Min,fullfile(here,'minimum_quality_budget.csv'));
Phase=cell2table(phaseRows,'VariableNames',{'mesh','late200_mean_lag2_density_rms', ...
    'late200_max_lag2_density_rms','late100_native_omega1_mean','late100_native_omega1_min', ...
    'late100_native_omega1_max','late100_common_E1_mean','late100_common_E1_min', ...
    'late100_common_E1_max','terminal_common_E1'});
writetable(Phase,fullfile(here,'late_phase_diagnostics.csv'));

% Central k=200 table and preregistered adequacy classification.
K=C(C.iteration==200,:); A=table();
A.mesh=K.mesh; A.omega1_200=K.native_omega1; A.omega2_200=K.native_omega2;
A.gap12_200=K.native_gap12; A.N_200=K.native_N;
A.raw_E1_omega1_200=K.common_raw_E1_omega1;
for i=1:height(K)
    ref=C(C.mesh==K.mesh(i) & C.iteration==1600,:);
    A.raw_E1_omega1_1600(i,1)=ref.common_raw_E1_omega1;
end
A.raw_E1_quality_loss=K.common_raw_E1_omega1_loss_to_k1600;
A.raw_E2_quality_loss=K.common_raw_E2_omega1_loss_to_k1600;
A.raw_E3_quality_loss=K.common_raw_E3_omega1_loss_to_k1600;
A.binary_E1_quality_loss=K.common_binary_E1_omega1_loss_to_k1600;
A.binary_E2_quality_loss=K.common_binary_E2_omega1_loss_to_k1600;
A.binary_E3_quality_loss=K.common_binary_E3_omega1_loss_to_k1600;
A.density_rms_to_k1600=K.density_rms_to_k1600;
A.binary_topology_turnover=K.binary_turnover_to_k1600;
A.raw_connected=K.raw_05_left_right_connected; A.binary_connected=K.binary_left_right_connected;
A.raw_components=K.raw_05_n_components; A.binary_components=K.binary_n_components;
A.cumulative_time_to_200_s=K.cumulative_loop_time_s;
A.modal_classification=K.modal_k200_classification;
A.healthy=K.lp_failures_through_checkpoint==0 & K.nonfinite_iterations_through_checkpoint==0 & ...
    abs(K.volume_residual)<=pre.health_rule.maximum_absolute_volume_residual;
A.within_0p5_raw_E1=A.raw_E1_quality_loss<=0.005;
A.within_1p0_raw_E1=A.raw_E1_quality_loss<=0.01;
A.within_2p5_raw_E1=A.raw_E1_quality_loss<=0.025;
A.adequacy_class=strings(height(A),1); A.binary_conclusion=strings(height(A),1);
for i=1:height(A)
    connected=A.raw_connected(i)&&A.binary_connected(i); modal=A.modal_classification(i)=="BIMODAL_ESTABLISHED";
    if ~A.healthy(i)||~connected||A.raw_E1_quality_loss(i)>0.025
        A.adequacy_class(i)="INADEQUATE";
    elseif modal && A.raw_E1_quality_loss(i)<=0.005
        A.adequacy_class(i)="STRICTLY_ADEQUATE";
    elseif modal && A.raw_E1_quality_loss(i)<=0.01
        A.adequacy_class(i)="PRACTICALLY_ADEQUATE";
    else
        A.adequacy_class(i)="COARSELY_ADEQUATE";
    end
    secondary=[A.binary_E1_quality_loss(i),A.binary_E2_quality_loss(i),A.binary_E3_quality_loss(i)];
    if all(secondary<=0.025)
        A.binary_conclusion(i)="ALL_BINARY_EVALUATORS_WITHIN_2P5PCT";
    else
        A.binary_conclusion(i)="BINARY_EVALUATOR_IMMATURITY_GT_2P5PCT";
    end
end
writetable(A,fullfile(here,'budget_adequacy.csv'));

nStrict=nnz(A.adequacy_class=="STRICTLY_ADEQUATE");
nPractical=nnz(A.adequacy_class=="STRICTLY_ADEQUATE"|A.adequacy_class=="PRACTICALLY_ADEQUATE");
nCoarse=nnz(A.adequacy_class~="INADEQUATE");
if nPractical==4, cross="ROBUST_PRACTICAL_BUDGET";
elseif nPractical>0, cross="RESOLUTION_SENSITIVE_BUDGET";
else, cross="INSUFFICIENT_BUDGET";
end
X=table("k=200",0.01,nStrict,nPractical,nCoarse,string(cross), ...
    'VariableNames',{'budget','declared_practical_loss_band','n_strictly_adequate', ...
    'n_practically_adequate_including_strict','n_coarsely_adequate_or_better','classification'});
writetable(X,fullfile(here,'cross_resolution_classification.csv'));

make_figures(Q,meshes,here);
fprintf('Built outputs: cross-resolution classification %s.\n',cross);
end

function make_figures(Q,meshes,here)
figDir=fullfile(here,'figures'); colors=lines(4);
for im=1:size(meshes,1)
    meshId=string(sprintf('%dx%d',meshes(im,1),meshes(im,2))); T=Q(Q.mesh==meshId,:);
    f=figure('Visible','off','Color','w','Position',[50 50 1450 950]);
    tl=tiledlayout(4,2,'Padding','compact','TileSpacing','compact'); title(tl,sprintf('%s fixed-budget trajectory (k=200 is not convergence)',meshId));
    nexttile; plot(T.iteration,T.common_raw_E1_ratio_to_k1600,'LineWidth',1.2); yline(1,'k:'); budget_line(true); ylabel('\omega_1/\omega_1(1600)'); grid on
    nexttile; plot(T.iteration,100*T.common_raw_E1_loss_to_k1600,'LineWidth',1.2); h=yline([0.5 1 2.5],'--'); set(h,'HandleVisibility','off'); budget_line(false); ylabel('E1 loss (%)'); ylim([-2 12]); grid on
    nexttile; plot(T.iteration,T.native_omega1,'LineWidth',1); hold on; plot(T.iteration,T.native_omega2,'LineWidth',1); budget_line(false); ylabel('native \omega'); legend('\omega_1','\omega_2','Location','best'); grid on
    nexttile; plot(T.iteration,100*T.native_gap12,'LineWidth',1); h=yline([1 2 5],'--'); set(h,'HandleVisibility','off'); budget_line(false); ylabel('gap_{12} (%)'); ylim([0 10]); grid on
    nexttile; stairs(T.iteration,T.native_N,'LineWidth',1); budget_line(false); ylabel('multiplicity N'); ylim([0.8 max(2.2,max(T.native_N)+0.2)]); grid on
    nexttile; semilogy(T.iteration,max(T.density_rms_to_k1600,1e-8),'LineWidth',1); budget_line(false); ylabel('density RMS to k=1600'); grid on
    nexttile; plot(T.iteration,100*T.binary_turnover_to_k1600,'LineWidth',1); budget_line(false); ylabel('binary turnover (%)'); xlabel('outer-iteration budget'); grid on
    nexttile; plot(T.iteration,T.cumulative_loop_time_s,'LineWidth',1); budget_line(false); ylabel('cumulative loop time (s)'); xlabel('outer-iteration budget'); grid on
    exportgraphics(f,fullfile(figDir,sprintf('quality_vs_budget_%dx%d.png',meshes(im,1),meshes(im,2))),'Resolution',180); close(f)
end
f=figure('Visible','off','Color','w','Position',[50 50 1050 650]); hold on; p=gobjects(4,1);
for im=1:size(meshes,1)
    meshId=string(sprintf('%dx%d',meshes(im,1),meshes(im,2))); T=Q(Q.mesh==meshId,:);
    p(im)=plot(T.iteration,100*T.common_raw_E1_loss_to_k1600,'LineWidth',1.4,'Color',colors(im,:),'DisplayName',meshId);
end
h=yline([0.5 1 2.5],'--'); set(h,'HandleVisibility','off'); budget_line(true);
text(1430,0.5,'0.5%','VerticalAlignment','bottom'); text(1470,1,'1.0%','VerticalAlignment','bottom'); text(1510,2.5,'2.5%','VerticalAlignment','bottom');
xlabel('outer-iteration budget'); ylabel('common-E1 raw \omega_1 loss vs k=1600 (%)');
title('Fixed-budget quality across resolutions (k=200 is not convergence)'); legend(p,'Location','northeast'); ylim([-2 12]); grid on
exportgraphics(f,fullfile(figDir,'all_meshes_common_E1_quality_loss.png'),'Resolution',200); close(f)
end

function budget_line(withLabel)
if withLabel, h=xline(200,'r--','k=200 fixed budget','LabelOrientation','horizontal','LineWidth',1.2);
else, h=xline(200,'r--','LineWidth',1.2); end
set(h,'HandleVisibility','off');
end

function [first,persist] = crossings(c)
ix=find(c,1,'first'); if isempty(ix), first=NaN; else, first=ix-1; end
bad=find(~c); lastBad=max([-1;bad(:)-1]);
if lastBad>=numel(c)-1, persist=NaN; else, persist=lastBad+1; end
end

function n=aligned_native(res)
n=[[res.hist.omega(1,:) res.omega(1)]' [res.hist.omega(2,:) res.omega(2)]' [res.hist.omega(3,:) res.omega(3)]'];
end

function N=detect_n(w,cfg)
N=1; while cfg.n+N<=numel(w)-1 && abs(w(cfg.n+N)-w(cfg.n))/w(cfg.n)<cfg.tolMult, N=N+1; end
end

function xb=volume_binary(x)
nSolid=round(0.5*numel(x)); [~,order]=sortrows([-x(:),(1:numel(x))'],[1 2]); xb=false(size(x)); xb(order(1:nSolid))=true;
end
