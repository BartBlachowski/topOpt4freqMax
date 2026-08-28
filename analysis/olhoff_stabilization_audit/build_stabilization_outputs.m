function build_stabilization_outputs()
%BUILD_STABILIZATION_OUTPUTS Consolidate validation tables and diagnostic figures.
maxNumCompThreads(1);here=fileparts(mfilename('fullpath'));repo=fileparts(fileparts(here));
meshes=[160 20;240 30;320 40;400 50];profiles={'S0','S1','S2'};bands=[.025 .01 .005];
baseQ=readtable(fullfile(repo,'analysis','olhoff_fixed_budget_audit','quality_budget_curves.csv'),'TextType','string');
runRows={};qualityRows={};modalRows={};healthRows={};topTables={};curve=struct();
for im=1:4
 nelx=meshes(im,1);nely=meshes(im,2);meshId=string(sprintf('%dx%d',nelx,nely));
 baseMat=fullfile(repo,'analysis','olhoff_native_convergence','results',sprintf('development_%dx%d.mat',nelx,nely));
 B=load(baseMat,'res');baseRes=B.res;baseX=double(baseRes.telemetry.rho_snapshots);baseRef=baseX(:,end);xbRef=volume_binary(baseRef);
 for ip=1:3
  p=profiles{ip};
  if strcmp(p,'S0')
   T=baseQ(baseQ.mesh==meshId,:);res=baseRes;X=baseX;loss=T.common_raw_E1_loss_to_k1600;
   native=[T.native_omega1 T.native_omega2 T.native_omega3];gap=T.native_gap12;N=T.native_N;
   runtime=T.cumulative_loop_time_s;move=[0 repmat(res.cfg.move,1,1600)]';dRms=[NaN res.telemetry.d_rms]';bound=[NaN res.telemetry.move_bound_fraction]';
   status="CAP_HIT";triggers=[];stage=1;matFile=baseMat;e1Terminal=T.common_raw_E1_ratio_to_k1600(end)*terminal_ref(repo,meshId);
  else
   phase='holdout';if im==2,phase='development';end
   matFile=fullfile(here,phase,sprintf('%s_%dx%d.mat',lower(p),nelx,nely));R=load(matFile,'res');res=R.res;X=double(res.rho_snapshots);
   T=readtable(fullfile(here,'raw',sprintf('e1_%s_%dx%d.csv',lower(p),nelx,nely)),'TextType','string');loss=T.loss_to_baseline_k1600;
   native=[[res.hist.omega(1,:) res.omega(1)]' [res.hist.omega(2,:) res.omega(2)]' [res.hist.omega(3,:) res.omega(3)]'];
   N=[res.hist.N detect_n(res.omega,res.cfg)]';gap=(native(:,2)-native(:,1))./native(:,1);
   runtime=[0 cumsum(res.hist.tEig+res.hist.tGrad+res.hist.tInner)]';move=[0 res.hist.moveLimit]';dRms=[NaN res.hist.dRms]';bound=[NaN res.hist.moveBoundFraction]';
   status=string(res.status);triggers=res.trigger_iterations;stage=res.final_policy_stage;e1Terminal=T.common_raw_E1_omega1(end);
   topTables{end+1}=readtable(fullfile(here,'raw',sprintf('checkpoints_%s_%dx%d.csv',lower(p),nelx,nely)),'TextType','string'); %#ok<AGROW>
  end
  for b=bands
   [first,persist,nExit]=crossings(loss<=b);timeFirst=time_at(runtime,first);timePersist=time_at(runtime,persist);
   if strcmp(p,'S0'),basePersist=persist;baseTime=timePersist;deltaK=0;deltaTime=0;
   else
    [~,basePersist]=baseline_cross(baseQ,meshId,b);baseTime=time_at(baseQ.cumulative_loop_time_s(baseQ.mesh==meshId),basePersist);
    deltaK=basePersist-persist;deltaTime=baseTime-timePersist;
   end
   qualityRows(end+1,:)={string(p),meshId,b,first,persist,nExit,timeFirst,timePersist,basePersist,deltaK,baseTime,deltaTime}; %#ok<AGROW>
  end
  [firstN,persistN,exitN]=crossings(N==2);[firstG,persistG,exitG]=crossings(gap<=.01);[firstBoth,persistBoth,exitBoth]=crossings(N==2&gap<=.01);
  modalRows(end+1,:)={string(p),meshId,firstN,persistN,exitN,firstG,persistG,exitG,firstBoth,persistBoth,exitBoth, ... %#ok<AGROW>
   native(end,1),native(end,2),native(end,3),gap(end),N(end)};
  loopTotal=runtime(end);eigTotal=sum_time(res,'eig');innerTotal=sum_time(res,'inner');
  if strcmp(p,'S0'),lpBad=nnz(res.telemetry.lp_flag~=1);finiteBad=nnz(~res.telemetry.finite_ok);volBad=nnz(abs(res.hist.vol-.5)>1e-6);
  else,lpBad=nnz(res.hist.lpFlag~=1);finiteBad=nnz(~res.hist.finiteOk);volBad=nnz(abs(res.hist.volumeResidual)>1e-6);end
  healthRows(end+1,:)={string(p),meshId,status,res.nOuter,lpBad,finiteBad,volBad,loopTotal,eigTotal,innerTotal,eigTotal/max(loopTotal,eps), ... %#ok<AGROW>
   res.wallclock,string(mat2str(triggers)),stage};
  cp=terminal_checkpoint(here,repo,p,meshId,nelx,nely);
  selection=selection_status(p,meshId,status,cp);
  runRows(end+1,:)={string(p),meshId,string(ternary(im==2,'CALIBRATION','HOLDOUT')),string(matFile),status,string(mat2str(triggers)),stage, ... %#ok<AGROW>
   loss(end),e1Terminal,cp.raw_E2_loss,cp.raw_E3_loss,cp.binary_E1_loss,cp.binary_E2_loss,cp.binary_E3_loss, ...
   cp.density_rms,cp.binary_turnover,cp.raw_connected,cp.binary_connected,cp.raw_components,cp.binary_components, ...
   native(end,1),native(end,2),native(end,3),gap(end),N(end),loopTotal,string(selection)};
  if strcmp(p,'S0')||strcmp(p,'S1')
   key=sprintf('%s_%d_%d',lower(p),nelx,nely);curve.(key)=table((0:numel(loss)-1)',loss,gap,N,move,dRms,bound,runtime, ...
    'VariableNames',{'iteration','raw_E1_loss','gap12','N','move','dRms','move_bound_fraction','runtime'});
  end
 end
end

% Disclose rejected S3 calibration.
p='S3';nelx=240;nely=30;meshId="240x30";R=load(fullfile(here,'development','s3_240x30.mat'),'res');res=R.res;
T=readtable(fullfile(here,'raw','e1_s3_240x30.csv'),'TextType','string');cpT=readtable(fullfile(here,'raw','checkpoints_s3_240x30.csv'),'TextType','string');cp=cpT(end,:);
runRows(end+1,:)={"S3",meshId,"CALIBRATION",string(fullfile(here,'development','s3_240x30.mat')),string(res.status),string(mat2str(res.trigger_iterations)),res.final_policy_stage, ...
 T.loss_to_baseline_k1600(end),T.common_raw_E1_omega1(end),cp.raw_E2_loss,cp.raw_E3_loss,cp.binary_E1_loss,cp.binary_E2_loss,cp.binary_E3_loss, ...
 cp.density_rms_to_baseline_k1600,cp.binary_turnover_to_baseline_k1600,cp.raw_connected,cp.binary_connected,cp.raw_components,cp.binary_components, ...
 NaN,NaN,NaN,NaN,res.hist.N(end),sum(res.hist.tEig+res.hist.tGrad+res.hist.tInner),"REJECTED_SOLVER_FAILURE"};
healthRows(end+1,:)={"S3",meshId,string(res.status),res.nOuter,1,0,0,sum(res.hist.tEig+res.hist.tGrad+res.hist.tInner),sum(res.hist.tEig),sum(res.hist.tInner),sum(res.hist.tEig)/sum(res.hist.tEig+res.hist.tGrad+res.hist.tInner),res.wallclock,string(mat2str(res.trigger_iterations)),res.final_policy_stage};

Runs=cell2table(runRows,'VariableNames',{'profile','mesh','stage','artifact','run_status','trigger_iterations','final_policy_stage', ...
 'terminal_raw_E1_loss','terminal_raw_E1_omega1','terminal_raw_E2_loss','terminal_raw_E3_loss','terminal_binary_E1_loss','terminal_binary_E2_loss','terminal_binary_E3_loss', ...
 'density_rms_to_baseline_k1600','binary_turnover_to_baseline_k1600','raw_connected','binary_connected','raw_components','binary_components', ...
 'native_omega1','native_omega2','native_omega3','native_gap12','native_N','total_loop_time_s','selection_status'});
writetable(Runs,fullfile(here,'stabilization_runs.csv'));
P=cell2table(qualityRows,'VariableNames',{'profile','mesh','quality_band','first_crossing','persistent_crossing','later_exits','time_to_first_s','time_to_persistent_s', ...
 'baseline_persistent_crossing','delta_k_persistent','baseline_time_to_persistent_s','delta_time_to_persistent_s'});writetable(P,fullfile(here,'persistent_quality.csv'));
M=cell2table(modalRows,'VariableNames',{'profile','mesh','first_N2','persistent_N2','N2_reopenings','first_gap1','persistent_gap1','gap1_reopenings', ...
 'first_bimodal_gap1','persistent_bimodal_gap1','bimodal_reopenings','terminal_omega1','terminal_omega2','terminal_omega3','terminal_gap12','terminal_N'});writetable(M,fullfile(here,'modal_stability.csv'));
H=cell2table(healthRows,'VariableNames',{'profile','mesh','run_status','iterations','lp_failures','nonfinite_iterations','volume_failures','loop_time_s','eigensolve_time_s','inner_time_s','eigensolve_share','wallclock_s','trigger_iterations','final_policy_stage'});writetable(H,fullfile(here,'solver_health.csv'));
Topology=vertcat(topTables{:});writetable(Topology,fullfile(here,'topology_stability.csv'));
save(fullfile(here,'raw','selected_curves.mat'),'curve','-v7.3');make_figures(curve,P,meshes,here);make_topology_snapshots(repo,here,meshes);
fprintf('Built stabilization outputs: runs=%d quality=%d modal=%d health=%d topology=%d\n',height(Runs),height(P),height(M),height(H),height(Topology));
end

function [first,persist,nExit]=crossings(c),ix=find(c,1);if isempty(ix),first=NaN;else,first=ix-1;end;bad=find(~c);last=max([-1;bad(:)-1]);if last>=numel(c)-1,persist=NaN;else,persist=last+1;end;nExit=nnz(diff(c)==-1);end
function [f,p]=baseline_cross(Q,mesh,b),T=Q(Q.mesh==mesh,:);[f,p]=crossings(T.common_raw_E1_loss_to_k1600<=b);end
function t=time_at(runtime,k),if isnan(k),t=NaN;else,t=runtime(k+1);end,end
function v=terminal_ref(repo,mesh),C=readtable(fullfile(repo,'analysis','olhoff_fixed_budget_audit','checkpoint_metrics.csv'),'TextType','string');v=C.common_raw_E1_omega1(C.mesh==mesh&C.iteration==1600);end
function N=detect_n(w,cfg),N=1;while cfg.n+N<=numel(w)-1&&abs(w(cfg.n+N)-w(cfg.n))/w(cfg.n)<cfg.tolMult,N=N+1;end,end
function t=sum_time(res,kind),if strcmp(kind,'eig'),t=sum(res.hist.tEig);else,t=sum(res.hist.tInner);end,end
function cp=terminal_checkpoint(here,repo,p,mesh,nelx,nely)
if strcmp(p,'S0')
 C=readtable(fullfile(repo,'analysis','olhoff_fixed_budget_audit','checkpoint_metrics.csv'),'TextType','string');r=C(C.mesh==mesh&C.iteration==1600,:);
 cp=struct('raw_E2_loss',0,'raw_E3_loss',0,'binary_E1_loss',0,'binary_E2_loss',0,'binary_E3_loss',0,'density_rms',0,'binary_turnover',0, ...
  'raw_connected',r.raw_05_left_right_connected,'binary_connected',r.binary_left_right_connected,'raw_components',r.raw_05_n_components,'binary_components',r.binary_n_components);
else
 C=readtable(fullfile(here,'raw',sprintf('checkpoints_%s_%dx%d.csv',lower(p),nelx,nely)),'TextType','string');r=C(end,:);
 cp=struct('raw_E2_loss',r.raw_E2_loss,'raw_E3_loss',r.raw_E3_loss,'binary_E1_loss',r.binary_E1_loss,'binary_E2_loss',r.binary_E2_loss,'binary_E3_loss',r.binary_E3_loss, ...
  'density_rms',r.density_rms_to_baseline_k1600,'binary_turnover',r.binary_turnover_to_baseline_k1600,'raw_connected',r.raw_connected,'binary_connected',r.binary_connected, ...
  'raw_components',r.raw_components,'binary_components',r.binary_components);
end
end
function s=selection_status(p,mesh,status,cp)
if strcmp(p,'S0'),s='CONTROL';elseif strcmp(status,'SOLVER_FAILURE'),s='REJECTED_SOLVER_FAILURE';elseif strcmp(p,'S2')&&mesh=="400x50",s='REJECTED_BINARY_QUALITY';elseif strcmp(p,'S2'),s='REJECTED_NOT_SELECTED';elseif cp.binary_E1_loss<=.025&&cp.binary_E2_loss<=.025&&cp.binary_E3_loss<=.025,s='SELECTED_VALIDATED';else,s='REJECTED_VALIDITY';end
end
function xb=volume_binary(x),n=round(.5*numel(x));[~,o]=sortrows([-x(:),(1:numel(x))'],[1 2]);xb=false(size(x));xb(o(1:n))=true;end

function make_figures(curve,P,meshes,here)
labels={'Raw E1 loss (%)','Native gap12 (%)','Multiplicity N','Move limit','Density update RMS','Move-bound fraction (%)'};fields={'raw_E1_loss','gap12','N','move','dRms','move_bound_fraction'};
for j=1:numel(fields)
 f=figure('Visible','off','Color','w','Position',[50 50 1300 850]);tl=tiledlayout(2,2,'Padding','compact','TileSpacing','compact');title(tl,[labels{j} ': baseline vs selected S1']);
 for im=1:4,nexttile;key0=sprintf('s0_%d_%d',meshes(im,1),meshes(im,2));key1=sprintf('s1_%d_%d',meshes(im,1),meshes(im,2));A=curve.(key0);B=curve.(key1);
  y0=A.(fields{j});y1=B.(fields{j});if ismember(fields{j},{'raw_E1_loss','gap12','move_bound_fraction'}),y0=100*y0;y1=100*y1;end
  if strcmp(fields{j},'N'),stairs(A.iteration,y0,'LineWidth',1);hold on;stairs(B.iteration,y1,'LineWidth',1);else,plot(A.iteration,y0,'LineWidth',1);hold on;plot(B.iteration,y1,'LineWidth',1);end
  tr=find(B.move<0.005&B.move>0,1);if ~isempty(tr),xline(B.iteration(tr),'k:','S1 trigger','HandleVisibility','off');end
  if strcmp(fields{j},'raw_E1_loss'),ylim([-2 12]);elseif strcmp(fields{j},'gap12'),ylim([0 10]);end
  grid on;title(sprintf('%dx%d',meshes(im,1),meshes(im,2)));xlabel('iteration');ylabel(labels{j});if im==1,legend('S0 baseline','S1 stabilized','Location','best');end
 end
 exportgraphics(f,fullfile(here,'figures',sprintf('fig%02d_%s.png',j,fields{j})),'Resolution',170);close(f)
end
% Full binary turnover to baseline late topology for S0 and S1.
f=figure('Visible','off','Color','w','Position',[50 50 1300 850]);tl=tiledlayout(2,2,'Padding','compact');title(tl,'Binary topology turnover to baseline k=1600');
for im=1:4,nexttile;nelx=meshes(im,1);nely=meshes(im,2);repo=fileparts(fileparts(here));
 B=load(fullfile(repo,'analysis','olhoff_native_convergence','results',sprintf('development_%dx%d.mat',nelx,nely)),'res');ref=volume_binary(double(B.res.telemetry.rho_snapshots(:,end)));
 S=load(fullfile(here,ternary(im==2,'development','holdout'),sprintf('s1_%dx%d.mat',nelx,nely)),'res');X=double(S.res.rho_snapshots);turn=NaN(1601,1);for q=1:1601,turn(q)=mean(volume_binary(X(:,q))~=ref);end
 Q=readtable(fullfile(repo,'analysis','olhoff_fixed_budget_audit','quality_budget_curves.csv'),'TextType','string');mesh=string(sprintf('%dx%d',nelx,nely));A=Q(Q.mesh==mesh,:);
 plot(A.iteration,100*A.binary_turnover_to_k1600,'LineWidth',1);hold on;plot(0:1600,100*turn,'LineWidth',1);grid on;title(mesh);xlabel('iteration');ylabel('turnover (%)');if im==1,legend('S0','S1');end,end
exportgraphics(f,fullfile(here,'figures','fig07_binary_topology_turnover.png'),'Resolution',170);close(f)
% Time to persistent 1% and 0.5%.
f=figure('Visible','off','Color','w','Position',[50 50 1100 600]);
band=[.01 .005];
for ib=1:2
 subplot(1,2,ib);b=band(ib);Y=NaN(4,2);
 for im=1:4
  mesh=string(sprintf('%dx%d',meshes(im,1),meshes(im,2)));
  pp={'S0','S1'};
  for ip=1:2
   r=P(P.mesh==mesh&P.profile==pp{ip}&abs(P.quality_band-b)<1e-12,:);Y(im,ip)=r.time_to_persistent_s;
  end
 end
 bar(Y);set(gca,'XTickLabel',compose('%dx%d',meshes(:,1),meshes(:,2)));ylabel('measured time (s)');
 title(sprintf('Persistent raw E1 <= %.1f%%',100*b));legend('S0','S1');grid on
end
exportgraphics(f,fullfile(here,'figures','fig08_time_to_persistent_quality.png'),'Resolution',180);close(f)
end
function make_topology_snapshots(repo,here,meshes)
baseDyn=readtable(fullfile(here,'baseline_late_dynamics.csv'),'TextType','string');events=readtable(fullfile(here,'raw','baseline_excursions.csv'),'TextType','string');
f=figure('Visible','off','Color','w','Position',[30 30 1500 900]);tl=tiledlayout(4,4,'Padding','compact','TileSpacing','compact');title(tl,'Topology states around first good entry and departure');
for im=1:4,nelx=meshes(im,1);nely=meshes(im,2);mesh=string(sprintf('%dx%d',nelx,nely));B=load(fullfile(repo,'analysis','olhoff_native_convergence','results',sprintf('development_%dx%d.mat',nelx,nely)),'res');
 S=load(fullfile(here,ternary(im==2,'development','holdout'),sprintf('s1_%dx%d.mat',nelx,nely)),'res');d=baseDyn(baseDyn.mesh==mesh,:);ex=events(events.mesh==mesh&abs(events.quality_band-.01)<1e-12,:);k1=d.first_1p0;if isempty(ex),kd=k1;dep='no later exit';else,kd=ex.exit_iteration(1);dep=sprintf('departure k=%d',kd);end
 states={double(B.res.telemetry.rho_snapshots(:,k1+1)),double(B.res.telemetry.rho_snapshots(:,kd+1)),double(S.res.rho_snapshots(:,kd+1)),double(S.res.rho_snapshots(:,end))};titles={sprintf('%s S0 first 1%% k=%d',mesh,k1),sprintf('S0 %s',dep),sprintf('S1 corresponding k=%d',kd),'S1 late k=1600'};
 for j=1:4,nexttile;imagesc(1-reshape(states{j},nely,nelx));axis image off;colormap(gray);title(titles{j});end
end
exportgraphics(f,fullfile(here,'figures','fig09_topology_snapshots.png'),'Resolution',180);close(f)
end
function x=ternary(c,a,b),if c,x=a;else,x=b;end,end
