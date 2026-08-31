% READ_ONLY_AUDIT  NOT_NEW_OPTIMIZATION_EVIDENCE
repo='/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(repo,'analysis','iteration_efficiency_phase2a'), ...
        fullfile(repo,'analysis','three_method_parametric_study'));
A=fullfile(repo,'analysis','iteration_efficiency_evaluator_discontinuity_audit');
meshes={'160x20','240x30','320x40','400x50','480x60','560x70','640x80','720x90','800x100'};
NX=[160 240 320 400 480 560 640 720 800]; NY=[20 30 40 50 60 70 80 90 100];
s01=single(0.1); rows={};

%% ---------- WP7 exposure census (stored data is single; double is unrecoverable)
for i=1:9
  f=fullfile(repo,'examples','Performance','final_campaign','raw','olhoff',['s1_' meshes{i} '.mat']);
  d=dir(f);
  if d.bytes==0, rows(i,:)={meshes{i},'Olhoff',0,0,0,0,0,0,0,0,0,'UNAVAILABLE'}; continue; end %#ok<AGROW>
  S=load(f,'res'); X=S.res.rho_snapshots; [ne,ns]=size(X); Xd=double(X);
  eqSingle = X==s01;                    % branch-ambiguous under single storage
  dist=abs(Xd-0.1);
  rows(i,:)={meshes{i},'Olhoff',ns,ne,nnz(eqSingle),nnz(any(eqSingle,1)), ...
     nnz(dist==0),nnz(dist<=double(eps(single(0.1)))),nnz(dist<=1e-12),nnz(dist<=1e-10),nnz(dist<=1e-8), ...
     'AVAILABLE_SINGLE_ONLY'}; %#ok<AGROW>
  fprintf('WP7 %s states=%d elems=%d eqSingle01=%d states_exposed=%d\n',meshes{i},ns,ne,nnz(eqSingle),nnz(any(eqSingle,1)));
  clear S X Xd eqSingle dist
end
writetable(cell2table(rows,'VariableNames',{'mesh','method','states','elements','elements_eq_single_0p1', ...
 'states_exposed','elements_exactly_0p1_asstored','within_1_float32_ulp','within_1e_12','within_1e_10','within_1e_8','evidence_class'}), ...
 fullfile(A,'TRAJECTORY_THRESHOLD_EXPOSURE.csv'));

%% ---------- WP10/WP11 full estimand propagation on a PRODUCTION mesh (160x20)
mi=1; nelx=NX(mi); nely=NY(mi);
S=load(fullfile(repo,'examples','Performance','final_campaign','raw','olhoff',['s1_' meshes{mi} '.mat']),'res');
X=S.res.rho_snapshots; ns=size(X,2); n=ns-1;
fprintf('\nWP11 %s : %d accepted states, status=%s\n',meshes{mi},n,S.res.status);
Qup=nan(n,3); Qlo=nan(n,3); H0=false(n,1); nAt=zeros(n,1);
t=tic;
for k=1:n
  xs=X(:,k+1); x=double(xs); hit=(xs==s01); nAt(k)=nnz(hit);
  ev=study_evaluate_design(x,nelx,nely,0.5);            % as stored: at-risk on linear branch
  Qup(k,:)=[ev.omega_raw_E1(1) ev.omega_raw_E2(1) ev.omega_raw_E3(1)];
  if nAt(k)>0
    xb=x; xb(hit)=0.1;                                   % force x<=0.1 -> x^6 branch
    ev2=study_evaluate_design(xb,nelx,nely,0.5);
    Qlo(k,:)=[ev2.omega_raw_E1(1) ev2.omega_raw_E2(1) ev2.omega_raw_E3(1)];
  else
    Qlo(k,:)=Qup(k,:);
  end
  tm=ie2a.topology_metrics(x,nelx,nely); H0(k)=tm.hard_gate_pass;
  if mod(k,200)==0, fprintf('  %d/%d (%.0fs)\n',k,n,toc(t)); end
end
fprintf('WP11 eval %.0fs\n',toc(t));
% Qlo = x^6 branch (LOWER mass -> HIGHER omega). Name by branch, not by magnitude.
refA=ie2a.reference_phase(Qup,H0); refB=ie2a.reference_phase(Qlo,H0);
levels=[.98 .99 .995]; P=100; B0=3200; BRef=3200;
[pa,ra]=localAcc(Qup,refA.Q_ref,H0,levels); [pb,rb]=localAcc(Qlo,refB.Q_ref,H0,levels);
sa=ie2a.scan_persistence(pa,P); sb=ie2a.scan_persistence(pb,P);
rel=abs(Qup-Qlo)./abs(Qup);
fprintf('\nWP11 %s branch-side E1 max %.3e | E2 max %.3e | E3 max %.3e\n',meshes{mi},max(rel(:,1)),max(rel(:,2)),max(rel(:,3)));
fprintf('WP11 reference: linear-branch status=%s b_ref=%s | x6-branch status=%s b_ref=%s\n', ...
  refA.status,mat2str(refA.b_ref),refB.status,mat2str(refB.b_ref));
er={};
for j=1:3
  mbA=NaN; mbB=NaN;
  if isfinite(refA.b_ref), mbA=ie2a.measurement_budget(B0,refA.b_ref,P,BRef).B_meas; end
  if isfinite(refB.b_ref), mbB=ie2a.measurement_budget(B0,refB.b_ref,P,BRef).B_meas; end
  er(j,:)={meshes{mi},levels(j),refA.b_ref,refB.b_ref,mbA,mbB,sa.k_enter(j),sb.k_enter(j), ...
     sa.k_cert(j),sb.k_cert(j),nnz(pa(:,j)~=pb(:,j)),max(abs(ra-rb))}; %#ok<AGROW>
  fprintf('WP11 q=%.3f k_enter %s vs %s | k_cert %s vs %s | acc diff %d\n',levels(j), ...
    num2str(sa.k_enter(j)),num2str(sb.k_enter(j)),num2str(sa.k_cert(j)),num2str(sb.k_cert(j)),nnz(pa(:,j)~=pb(:,j)));
end
writetable(cell2table(er,'VariableNames',{'mesh','q','b_ref_linear_branch','b_ref_x6_branch','B_meas_linear','B_meas_x6', ...
 'k_enter_linear','k_enter_x6','k_cert_linear','k_cert_x6','n_acceptance_differs','max_robust_perturbation'}), ...
 fullfile(A,'ESTIMAND_IMPACT.csv'));
qi={};
for j=1:3
  qi(j,:)={meshes{mi},levels(j),nnz(nAt>0),nnz(pa(:,j)~=pb(:,j)),min(abs(ra-levels(j))),max(abs(ra-rb)), ...
      nnz(abs(ra-levels(j))<=max(abs(ra-rb)))}; %#ok<AGROW>
end
writetable(cell2table(qi,'VariableNames',{'mesh','q','n_states_exposed','n_actual_flips','min_decision_margin', ...
 'max_branch_perturbation','n_potential_flips'}),fullfile(A,'QUALITY_CLASSIFICATION_IMPACT.csv'));
save(fullfile(A,'scripts','wp11_160x20.mat'),'Qup','Qlo','H0','nAt','refA','refB','sa','sb','-v7.3');
function [pass,rob]=localAcc(Q,qref,H0,levels)
ratio=Q./qref; rob=min(ratio,[],2); pass=false(size(Q,1),numel(levels));
for j=1:numel(levels), pass(:,j)=H0&rob>=levels(j); end
end
