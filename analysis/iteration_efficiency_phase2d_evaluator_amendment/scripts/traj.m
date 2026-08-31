% OFFLINE_AMENDMENT_VALIDATION  NO_NEW_OPTIMIZATION
repo='/Users/piotrek/Programming/topOpt4freqMax';
D=fullfile(repo,'analysis','iteration_efficiency_phase2d_evaluator_amendment');
addpath(D,fullfile(repo,'analysis','iteration_efficiency_phase2a'),fullfile(repo,'analysis','three_method_parametric_study'));
Q=@(ev)[ev.omega_raw_E1(1) ev.omega_raw_E2(1) ev.omega_raw_E3(1)];
s01=single(0.1); levels=[.98 .99 .995]; P=100; B0=3200; BRef=3200;
meshes={'160x20','240x30','320x40','400x50','480x60','560x70','640x80','720x90'};
NX=[160 240 320 400 480 560 640 720]; NY=[20 30 40 50 60 70 80 90];

%% ---- WP8 full re-evaluation of the 160x20 production trajectory
mi=1; nelx=NX(mi); nely=NY(mi);
S=load(fullfile(repo,'examples','Performance','final_campaign','raw','olhoff',['s1_' meshes{mi} '.mat']),'res');
X=S.res.rho_snapshots; n=size(X,2)-1;
W=load(fullfile(repo,'analysis','iteration_efficiency_evaluator_discontinuity_audit','scripts','wp11_160x20.mat'));
Nup=nan(n,3); Nlo=nan(n,3); H0=false(n,1); nAt=zeros(n,1); hgOld=false(n,1);
t=tic;
for k=1:n
  xs=X(:,k+1); x=double(xs); hit=(xs==s01); nAt(k)=nnz(hit);
  Nup(k,:)=Q(ie2d.study_evaluate_design_eq4a(x,nelx,nely,0.5));
  if nAt(k)>0, xb=x; xb(hit)=0.1; Nlo(k,:)=Q(ie2d.study_evaluate_design_eq4a(xb,nelx,nely,0.5));
  else, Nlo(k,:)=Nup(k,:); end
  tm=ie2a.topology_metrics(x,nelx,nely); H0(k)=tm.hard_gate_pass; hgOld(k)=W.H0(k);
  if mod(k,400)==0, fprintf('  amended eval %d/%d (%.0fs)\n',k,n,toc(t)); end
end
fprintf('WP8 amended 160x20 eval %.0fs ; states=%d\n',toc(t),n);
% WP10 hard-gate invariance
fprintf('WP10 hard gate identical old vs amended: %d/%d (topology_metrics does not consume E2/E3)\n',nnz(H0==hgOld),n);

save(fullfile(D,'scripts','traj.mat'),'Nup','Nlo','H0','nAt','-v7.3');
relOld=abs(W.Qup-W.Qlo)./abs(W.Qup);           % old branch-side sensitivity
relNew=abs(Nup-Nlo)./abs(Nup);                 % amended branch-side sensitivity
fprintf('WP8 branch-side sensitivity 160x20: OLD E2 max %.4e | NEW E2 max %.4e | reduction %.3g\n', ...
   max(relOld(:,2)),max(relNew(:,2)),max(relOld(:,2))/max(relNew(:,2)));
writetable(table((1:n).',nAt,W.Qup(:,1),W.Qup(:,2),W.Qup(:,3),Nup(:,1),Nup(:,2),Nup(:,3), ...
  relOld(:,2),relNew(:,2),relOld(:,3),relNew(:,3),H0, ...
  'VariableNames',{'k','n_atrisk','old_E1','old_E2','old_E3','new_E1','new_E2','new_E3', ...
  'old_branch_rel_E2','new_branch_rel_E2','old_branch_rel_E3','new_branch_rel_E3','hard_gate'}), ...
  fullfile(D,'AMENDED_OLHOFF_TRAJECTORY_EVALUATION.csv'));

%% ---- WP11 reference + persistence under the amended evaluator
refA=ie2a.reference_phase(Nup,H0); refB=ie2a.reference_phase(Nlo,H0);
fprintf('\nWP11 amended reference: as-stored status=%s b_ref=%s | branch-forced status=%s b_ref=%s\n', ...
  refA.status,mat2str(refA.b_ref),refB.status,mat2str(refB.b_ref));
fprintf('WP11 old-evaluator reference on same trajectory was: %s / %s\n',W.refA.status,W.refB.status);
rp={};
if isfinite(refA.b_ref)
  [pa,ra]=localAcc(Nup,refA.Q_ref,H0,levels); [pb,rb]=localAcc(Nlo,refB.Q_ref,H0,levels);
  sa=ie2a.scan_persistence(pa,P); sb=ie2a.scan_persistence(pb,P);
  mbA=ie2a.measurement_budget(B0,refA.b_ref,P,BRef); mbB=ie2a.measurement_budget(B0,refB.b_ref,P,BRef);
  for j=1:3
    rp(j,:)={meshes{mi},levels(j),refA.b_ref,refB.b_ref,mbA.B_meas,mbB.B_meas, ...
      sa.k_enter(j),sb.k_enter(j),sa.k_cert(j),sb.k_cert(j),nnz(pa(:,j)~=pb(:,j)), ...
      min(abs(ra-levels(j))),max(abs(ra-rb))};
  end
else
  for j=1:3
    rp(j,:)={meshes{mi},levels(j),NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN};
  end
  fprintf('WP11 LIMITATION: reference not establishable on this artifact under EITHER evaluator.\n');
  fprintf('   The frozen design requires a separate B_ref=3200 reference trajectory\n');
  fprintf('   (reference.trajectory_separate_from_measurement=true); stored production files are\n');
  fprintf('   the 1600-horizon measurement runs. Regenerating one needs an optimizer run, which\n');
  fprintf('   Phase 2D forbids. Recorded, not worked around.\n');
end
%% ---- WP9 binding-evaluator sensitivity (surrogate normalisation, diagnostic only)
% Q_ref from the frozen reference phase is unavailable on this artifact (see WP11).
% For a SENSITIVITY-ONLY diagnostic, normalise each evaluator by its own trajectory
% maximum. This is NOT the frozen reference and is not used for any acceptance decision;
% it only answers "can branch side change which evaluator binds the minimum".
nrm=max(Nup,[],1); nrmO=max(W.Qup,[],1);
[~,ga]=min(Nup./nrm,[],2); [~,gb]=min(Nlo./nrm,[],2);
[~,oa]=min(W.Qup./nrmO,[],2); [~,ob]=min(W.Qlo./nrmO,[],2);
fprintf('\nWP9 binding evaluator (surrogate normalisation, diagnostic):\n');
fprintf('   OLD Eq.(4)  changes with branch side: %d of %d (%.2f%%)\n',nnz(oa~=ob),n,100*nnz(oa~=ob)/n);
fprintf('   NEW Eq.(4a) changes with branch side: %d of %d (%.2f%%)\n',nnz(ga~=gb),n,100*nnz(ga~=gb)/n);
writetable(table(["E1";"E2";"E3"],[nnz(oa==1);nnz(oa==2);nnz(oa==3)],[nnz(ob==1);nnz(ob==2);nnz(ob==3)], ...
  [nnz(ga==1);nnz(ga==2);nnz(ga==3)],[nnz(gb==1);nnz(gb==2);nnz(gb==3)], ...
  'VariableNames',{'evaluator','old_binding_as_stored','old_binding_branch_forced', ...
  'new_binding_as_stored','new_binding_branch_forced'}),fullfile(D,'BINDING_EVALUATOR_ANALYSIS.csv'));
%% ---- WP13 margins
T6=load(fullfile(D,'scripts','wp4_7.mat'));
ulpOld=max(T6.T6.old_rel_dE2); ulpNew=max(T6.T6.new_rel_dE2);
f32Old=max(T6.T7.old_rel_E2);  f32New=max(T6.T7.new_rel_E2);
mg={};
for j=1:3
  band=1-levels(j);
  mg(j,:)={levels(j),band,ulpOld,ulpNew,f32Old,f32New,ulpOld/band,ulpNew/band,f32Old/band,f32New/band};
end
writetable(cell2table(mg,'VariableNames',{'q','acceptance_band','old_max_double_ulp','new_max_double_ulp', ...
 'old_max_float32','new_max_float32','old_ulp_over_band','new_ulp_over_band','old_f32_over_band','new_f32_over_band'}), ...
 fullfile(D,'NUMERICAL_MARGIN_ANALYSIS.csv'));
fprintf('\nWP13 vs q=0.995 band (0.005): old ULP %.3g of band -> new %.3g ; old f32 %.3g of band -> new %.3g\n', ...
  ulpOld/0.005,ulpNew/0.005,f32Old/0.005,f32New/0.005);
%% ---- WP8 coverage: final state of every available mesh, old vs amended
fr={};
for i=1:8
  f=fullfile(repo,'examples','Performance','final_campaign','raw','olhoff',['s1_' meshes{i} '.mat']);
  Si=load(f,'res'); xd=double(Si.res.rho); xs=double(Si.res.rho_snapshots(:,end));
  qo_d=Q(study_evaluate_design(xd,NX(i),NY(i),0.5)); qo_s=Q(study_evaluate_design(xs,NX(i),NY(i),0.5));
  qn_d=Q(ie2d.study_evaluate_design_eq4a(xd,NX(i),NY(i),0.5)); qn_s=Q(ie2d.study_evaluate_design_eq4a(xs,NX(i),NY(i),0.5));
  ro=abs(qo_d-qo_s)./abs(qo_d); rn=abs(qn_d-qn_s)./abs(qn_d);
  fr(i,:)={meshes{i},nnz(Si.res.rho_snapshots(:,end)==s01),ro(1),ro(2),ro(3),rn(1),rn(2),rn(3)}; %#ok<AGROW>
  fprintf('WP8 final %s: old relE2=%.3e new relE2=%.3e\n',meshes{i},ro(2),rn(2));
  clear Si
end
writetable(cell2table(fr,'VariableNames',{'mesh','n_atrisk_final','old_rel_E1','old_rel_E2','old_rel_E3', ...
 'new_rel_E1','new_rel_E2','new_rel_E3'}),fullfile(D,'PHASE2B_CASE_REEVALUATION.csv'));

function [pass,rob]=localAcc(Qv,qref,H0,levels)
ratio=Qv./qref; rob=min(ratio,[],2); pass=false(size(Qv,1),numel(levels));
for j=1:numel(levels), pass(:,j)=H0&rob>=levels(j); end
end
