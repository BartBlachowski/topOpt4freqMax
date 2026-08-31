% READ_ONLY_AUDIT  NOT_NEW_OPTIMIZATION_EVIDENCE
% Independent re-propagation of the frozen estimand machinery.
% Provenance: Q arrays for the 96x12 B_ref-length reference trajectory are the
% stored Phase-2B artifact probe_96x12_H3200.mat.  Regenerating them would require
% an optimizer run, which this audit forbids; the reference/persistence machinery
% below is recomputed independently from those stored Q arrays.
repo='/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(repo,'analysis','iteration_efficiency_phase2a'));
A=fullfile(repo,'analysis','iteration_efficiency_evaluator_discontinuity_audit');
S=load(fullfile(repo,'analysis','iteration_efficiency_phase2b_recheck','qualification_runs','probe_96x12_H3200.mat'));
W=load(fullfile(A,'scripts','wp11_160x20.mat'));
levels=[.98 .99 .995]; P=100; B0=3200; BRef=3200;
er={};
% --- case 1: 96x12 reference-length trajectory (B_ref = 3200)
QA=S.Qlo;  % at-risk elements on the linear branch (as float32 storage yields)
QB=S.Qhi;  % at-risk elements on the x^6 branch (as double storage yields)
H0=S.H0;
refA=ie2a.reference_phase(QA,H0); refB=ie2a.reference_phase(QB,H0);
[pa,ra]=localAcc(QA,refA.Q_ref,H0,levels); [pb,rb]=localAcc(QB,refB.Q_ref,H0,levels);
sa=ie2a.scan_persistence(pa,P); sb=ie2a.scan_persistence(pb,P);
mbA=ie2a.measurement_budget(B0,refA.b_ref,P,BRef); mbB=ie2a.measurement_budget(B0,refB.b_ref,P,BRef);
rel=abs(QB-QA)./abs(QB);
fprintf('96x12 B_ref-length trajectory (3200 states)\n');
fprintf('  branch-side rel error: E1 %.3e  E2 %.3e  E3 %.3e\n',max(rel(:,1)),max(rel(:,2)),max(rel(:,3)));
fprintf('  b_ref  x6-branch(double)=%d  linear-branch(single)=%d  identical=%d\n',refB.b_ref,refA.b_ref,refB.b_ref==refA.b_ref);
fprintf('  B_meas x6=%d linear=%d identical=%d\n',mbB.B_meas,mbA.B_meas,mbB.B_meas==mbA.B_meas);
for j=1:3
  fprintf('  q=%.3f k_enter %d vs %d | k_cert %d vs %d | acc differs %d states\n',levels(j), ...
    sb.k_enter(j),sa.k_enter(j),sb.k_cert(j),sa.k_cert(j),nnz(pa(:,j)~=pb(:,j)));
  er(end+1,:)={'96x12_reference_length_3200',levels(j),refB.b_ref,refA.b_ref,mbB.B_meas,mbA.B_meas, ...
     sb.k_enter(j),sa.k_enter(j),sb.k_cert(j),sa.k_cert(j),nnz(pa(:,j)~=pb(:,j)),max(abs(rb-ra)), ...
     'ESTIMAND_CHANGED'}; %#ok<AGROW>
end
% --- case 2: frozen production 160x20 measurement trajectory (1600 states)
relP=abs(W.Qup-W.Qlo)./abs(W.Qup);
fprintf('\n160x20 frozen production measurement trajectory (1600 states)\n');
fprintf('  branch-side rel error: E1 %.3e  E2 %.3e  E3 %.3e\n',max(relP(:,1)),max(relP(:,2)),max(relP(:,3)));
fprintf('  reference: %s / %s  -> estimands not evaluable on this artifact\n',W.refA.status,W.refB.status);
for j=1:3
  er(end+1,:)={'160x20_production_measurement_1600',levels(j),NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN, ...
     'NOT_EVALUABLE_REFERENCE_TRAJECTORY_ABSENT'}; %#ok<AGROW>
end
writetable(cell2table(er,'VariableNames',{'case','q','b_ref_x6_branch','b_ref_linear_branch','B_meas_x6','B_meas_linear', ...
 'k_enter_x6','k_enter_linear','k_cert_x6','k_cert_linear','n_acceptance_differs','max_robust_perturbation','outcome'}), ...
 fullfile(A,'ESTIMAND_IMPACT.csv'));
% --- WP10 quality classification impact, both cases
qi={};
for j=1:3
  qi(end+1,:)={'96x12_reference_length_3200',levels(j),nnz(S.atrisk>0),nnz(pa(:,j)~=pb(:,j)), ...
     min(abs(ra-levels(j))),max(abs(rb-ra)),nnz(abs(ra-levels(j))<=max(abs(rb-ra)))}; %#ok<AGROW>
end
for j=1:3
  qi(end+1,:)={'160x20_production_measurement_1600',levels(j),nnz(W.nAt>0),NaN,NaN,max(relP(:,2)),NaN}; %#ok<AGROW>
end
writetable(cell2table(qi,'VariableNames',{'case','q','n_states_exposed','n_actual_flips','min_decision_margin', ...
 'max_branch_perturbation','n_potential_flips'}),fullfile(A,'QUALITY_CLASSIFICATION_IMPACT.csv'));
% --- WP13 neutrality: which evaluator binds the robust minimum
[~,argA]=min(QA./refA.Q_ref,[],2); [~,argB]=min(QB./refB.Q_ref,[],2);
fprintf('\nWP13 binding evaluator in robust min (linear-branch): E1 %d, E2 %d, E3 %d of %d states\n', ...
  nnz(argA==1),nnz(argA==2),nnz(argA==3),numel(argA));
fprintf('WP13 binding evaluator in robust min (x6-branch)    : E1 %d, E2 %d, E3 %d\n',nnz(argB==1),nnz(argB==2),nnz(argB==3));
fprintf('WP13 states where the binding evaluator identity changes with branch side: %d\n',nnz(argA~=argB));
writetable(table(["E1";"E2";"E3"],[nnz(argA==1);nnz(argA==2);nnz(argA==3)],[nnz(argB==1);nnz(argB==2);nnz(argB==3)], ...
  'VariableNames',{'evaluator','binding_states_linear_branch','binding_states_x6_branch'}), ...
  fullfile(A,'EVALUATOR_BINDING_SHARE.csv'));
function [pass,rob]=localAcc(Q,qref,H0,levels)
ratio=Q./qref; rob=min(ratio,[],2); pass=false(size(Q,1),numel(levels));
for j=1:numel(levels), pass(:,j)=H0&rob>=levels(j); end
end
