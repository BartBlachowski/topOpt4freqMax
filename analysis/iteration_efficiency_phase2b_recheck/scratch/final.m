repo='/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(repo,'analysis','iteration_efficiency_phase2b_recheck'));
[p,guard]=ie2br.setup_paths(); %#ok<ASGLU>
out=p.phase2br; levels=[.98 .99 .995]; P=100; B0=3200; BRef=3200;
S=load(fullfile(p.runs,'probe_96x12_H3200.mat'));
Rr=load(fullfile(p.runs,'resolve_96x12.mat'));
Qs=S.Qlo; Qd=S.Qhi; H0=S.H0; n=size(Qs,1);

%% ---- WP3P prefix identity
writetable(table(Rr.sample,Rr.prefixBit,Rr.castOk,'VariableNames',{'k','prefix_bit_identical','cast_identity'}), ...
    fullfile(out,'PREFIX_IDENTITY.csv'));

%% ---- identification of the double trajectory with the upper bracket
relDH=abs(Rr.Qd-Rr.Qh)./abs(Rr.Qd);
idn=table(Rr.sample,Rr.nAt,Rr.nBelow,relDH(:,1),relDH(:,2),relDH(:,3), ...
 'VariableNames',{'k','n_atrisk_elements','n_double_le_0p1','rel_E1_double_vs_upper','rel_E2_double_vs_upper','rel_E3_double_vs_upper'});
writetable(idn,fullfile(out,'DOUBLE_TRAJECTORY_IDENTIFICATION.csv'));

%% ---- WP8 evaluator error, full trajectory (double=Qd, single=Qs)
rel=abs(Qd-Qs)./abs(Qd); ab=abs(Qd-Qs);
E=table(["E1";"E2";"E3"],median(rel).',prctile(rel,95).',max(rel).',max(ab).', ...
 'VariableNames',{'evaluator','median_relative_error','p95_relative_error','maximum_relative_error','maximum_absolute_error'});
writetable(E,fullfile(out,'EVALUATOR_ERROR_SUMMARY.csv'));

%% ---- WP8B stratification by at-risk count
bins=[0 0; 1 16; 17 64; 65 1e9]; lbl=["no_atrisk";"1_16";"17_64";"ge_65"]; rows={};
for i=1:4
  m=S.atrisk>=bins(i,1)&S.atrisk<=bins(i,2); if bins(i,1)==0, m=S.atrisk==0; end
  rows(i,:)={lbl(i),nnz(m),max(rel(m,1)),max(rel(m,2)),max(rel(m,3)),median(rel(m,2))}; %#ok<AGROW>
end
writetable(cell2table(rows,'VariableNames',{'stratum','n_states','max_rel_E1','max_rel_E2','max_rel_E3','median_rel_E2'}), ...
    fullfile(out,'EVALUATOR_ERROR_STRATIFIED.csv'));

%% ---- WP10 reference equivalence
refS=ie2a.reference_phase(Qs,H0); refD=ie2a.reference_phase(Qd,H0);
mbS=ie2a.measurement_budget(B0,refS.b_ref,P,BRef); mbD=ie2a.measurement_budget(B0,refD.b_ref,P,BRef);
writetable(table(["status";"b_ref";"Q_ref_E1";"Q_ref_E2";"Q_ref_E3"], ...
 [string(refD.status);string(refD.b_ref);string(refD.Q_ref(1));string(refD.Q_ref(2));string(refD.Q_ref(3))], ...
 [string(refS.status);string(refS.b_ref);string(refS.Q_ref(1));string(refS.Q_ref(2));string(refS.Q_ref(3))], ...
 [string(strcmp(refD.status,refS.status));string(isequaln(refD.b_ref,refS.b_ref));"n/a";"n/a";"n/a"], ...
 'VariableNames',{'quantity','double','single','identical'}),fullfile(out,'REFERENCE_EQUIVALENCE.csv'));
writetable(table(["B_meas";"certification_tail_truncated";"requested_end"], ...
 [mbD.B_meas;mbD.certification_tail_truncated;mbD.requested_end], ...
 [mbS.B_meas;mbS.certification_tail_truncated;mbS.requested_end], ...
 'VariableNames',{'quantity','double','single'}),fullfile(out,'B_MEAS_EQUIVALENCE.csv'));

%% ---- WP9/WP12 quality + persistence equivalence, each using its own Q_ref
[passD,robD]=localAcc(Qd,refD.Q_ref,H0,levels);
[passS,robS]=localAcc(Qs,refS.Q_ref,H0,levels);
perD=ie2a.scan_persistence(passD,P); perS=ie2a.scan_persistence(passS,P);
qrows={};
for j=1:3
  qrows(j,:)={levels(j),perD.k_enter(j),perS.k_enter(j),isequaln(perD.k_enter(j),perS.k_enter(j)), ...
    perD.k_cert(j),perS.k_cert(j),isequaln(perD.k_cert(j),perS.k_cert(j)), ...
    nnz(passD(:,j)~=passS(:,j)),perD.instantaneous_first(j),perS.instantaneous_first(j)}; %#ok<AGROW>
end
writetable(cell2table(qrows,'VariableNames',{'q','k_enter_double','k_enter_single','k_enter_identical', ...
 'k_cert_double','k_cert_single','k_cert_identical','n_states_acceptance_differs','inst_first_double','inst_first_single'}), ...
 fullfile(out,'PERSISTENCE_EQUIVALENCE.csv'));
mrows={};
for j=1:3
  mrows(j,:)={levels(j),min(abs(robS-levels(j))),max(abs(robD-robS)),nnz(abs(robS-levels(j))<=abs(robD-robS)), ...
     nnz(passD(:,j)~=passS(:,j))}; %#ok<AGROW>
end
writetable(cell2table(mrows,'VariableNames',{'q','min_margin_single','max_robust_perturbation','n_states_margin_within_perturbation','n_classification_flips'}), ...
 fullfile(out,'QUALITY_DECISION_EQUIVALENCE.csv'));

%% ---- WP6/WP7 binary + topology equivalence over sampled genuine pairs
writetable(table(Rr.sample,Rr.binDiff,Rr.binDiff==0,Rr.hgD,Rr.hgS,Rr.hgD==Rr.hgS, ...
 'VariableNames',{'k','binary_differing_elements','binary_identical','hard_gate_double','hard_gate_single','hard_gate_identical'}), ...
 fullfile(out,'TOPOLOGY_DECISION_EQUIVALENCE.csv'));
writetable(table((1:n).',S.atrisk,S.atrisk3,S.cutTie,rel(:,1),rel(:,2),rel(:,3),robD,robS,H0, ...
 'VariableNames',{'k','n_atrisk_0p1','n_atrisk_1e_3','cutoff_tie_single','rel_E1','rel_E2','rel_E3','robust_double','robust_single','H0'}), ...
 fullfile(out,'PAIRED_STATE_METRICS.csv'));

%% ---- a_sig regime table (Patch 5)
mp=[160 20;240 30;320 40;400 50;480 60;560 70;640 80;720 90;800 100];
mq=[48 6;80 10;96 12;128 16;24 4];
as=@(nx,ny) 0.01*nx*ny/8;
ar={};
for i=1:size(mp,1), ar(end+1,:)={sprintf('%dx%d',mp(i,1),mp(i,2)),'production',as(mp(i,1),mp(i,2)),'-'}; end %#ok<AGROW>
for i=1:size(mq,1)
  v=as(mq(i,1),mq(i,2));
  if v<4, cl='STRICTER_THAN_PRODUCTION'; elseif v<=100, cl='COMPARABLE'; else, cl='WEAKER_THAN_PRODUCTION'; end
  ar(end+1,:)={sprintf('%dx%d',mq(i,1),mq(i,2)),'qualification',v,cl}; %#ok<AGROW>
end
writetable(cell2table(ar,'VariableNames',{'mesh','role','a_sig_elements','regime_vs_production'}),fullfile(out,'A_SIG_REGIME.csv'));

fprintf('\n=== HEADLINE ===\n');
fprintf('E1 max rel %.3e | E2 max rel %.3e | E3 max rel %.3e\n',max(rel(:,1)),max(rel(:,2)),max(rel(:,3)));
fprintf('b_ref double=%d single=%d identical=%d\n',refD.b_ref,refS.b_ref,isequaln(refD.b_ref,refS.b_ref));
fprintf('B_meas double=%d single=%d identical=%d\n',mbD.B_meas,mbS.B_meas,mbD.B_meas==mbS.B_meas);
for j=1:3
 fprintf('q=%.3f k_enter %d vs %d (%d) | k_cert %d vs %d (%d) | acc diff %d states\n',levels(j), ...
  perD.k_enter(j),perS.k_enter(j),isequaln(perD.k_enter(j),perS.k_enter(j)), ...
  perD.k_cert(j),perS.k_cert(j),isequaln(perD.k_cert(j),perS.k_cert(j)),nnz(passD(:,j)~=passS(:,j)));
end
fprintf('binary identical on sampled genuine pairs: %d/%d ; hard gate identical: %d/%d\n', ...
  nnz(Rr.binDiff==0),numel(Rr.binDiff),nnz(Rr.hgD==Rr.hgS),numel(Rr.hgD));
fprintf('cutoff-tie states (single): %d of %d\n',nnz(S.cutTie),n);
save(fullfile(p.runs,'final_96x12.mat'),'refD','refS','mbD','mbS','perD','perS','passD','passS','robD','robS','rel','-v7.3');
function [pass,rob]=localAcc(Q,qref,H0,levels)
ratio=Q./qref; rob=min(ratio,[],2); pass=false(size(Q,1),numel(levels));
for j=1:numel(levels), pass(:,j)=H0&rob>=levels(j); end
end
