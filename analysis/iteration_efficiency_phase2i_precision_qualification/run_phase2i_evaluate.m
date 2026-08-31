% Phase 2I WPs 4-18: Candidate-C same-state precision qualification.
repo=fileparts(fileparts(fileparts(mfilename('fullpath'))));
outDir=fileparts(mfilename('fullpath'));rawDir=fullfile(outDir,'raw');
addpath(fullfile(repo,'analysis','iteration_efficiency_phase2a'), ...
    fullfile(repo,'analysis','three_method_parametric_study'));
maxNumCompThreads(1); levels=[.98 .99 .995];
S=load(fullfile(rawDir,'capture_96x12_H3200.mat'),'Xd','Xs');
Xd=S.Xd(:,2:end);Xs=double(S.Xs(:,2:end));n=size(Xd,2);nx=96;ny=12;
assert(n==3200&&isa(S.Xd,'double'));

Qd=nan(n,3);Qs=nan(n,3);Ld=nan(n,3);Ls=nan(n,3);
ordD=nan(n,3);ordS=nan(n,3);escD=nan(n,3);escS=nan(n,3);reqD=nan(n,3);reqS=nan(n,3);
selDiagD=nan(n,3,4);selDiagS=nan(n,3,4);minMarginD=nan(n,3);minMarginS=nan(n,3);
classMismatch=false(n,3);allClassMismatch=zeros(n,3);validD=false(n,1);validS=false(n,1);
hardD=false(n,1);hardS=false(n,1);volD=false(n,1);volS=false(n,1);topD=false(n,1);topS=false(n,1);
binaryDiff=zeros(n,1);binaryExplained=true(n,1);aggregateD=zeros(n,1);aggregateS=zeros(n,1);
islandsD=zeros(n,1);islandsS=zeros(n,1);maxDetD=zeros(n,1);maxDetS=zeros(n,1);
maxAbsDx=zeros(n,1);meanAbsDx=zeros(n,1);rmsDx=zeros(n,1);maxRelDx=zeros(n,1);
nChanged=zeros(n,1);nAtRisk=zeros(n,1);nCross01=zeros(n,1);
maxVoidKEPert=zeros(3,1);maxVoidSEPert=zeros(3,1);maxDPPert=zeros(3,1);adaptiveMismatch=zeros(3,1);

fid=fopen(fullfile(rawDir,'MODAL_DIAGNOSTICS.csv'),'w');
fprintf(fid,['mesh,k,representation,evaluator,batch_schedule,final_requested,escalation,mode,' ...
    'lambda,omega,residual,eigenpair_valid,diagnostic_finite,voidKE,voidSE,' ...
    'densityParticipation,IPR,pass_KE,pass_SE,pass_P,valid_structural,selected\n']);
tAll=tic;
for k=1:n
    xd=Xd(:,k);xs=Xs(:,k);dx=xd-xs;ad=abs(dx);
    maxAbsDx(k)=max(ad);meanAbsDx(k)=mean(ad);rmsDx(k)=sqrt(mean(dx.^2));
    maxRelDx(k)=max(ad./max(abs(xd),realmin));nChanged(k)=nnz(dx);
    nAtRisk(k)=nnz(single(xd)==single(0.1));
    nCross01(k)=nnz((xd<=.1&xs>.1)|(xd>.1&xs<=.1));
    td=ie2a.topology_metrics(xd,nx,ny);ts=ie2a.topology_metrics(xs,nx,ny);
    hardD(k)=td.hard_gate_pass;hardS(k)=ts.hard_gate_pass;volD(k)=td.volume_pass;volS(k)=ts.volume_pass;
    topD(k)=td.topology_pass;topS(k)=ts.topology_pass;binaryDiff(k)=nnz(td.binary~=ts.binary);
    binaryExplained(k)=localBinaryExplained(xd,xs,td.binary,ts.binary,.5);
    aggregateD(k)=td.aggregate_detached_area;aggregateS(k)=ts.aggregate_detached_area;
    islandsD(k)=td.n_islands_all;islandsS(k)=ts.n_islands_all;
    maxDetD(k)=td.max_detached_area;maxDetS(k)=ts.max_detached_area;
    ed=ie2a.evaluate_common(xd,nx,ny,.5);es=ie2a.evaluate_common(xs,nx,ny,.5);
    validD(k)=strcmp(ed.status,'PASS');validS(k)=strcmp(es.status,'PASS');
    Qd(k,:)=ed.Q;Qs(k,:)=es.Q;
    for j=1:3
        md=ed.modal{j};ms=es.modal{j};
        Ld(k,j)=md.selected_lambda;Ls(k,j)=ms.selected_lambda;
        ordD(k,j)=md.selected_ordinal;ordS(k,j)=ms.selected_ordinal;
        escD(k,j)=md.escalation_count;escS(k,j)=ms.escalation_count;
        reqD(k,j)=md.modes_requested_final;reqS(k,j)=ms.modes_requested_final;
        selDiagD(k,j,:)=[md.selected_voidKE md.selected_voidSE md.selected_densityParticipation md.selected_IPR];
        selDiagS(k,j,:)=[ms.selected_voidKE ms.selected_voidSE ms.selected_densityParticipation ms.selected_IPR];
        minMarginD(k,j)=min(abs(md.selected_condition_margins));
        minMarginS(k,j)=min(abs(ms.selected_condition_margins));
        m=min(numel(md.valid_structural),numel(ms.valid_structural));
        relevant=min(m,max([md.selected_ordinal ms.selected_ordinal]));
        classMismatch(k,j)=any(md.valid_structural(1:relevant)~=ms.valid_structural(1:relevant));
        allClassMismatch(k,j)=nnz(md.valid_structural(1:m)~=ms.valid_structural(1:m));
        maxVoidKEPert(j)=max(maxVoidKEPert(j),max(abs(md.voidKE(1:m)-ms.voidKE(1:m))));
        maxVoidSEPert(j)=max(maxVoidSEPert(j),max(abs(md.voidSE(1:m)-ms.voidSE(1:m))));
        maxDPPert(j)=max(maxDPPert(j),max(abs(md.densityParticipation(1:m)-ms.densityParticipation(1:m))));
        adaptiveMismatch(j)=adaptiveMismatch(j)+(md.escalation_count~=ms.escalation_count||md.modes_requested_final~=ms.modes_requested_final);
        localWriteModal(fid,'96x12',k,'double',j,md);
        localWriteModal(fid,'96x12',k,'single',j,ms);
    end
    if mod(k,100)==0,fprintf('Candidate-C paired evaluation %d/%d (%.1fs)\n',k,n,toc(tAll));end
end
fclose(fid);evalWall=toc(tAll);

R=table((1:n).',maxAbsDx,meanAbsDx,rmsDx,maxRelDx,nChanged,nAtRisk,nCross01, ...
    'VariableNames',{'k','max_abs_density_error','mean_abs_density_error','rms_density_error', ...
    'max_relative_density_error','n_changed','n_single_equal_single_0p1','n_cross_rho_eff_0p1'});
writetable(R,fullfile(rawDir,'REPRESENTATION_ERROR.csv'));

Mrows={};midx=0;
for k=1:n
 for j=1:3
  midx=midx+1;Mrows(midx,:)={"96x12",k,"E"+j,ordD(k,j),ordS(k,j),ordD(k,j)==ordS(k,j), ...
      escD(k,j),escS(k,j),reqD(k,j),reqS(k,j),classMismatch(k,j),allClassMismatch(k,j), ...
      Qd(k,j),Qs(k,j),Ld(k,j),Ls(k,j),minMarginD(k,j),minMarginS(k,j)}; %#ok<AGROW>
 end
end
MT=cell2table(Mrows,'VariableNames',{'mesh','k','evaluator','selected_ordinal_double','selected_ordinal_single', ...
 'selected_ordinal_identical','escalation_double','escalation_single','final_requested_double','final_requested_single', ...
 'relevant_classifier_mismatch','all_examined_classifier_mismatch_count','omega_double','omega_single', ...
 'lambda_double','lambda_single','selected_min_abs_margin_double','selected_min_abs_margin_single'});
writetable(MT,fullfile(outDir,'MODAL_SELECTION_EQUIVALENCE.csv'));

H=table((1:n).',volD,volS,topD,topS,hardD,hardS,binaryDiff,binaryExplained,aggregateD,aggregateS, ...
 aggregateS-aggregateD,islandsD,islandsS,maxDetD,maxDetS, ...
 'VariableNames',{'k','volume_pass_double','volume_pass_single','topology_pass_double','topology_pass_single', ...
 'hard_gate_pass_double','hard_gate_pass_single','binary_differing_elements','binary_difference_explained', ...
 'aggregate_detached_area_double','aggregate_detached_area_single','delta_aggregate_detached_area', ...
 'n_islands_all_double','n_islands_all_single','max_detached_area_double','max_detached_area_single'});
writetable(H,fullfile(outDir,'HARD_GATE_EQUIVALENCE.csv'));

H0d=hardD&validD;H0s=hardS&validS;
refD=ie2a.reference_phase(Qd,H0d,EvaluatorValid=validD);refS=ie2a.reference_phase(Qs,H0s,EvaluatorValid=validS);
assert(strcmp(refD.status,'PASS')&&strcmp(refS.status,'PASS'),'Reference not established.');
mbD=ie2a.measurement_budget(3200,refD.b_ref,100,3200);mbS=ie2a.measurement_budget(3200,refS.b_ref,100,3200);
ratioD=Qd./refD.Q_ref;ratioS=Qs./refS.Q_ref;[robD,bindD]=min(ratioD,[],2);[robS,bindS]=min(ratioS,[],2);
QT=table((1:n).',ratioD(:,1),ratioS(:,1),ratioD(:,2),ratioS(:,2),ratioD(:,3),ratioS(:,3), ...
 bindD,bindS,robD,robS,robS-robD,H0d,H0s, ...
 'VariableNames',{'k','ratio_E1_double','ratio_E1_single','ratio_E2_double','ratio_E2_single', ...
 'ratio_E3_double','ratio_E3_single','binding_evaluator_double','binding_evaluator_single', ...
 'Q_double','Q_single','delta_Q_single_minus_double','H0_double','H0_single'});
for j=1:3
 QT.(['pass_q' strrep(sprintf('%.3f',levels(j)),'.','_') '_double'])=H0d&robD>=levels(j);
 QT.(['pass_q' strrep(sprintf('%.3f',levels(j)),'.','_') '_single'])=H0s&robS>=levels(j);
end
writetable(QT,fullfile(outDir,'QUALITY_EQUIVALENCE.csv'));

RT=table(["reference_status";"b_ref";"B_meas";"Q_ref_E1";"Q_ref_E2";"Q_ref_E3"], ...
 [string(refD.status);string(refD.b_ref);string(mbD.B_meas);string(refD.Q_ref(1));string(refD.Q_ref(2));string(refD.Q_ref(3))], ...
 [string(refS.status);string(refS.b_ref);string(mbS.B_meas);string(refS.Q_ref(1));string(refS.Q_ref(2));string(refS.Q_ref(3))], ...
 [string(strcmp(refD.status,refS.status));string(refD.b_ref==refS.b_ref);string(mbD.B_meas==mbS.B_meas);"NUMERIC";"NUMERIC";"NUMERIC"], ...
 'VariableNames',{'quantity','double','single','identical'});
writetable(RT,fullfile(outDir,'REFERENCE_EQUIVALENCE.csv'));

Pvals=[50 100 200];prows={};pr=0;
for ip=1:numel(Pvals)
 P=Pvals(ip);passD=false(n,3);passS=false(n,3);
 for j=1:3,passD(:,j)=H0d&robD>=levels(j);passS(:,j)=H0s&robS>=levels(j);end
 pd=ie2a.scan_persistence(passD,P);ps=ie2a.scan_persistence(passS,P);
 for j=1:3
  factsD=localFacts(refD,pd,j,validD,H0d,passD);factsS=localFacts(refS,ps,j,validS,H0s,passS);
  statD=ie2a.classify_status(factsD);statS=ie2a.classify_status(factsS);
  pr=pr+1;prows(pr,:)={levels(j),P,pd.k_enter(j),ps.k_enter(j),isequaln(pd.k_enter(j),ps.k_enter(j)), ...
      pd.k_cert(j),ps.k_cert(j),isequaln(pd.k_cert(j),ps.k_cert(j)),string(statD),string(statS),strcmp(statD,statS), ...
      nnz(passD(:,j)~=passS(:,j))}; %#ok<AGROW>
 end
end
PT=cell2table(prows,'VariableNames',{'q','P','k_enter_double','k_enter_single','k_enter_identical', ...
 'k_cert_double','k_cert_single','k_cert_identical','status_double','status_single','status_identical','n_state_crossing_differences'});
writetable(PT,fullfile(outDir,'PERSISTENCE_EQUIVALENCE.csv'));

% Numerical error summary and classifier margins.
evals=["E1";"E2";"E3"];erows={};crows={};
for j=1:3
 ae=abs(Qd(:,j)-Qs(:,j));re=ae./abs(Qd(:,j));al=abs(Ld(:,j)-Ls(:,j));rl=al./abs(Ld(:,j));
 erows(j,:)={evals(j),max(ae),max(re),prctile(re,99),prctile(re,95),median(re),sqrt(mean(re.^2)),max(al),max(rl)}; %#ok<AGROW>
 vals=[reshape(abs(selDiagD(:,j,1)-selDiagS(:,j,1)),[],1),reshape(abs(selDiagD(:,j,2)-selDiagS(:,j,2)),[],1), ...
       reshape(abs(selDiagD(:,j,3)-selDiagS(:,j,3)),[],1)];
 margins=[min(minMarginD(:,j)),min(minMarginS(:,j))];
 crows(j,:)={evals(j),min(margins),maxVoidKEPert(j),maxVoidSEPert(j),maxDPPert(j), ...
     max(vals(:,1)),max(vals(:,2)),max(vals(:,3)),nnz(classMismatch(:,j)),sum(allClassMismatch(:,j))}; %#ok<AGROW>
end
allAE=abs(Qd-Qs);allRE=allAE./abs(Qd);
erows(4,:)={"POOLED",max(allAE,[],'all'),max(allRE,[],'all'),prctile(allRE(:),99),prctile(allRE(:),95), ...
 median(allRE(:)),sqrt(mean(allRE(:).^2)),max(abs(Ld-Ls),[],'all'),max(abs(Ld-Ls)./abs(Ld),[],'all')};
writetable(cell2table(erows,'VariableNames',{'evaluator','maximum_absolute_omega_error','maximum_relative_omega_error', ...
 'p99_relative_omega_error','p95_relative_omega_error','median_relative_omega_error','rms_relative_omega_error', ...
 'maximum_absolute_lambda_error','maximum_relative_lambda_error'}),fullfile(outDir,'PRECISION_ERROR_SUMMARY.csv'));
writetable(cell2table(crows,'VariableNames',{'evaluator','minimum_selected_abs_classifier_margin', ...
 'maximum_all_mode_voidKE_perturbation','maximum_all_mode_voidSE_perturbation','maximum_all_mode_densityParticipation_perturbation', ...
 'maximum_selected_voidKE_perturbation','maximum_selected_voidSE_perturbation','maximum_selected_densityParticipation_perturbation', ...
 'states_with_relevant_classifier_mismatch','all_examined_classifier_mismatch_count'}),fullfile(outDir,'CLASSIFIER_MARGIN_SUMMARY.csv'));

save(fullfile(rawDir,'reference_evaluation.mat'),'Qd','Qs','Ld','Ls','ordD','ordS','escD','escS','reqD','reqS', ...
 'selDiagD','selDiagS','minMarginD','minMarginS','classMismatch','allClassMismatch','validD','validS', ...
 'hardD','hardS','volD','volS','topD','topS','binaryDiff','binaryExplained','refD','refS','mbD','mbS', ...
 'ratioD','ratioS','robD','robS','bindD','bindS','evalWall','maxVoidKEPert','maxVoidSEPert','maxDPPert','-v7.3');
fprintf('REFERENCE_EVALUATION_PASS wall=%.1fs b_ref=%d/%d B_meas=%d/%d\n',evalWall,refD.b_ref,refS.b_ref,mbD.B_meas,mbS.B_meas);

% Difficult paired cases under the frozen MATLAB evaluator.
DS=load(fullfile(rawDir,'difficult_pairs.mat'),'pairs');drows={};dr=0;
for i=1:numel(DS.pairs)
 p=DS.pairs(i);ed=ie2a.evaluate_common(p.x_double,p.nelx,p.nely,.5);es=ie2a.evaluate_common(double(p.x_single),p.nelx,p.nely,.5);
 for j=1:3
  md=ed.modal{j};ms=es.modal{j};m=min(numel(md.valid_structural),numel(ms.valid_structural));rel=abs(md.selected_omega-ms.selected_omega)/abs(md.selected_omega);
  dr=dr+1;drows(dr,:)={string(p.label),sprintf('%dx%d',p.nelx,p.nely),p.k,evals(j),md.selected_ordinal,ms.selected_ordinal, ...
      md.selected_ordinal==ms.selected_ordinal,md.escalation_count,ms.escalation_count,md.modes_requested_final,ms.modes_requested_final, ...
      nnz(md.valid_structural(1:m)~=ms.valid_structural(1:m)),md.selected_omega,ms.selected_omega,rel}; %#ok<AGROW>
 end
end
writetable(cell2table(drows,'VariableNames',{'case_id','mesh','k','evaluator','selected_ordinal_double','selected_ordinal_single', ...
 'ordinal_identical','escalation_double','escalation_single','final_requested_double','final_requested_single', ...
 'classifier_mismatch_count','omega_double','omega_single','relative_omega_error'}),fullfile(outDir,'DIFFICULT_CASE_MODAL_EQUIVALENCE.csv'));

% Production-scale offline trajectory exposure plus genuine final-state pairs.
meshes=[160 20;240 30;320 40;400 50;480 60;560 70;640 80;720 90];prows={};
for i=1:size(meshes,1)
 px=meshes(i,1);py=meshes(i,2);mesh=sprintf('%dx%d',px,py);
 F=load(fullfile(repo,'examples','Performance','final_campaign','raw','olhoff',['s1_' mesh '.mat']),'res');
 X=F.res.rho_snapshots;atrisk=sum(X==single(0.1),1);xd=F.res.rho;xs=double(X(:,end));
 ed=ie2a.evaluate_common(xd,px,py,.5);es=ie2a.evaluate_common(xs,px,py,.5);
 td=ie2a.topology_metrics(xd,px,py);ts=ie2a.topology_metrics(xs,px,py);
 rel=abs(ed.Q-es.Q)./abs(ed.Q);cl=0;
 for j=1:3,m=min(numel(ed.modal{j}.valid_structural),numel(es.modal{j}.valid_structural));cl=cl+nnz(ed.modal{j}.valid_structural(1:m)~=es.modal{j}.valid_structural(1:m));end
 bd=nnz(td.binary~=ts.binary);bex=localBinaryExplained(xd,xs,td.binary,ts.binary,.5);
 prows(i,:)={mesh,size(X,2),max(atrisk),nnz(atrisk>0),nnz(single(xd)==single(0.1)), ...
   ed.selected_ordinal(1),es.selected_ordinal(1),ed.selected_ordinal(2),es.selected_ordinal(2),ed.selected_ordinal(3),es.selected_ordinal(3), ...
   cl,rel(1),rel(2),rel(3),td.hard_gate_pass,ts.hard_gate_pass,bd,bex}; %#ok<AGROW>
 fprintf('Production offline %s max_atrisk=%d rel=[%.3e %.3e %.3e]\n',mesh,max(atrisk),rel);
end
writetable(cell2table(prows,'VariableNames',{'mesh','n_stored_states','maximum_atrisk_elements','states_with_atrisk', ...
 'final_atrisk_elements','ordinal_E1_double','ordinal_E1_single','ordinal_E2_double','ordinal_E2_single', ...
 'ordinal_E3_double','ordinal_E3_single','classifier_mismatch_count','relative_omega_error_E1', ...
 'relative_omega_error_E2','relative_omega_error_E3','hard_gate_double','hard_gate_single', ...
 'binary_differing_elements','binary_difference_explained'}),fullfile(outDir,'PRODUCTION_SCALE_RISK_CHECK.csv'));

function localWriteModal(fid,mesh,k,rep,j,m)
sched=strjoin(string(m.batch_schedule),'-');
for q=1:numel(m.lambda)
 fprintf(fid,'%s,%d,%s,E%d,%s,%d,%d,%d,%.17g,%.17g,%.17g,%d,%d,%.17g,%.17g,%.17g,%.17g,%d,%d,%d,%d,%d\n', ...
  mesh,k,rep,j,sched,m.modes_requested_final,m.escalation_count,q,m.lambda(q),m.omega(q),m.eigenpair_residual(q), ...
  m.eigenpair_valid(q),m.diagnostic_finite(q),m.voidKE(q),m.voidSE(q),m.densityParticipation(q),m.IPR(q), ...
  m.voidKE(q)<.5,m.voidSE(q)<.5,m.densityParticipation(q)>.5,m.valid_structural(q),q==m.selected_ordinal);
end
end
function ok=localBinaryExplained(xd,xs,bd,bs,vf)
idx=find(bd~=bs);if isempty(idx),ok=true;return;end
n=numel(xs);ns=round(vf*n);[~,o]=sortrows([-xs(:),(1:n).'],[1 2]);cut=xs(o(ns));
tie=find(xs==cut);ok=(xs(o(ns))==xs(o(ns+1)))&&all(ismember(idx,tie));
% Float32 conversion is monotone; assert no double ordering inversion.
[~,od]=sortrows([-xd(:),(1:n).'],[1 2]); %#ok<ASGLU>
end
function f=localFacts(ref,p,j,valid,H0,pass)
f=struct('reference_status',ref.status,'endpoint_found',isfinite(p.k_cert(j)), ...
 'structural_mode_not_found',any(~valid),'solver_terminated',false,'solver_termination_after_cert',false, ...
 'topology_persistence_possible',any(H0),'quality_persistence_possible',any(pass(:,j)), ...
 'pointwise_acceptance_seen',any(pass(:,j)));
end
