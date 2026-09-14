function [M,E,Top,D]=analyze_pairs()
%ANALYZE_PAIRS Representation, projection, topology, and E1/E2/E3 comparisons.
p=ie2b.paths();addpath(p.phase2a,fullfile(p.repo,'analysis','three_method_parametric_study'));
files={fullfile(p.runs,'gray_full_24x4_h200_paired_states.mat'), ...
    fullfile(p.runs,'s1_transition_96x12_h320_paired_states.mat')};
labels={'gray_full_24x4_h200','s1_transition_96x12_h320'};dims=[24 4;96 12];
rows={};q=0;
for f=1:numel(files)
    S=load(files{f},'x_double','x_single','pairIterations');
    for i=1:numel(S.pairIterations)
        q=q+1;rows(q,:)=localRow(labels{f},sprintf('%dx%d',dims(f,1),dims(f,2)), ...
            'tiny_prefix_pair',S.pairIterations(i),S.x_double(:,i),double(S.x_single(:,i)),dims(f,1),dims(f,2),[]); %#ok<AGROW>
        if mod(i,25)==0||i==numel(S.pairIterations),fprintf('paired evaluator %s %d/%d\n',labels{f},i,numel(S.pairIterations));end
    end
end
F=load(fullfile(p.runs,'frozen_production_final_pairs.mat'));
C=readtable(fullfile(p.repo,'examples','Performance','final_campaign','common_evaluators.csv'),'TextType','string');
for i=1:numel(F.pair_mesh)
    mesh=F.pair_mesh{i};tok=sscanf(mesh,'%dx%d');nx=tok(1);ny=tok(2);r=C(C.Method=="Olhoff"&C.Mesh==string(mesh),:);
    qd=[r.omega1_common_raw_E1 r.omega1_common_raw_E2 r.omega1_common_raw_E3];
    q=q+1;rows(q,:)=localRow(['frozen_final_' mesh],mesh,'frozen_production_final_pair',NaN, ...
        F.final_double{i},double(F.final_single{i}),nx,ny,qd); %#ok<AGROW>
    fprintf('paired evaluator frozen final %s\n',mesh);
end
names={'case_id','mesh','scope','iteration','n_elements','max_abs_dx','mean_abs_dx','rms_dx','max_relative_dx', ...
    'n_exact','n_changed','fraction_changed','cutoff_gap_double','cutoff_gap_single','cutoff_tie_double','cutoff_tie_single', ...
    'binary_differing_elements','binary_differing_fraction','binary_identical','topology_pass_double','topology_pass_single', ...
    'topology_decision_identical','support_connected_double','support_connected_single','det_max_double','det_max_single', ...
    'aggregate_detached_double','aggregate_detached_single','n_islands_all_double','n_islands_all_single', ...
    'Q_double_E1','Q_single_E1','abs_Q_error_E1','rel_Q_error_E1', ...
    'Q_double_E2','Q_single_E2','abs_Q_error_E2','rel_Q_error_E2', ...
    'Q_double_E3','Q_single_E3','abs_Q_error_E3','rel_Q_error_E3','robust_relative_error_bound'};
M=cell2table(rows,'VariableNames',names);writetable(M,fullfile(p.phase2b,'paired_state_metrics.csv'));
Top=M(:,{'case_id','mesh','scope','iteration','binary_differing_elements','binary_identical','topology_pass_double','topology_pass_single', ...
    'topology_decision_identical','support_connected_double','support_connected_single','det_max_double','det_max_single', ...
    'aggregate_detached_double','aggregate_detached_single','n_islands_all_double','n_islands_all_single'});
writetable(Top,fullfile(p.phase2b,'topology_equivalence.csv'));
E=localSummary(M);writetable(E,fullfile(p.phase2b,'evaluator_error_summary.csv'));
D=table(["exact_count_binary";"topology_PASS_FAIL";"quality_thresholds";"b_ref";"B_meas";"k_enter";"k_cert";"final_status"], ...
    [string(all(M.binary_identical));string(all(M.topology_decision_identical));"NOT_EXERCISED_NO_FROZEN_Q_REF";"NOT_EXERCISED_TRAJECTORIES_LT_REFERENCE_MINIMUM"; ...
    "NOT_EXERCISED_WITH_PAIRED_B_REF";"NOT_EXERCISED_WITH_FROZEN_ACCEPTANCE";"NOT_EXERCISED_WITH_FROZEN_ACCEPTANCE";"NOT_EXERCISED_END_TO_END"], ...
    'VariableNames',{'decision','result'});writetable(D,fullfile(p.phase2b,'decision_equivalence.csv'));
end
function row=localRow(id,mesh,scope,iteration,xd,xs,nx,ny,knownQd)
xd=double(xd(:));xs=double(xs(:));d=xd-xs;ne=numel(xd);changed=d~=0;
bd=ie2a.exact_count_binary(xd,.5);bs=ie2a.exact_count_binary(xs,.5);diffb=nnz(bd~=bs);
td=ie2a.topology_metrics(xd,nx,ny);ts=ie2a.topology_metrics(xs,nx,ny);
ed=study_evaluate_design(xd,nx,ny,.5);es=study_evaluate_design(xs,nx,ny,.5);
qd=[ed.omega_raw_E1(1) ed.omega_raw_E2(1) ed.omega_raw_E3(1)];
if ~isempty(knownQd),assert(max(abs(qd-knownQd))<1e-6,'ie2b:FrozenEvaluatorRegression','Frozen double evaluator mismatch at %s.',mesh);end
qs=[es.omega_raw_E1(1) es.omega_raw_E2(1) es.omega_raw_E3(1)];ae=abs(qd-qs);re=ae./abs(qd);
[gd,tied]=localCutoff(xd);[gs,ties]=localCutoff(xs);
row={id,mesh,scope,iteration,ne,max(abs(d)),mean(abs(d)),sqrt(mean(d.^2)),max(abs(d))/max(abs(xd)), ...
    sum(~changed),sum(changed),mean(changed),gd,gs,tied,ties,diffb,diffb/ne,diffb==0,td.topology_pass,ts.topology_pass, ...
    td.topology_pass==ts.topology_pass,td.required_connected,ts.required_connected,td.max_detached_elements,ts.max_detached_elements, ...
    td.aggregate_detached_elements,ts.aggregate_detached_elements,td.n_islands_all,ts.n_islands_all, ...
    qd(1),qs(1),ae(1),re(1),qd(2),qs(2),ae(2),re(2),qd(3),qs(3),ae(3),re(3),max(re)};
end
function [gap,tie]=localCutoff(x)
n=numel(x);ns=round(.5*n);[~,o]=sortrows([-x,(1:n).'],[1 2]);lo=x(o(ns));hi=x(o(ns+1));gap=lo-hi;tie=gap==0;
end
function E=localSummary(M)
evals=["E1";"E2";"E3"];med=zeros(3,1);p95=med;mx=med;absmax=med;
for j=1:3
    r=M.(['rel_Q_error_E' num2str(j)]);a=M.(['abs_Q_error_E' num2str(j)]);
    med(j)=median(r);p95(j)=prctile(r,95);mx(j)=max(r);absmax(j)=max(a);
end
E=table(evals,med,p95,mx,absmax,'VariableNames',{'evaluator','median_relative_error','p95_relative_error','maximum_relative_error','maximum_absolute_error'});
end
