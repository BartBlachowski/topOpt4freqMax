% Focused Phase 2I WP7 evaluator replay after difficult-pair capture.
repo=fileparts(fileparts(fileparts(mfilename('fullpath'))));
outDir=fileparts(mfilename('fullpath'));rawDir=fullfile(outDir,'raw');
addpath(fullfile(repo,'analysis','iteration_efficiency_phase2a'), ...
    fullfile(repo,'analysis','three_method_parametric_study'));
maxNumCompThreads(1);evals=["E1";"E2";"E3"];
DS=load(fullfile(rawDir,'difficult_pairs.mat'),'pairs');drows={};dr=0;
for ii=1:numel(DS.pairs)
    p=DS.pairs(ii);
    ed=ie2a.evaluate_common(p.x_double,p.nelx,p.nely,.5);
    es=ie2a.evaluate_common(double(p.x_single),p.nelx,p.nely,.5);
    for j=1:3
        md=ed.modal{j};ms=es.modal{j};m=min(numel(md.valid_structural),numel(ms.valid_structural));
        rel=abs(md.selected_omega-ms.selected_omega)/abs(md.selected_omega);dr=dr+1;
        drows(dr,:)={string(p.label),sprintf('%dx%d',p.nelx,p.nely),p.k,evals(j), ...
            md.selected_ordinal,ms.selected_ordinal,md.selected_ordinal==ms.selected_ordinal, ...
            md.escalation_count,ms.escalation_count,md.modes_requested_final,ms.modes_requested_final, ...
            nnz(md.valid_structural(1:m)~=ms.valid_structural(1:m)),md.selected_omega,ms.selected_omega,rel}; %#ok<AGROW>
    end
end
T=cell2table(drows,'VariableNames',{'case_id','mesh','k','evaluator','selected_ordinal_double', ...
    'selected_ordinal_single','ordinal_identical','escalation_double','escalation_single', ...
    'final_requested_double','final_requested_single','classifier_mismatch_count','omega_double', ...
    'omega_single','relative_omega_error'});
writetable(T,fullfile(outDir,'DIFFICULT_CASE_MODAL_EQUIVALENCE.csv'));
assert(all(T.ordinal_identical)&&all(T.classifier_mismatch_count==0));
fprintf('DIFFICULT_EVALUATION_PASS cases=%d max_ordinal=%d\n',numel(DS.pairs),max(T.selected_ordinal_double));
