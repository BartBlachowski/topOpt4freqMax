% READ_ONLY_AUDIT  NOT_NEW_OPTIMIZATION_EVIDENCE
repo='/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(repo,'analysis','iteration_efficiency_phase2a'), ...
        fullfile(repo,'analysis','three_method_parametric_study'));
A=fullfile(repo,'analysis','iteration_efficiency_evaluator_discontinuity_audit');

%% ---------- WP5: exact jump at the branch
g=@(x) (x<=0.1).*x.^6 + (x>0.1).*x;
lo=g(0.1); hiLim=0.1+eps(0.1);
fprintf('WP5 g(0.1)=%.6e   lim x->0.1+ g = %.6e\n',g(0.1),g(hiLim));
fprintf('WP5 absolute jump = %.6e  multiplicative jump = %.6e\n',g(hiLim)-g(0.1),g(hiLim)/g(0.1));
% continuity-restoring coefficients from Du2007 (4a)/(4b)
fprintf('WP5 (4a) c0=1e5 : c0*0.1^6 = %.6f   (matches 0.1: %d)\n',1e5*0.1^6,abs(1e5*0.1^6-0.1)<1e-15);
fprintf('WP5 (4b) c1=6e5,c2=-5e6 : %.6f  slope %.6f\n',6e5*0.1^6-5e6*0.1^7,6*6e5*0.1^5-7*5e6*0.1^6);

%% ---------- WP8: can repeated move-limit subtraction reach 0.099999999999999644729?
target=0.099999999999999644729;
v=0.5; for i=1:80, v=v-0.005; end
fprintf('\nWP8 0.5 minus 0.005 x80 (sequential) = %.20g\n',v);
fprintf('WP8 target observed                   = %.20g\n',target);
fprintf('WP8 sequential reproduces target: %d   (ulps below 0.1: %.1f)\n',v==target,(0.1-v)/eps(0.1));
v2=0.5-80*0.005; fprintf('WP8 single-expression 0.5-80*0.005    = %.20g (equals 0.1 exactly: %d)\n',v2,v2==0.1);
% mixed move sequence 0.005 then 0.0025
v3=0.5; for i=1:60, v3=v3-0.005; end; for i=1:40, v3=v3-0.0025; end
fprintf('WP8 0.005 x60 then 0.0025 x40         = %.20g\n',v3);
fprintf('WP8 float32 image of target = %.20g  (> 0.1: %d)\n',double(single(target)),double(single(target))>0.1);

%% ---------- WP6: double-ULP sensitivity on genuine stored paired states
P=load(fullfile(repo,'analysis','iteration_efficiency_phase2b_precision','qualification_runs', ...
    's1_transition_96x12_h320_paired_states.mat'),'x_double','pairIterations');
nelx=96; nely=12; rows={};
sel=[]; for i=1:numel(P.pairIterations)
    x=double(P.x_double(:,i)); if nnz(abs(x-0.1)<1e-9)>0, sel(end+1)=i; end %#ok<AGROW>
end
fprintf('\nWP6 states with elements within 1e-9 of 0.1: %d of %d\n',numel(sel),numel(P.pairIterations));
sel=sel(round(linspace(1,max(1,numel(sel)),min(6,numel(sel)))));
for t=1:numel(sel)
    i=sel(t); x=double(P.x_double(:,i)); k=P.pairIterations(i);
    at=find(abs(x-0.1)<1e-9); nAt=numel(at);
    xm=x; xm(at)=0.1-eps(0.1);          % one double ULP below -> x^6 branch
    xp=x; xp(at)=0.1+eps(0.1);          % one double ULP above -> linear branch
    em=ie2a.evaluate_common(xm,nelx,nely,0.5); ep=ie2a.evaluate_common(xp,nelx,nely,0.5);
    dq=abs(ep.Q_raw-em.Q_raw)./abs(em.Q_raw);
    rows(end+1,:)={k,nAt,2*eps(0.1),dq(1),dq(2),dq(3),em.Q_raw(2),ep.Q_raw(2)}; %#ok<AGROW>
    fprintf('WP6 k=%4d nAt=%4d  dx=%.3e  relE1=%.3e relE2=%.3e relE3=%.3e\n',k,nAt,2*eps(0.1),dq(1),dq(2),dq(3));
end
if ~isempty(rows)
  writetable(cell2table(rows,'VariableNames',{'iteration','n_elements_at_branch','density_perturbation', ...
   'rel_dE1','rel_dE2','rel_dE3','E2_lower_branch','E2_upper_branch'}),fullfile(A,'DOUBLE_ULP_SENSITIVITY.csv'));
end

%% ---------- WP9: independent recomputation of E1/E2/E3 from stored paired density fields
files={'gray_full_24x4_h200_paired_states.mat','s1_transition_96x12_h320_paired_states.mat'};
dims=[24 4;96 12]; out={};
for f=1:2
  S=load(fullfile(repo,'analysis','iteration_efficiency_phase2b_precision','qualification_runs',files{f}), ...
      'x_double','x_single','pairIterations');
  for i=1:numel(S.pairIterations)
    xd=double(S.x_double(:,i)); xs=double(S.x_single(:,i));
    ed=ie2a.evaluate_common(xd,dims(f,1),dims(f,2),0.5); es=ie2a.evaluate_common(xs,dims(f,1),dims(f,2),0.5);
    r=abs(ed.Q_raw-es.Q_raw)./abs(ed.Q_raw);
    td=ie2a.topology_metrics(xd,dims(f,1),dims(f,2)); ts=ie2a.topology_metrics(xs,dims(f,1),dims(f,2));
    out(end+1,:)={files{f},S.pairIterations(i),nnz((xd<=0.1)~=(xs<=0.1)),r(1),r(2),r(3), ...
        td.hard_gate_pass,ts.hard_gate_pass,td.hard_gate_pass==ts.hard_gate_pass}; %#ok<AGROW>
  end
  fprintf('WP9 recomputed %s (%d states)\n',files{f},numel(S.pairIterations));
end
T=cell2table(out,'VariableNames',{'source_file','iteration','n_branch_crossings','rel_E1','rel_E2','rel_E3', ...
  'hard_gate_double','hard_gate_single','hard_gate_identical'});
writetable(T,fullfile(A,'PHASE2B_INDEPENDENT_REPRODUCTION.csv'));
fprintf('\nWP9 INDEPENDENT: max relE1=%.3e relE2=%.3e relE3=%.3e ; hard gate identical %d/%d ; states with branch crossings %d\n', ...
  max(T.rel_E1),max(T.rel_E2),max(T.rel_E3),nnz(T.hard_gate_identical),height(T),nnz(T.n_branch_crossings>0));
