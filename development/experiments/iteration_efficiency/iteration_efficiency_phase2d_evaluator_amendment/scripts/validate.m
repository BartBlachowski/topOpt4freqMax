% OFFLINE_AMENDMENT_VALIDATION  NO_NEW_OPTIMIZATION
repo='/Users/piotrek/Programming/topOpt4freqMax';
D=fullfile(repo,'analysis','iteration_efficiency_phase2d_evaluator_amendment');
addpath(D,fullfile(repo,'analysis','iteration_efficiency_phase2a'), ...
        fullfile(repo,'analysis','three_method_parametric_study'));
old=@(x,nx,ny) study_evaluate_design(x,nx,ny,0.5);
new=@(x,nx,ny) ie2d.study_evaluate_design_eq4a(x,nx,ny,0.5);
Q=@(ev)[ev.omega_raw_E1(1) ev.omega_raw_E2(1) ev.omega_raw_E3(1)];

%% ================= WP4 unit tests
g4 =@(x) (x<=0.1).*x.^6      + (x>0.1).*x;
g4a=@(x) (x<=0.1).*1e5.*x.^6 + (x>0.1).*x;
pts=[0, 1e-3, 0.05, 0.1-eps(0.1), 0.1, 0.1+eps(0.1), 0.2, 0.5, 1.0];
fprintf('WP4 unit tests\n  %-24s %-14s %-14s\n','x','g_eq4','g_eq4a');
for i=1:numel(pts), fprintf('  %-24.17g %-14.6e %-14.6e\n',pts(i),g4(pts(i)),g4a(pts(i))); end
fprintf('  1e5*(0.1)^6 = %.17g ; equals 0.1 to %.3e\n',1e5*0.1^6,abs(1e5*0.1^6-0.1));
fprintf('  C0 check |g4a(0.1-) - g4a(0.1+)| = %.3e   (old Eq.4: %.3e)\n', ...
    abs(g4a(0.1-eps(0.1))-g4a(0.1+eps(0.1))),abs(g4(0.1-eps(0.1))-g4(0.1+eps(0.1))));
fprintf('  C1 one-sided derivatives at 0.1: low 6*1e5*0.1^5 = %.6f ; high = %.6f -> C1 %d\n', ...
    6*1e5*0.1^5,1,abs(6*1e5*0.1^5-1)<1e-12);
% E1 and stiffness must be untouched: field with and without low-density elements
rng_x=[0.05*ones(48,1);0.6*ones(48,1)];                      % 96 elements, 24x4
e_o=old(rng_x,24,4); e_n=new(rng_x,24,4);
fprintf('  E1 identical with low-density elements present: %d (|d|=%.3e)\n', ...
   e_o.omega_raw_E1(1)==e_n.omega_raw_E1(1),abs(e_o.omega_raw_E1(1)-e_n.omega_raw_E1(1)));
hi=0.6*ones(96,1); e_o2=old(hi,24,4); e_n2=new(hi,24,4);
fprintf('  all-x>0.1 field: E2 identical %d, E3 identical %d (no low branch taken)\n', ...
   e_o2.omega_raw_E2(1)==e_n2.omega_raw_E2(1),e_o2.omega_raw_E3(1)==e_n2.omega_raw_E3(1));
fprintf('  binary/topology fields identical: solid count %d, binary volume %d\n', ...
   e_o.binary_solid_count==e_n.binary_solid_count, e_o.binary_volume==e_n.binary_volume);

%% ================= WP5 + WP7 : paired states, OLD defect then AMENDED
files={'gray_full_24x4_h200_paired_states.mat','s1_transition_96x12_h320_paired_states.mat'};
dims=[24 4;96 12]; rows={}; rows7={};
for f=1:2
  S=load(fullfile(repo,'analysis','iteration_efficiency_phase2b_precision','qualification_runs',files{f}), ...
      'x_double','x_single','pairIterations');
  for i=1:numel(S.pairIterations)
    xd=double(S.x_double(:,i)); xs=double(S.x_single(:,i)); nx=dims(f,1); ny=dims(f,2);
    qo_d=Q(old(xd,nx,ny)); qo_s=Q(old(xs,nx,ny));
    qn_d=Q(new(xd,nx,ny)); qn_s=Q(new(xs,nx,ny));
    ro=abs(qo_d-qo_s)./abs(qo_d); rn=abs(qn_d-qn_s)./abs(qn_d);
    nx_cross=nnz((xd<=0.1)~=(xs<=0.1));
    rows7(end+1,:)={files{f},S.pairIterations(i),nx_cross,max(abs(xd-xs)), ...
       ro(1),ro(2),ro(3),rn(1),rn(2),rn(3)}; %#ok<AGROW>
  end
  fprintf('WP5/WP7 processed %s\n',files{f});
end
T7=cell2table(rows7,'VariableNames',{'source_file','iteration','n_branch_crossings','max_abs_dx', ...
  'old_rel_E1','old_rel_E2','old_rel_E3','new_rel_E1','new_rel_E2','new_rel_E3'});
writetable(T7,fullfile(D,'EQ4A_SINGLE_ROUNDING_STABILITY.csv'));
writetable(T7(T7.n_branch_crossings>0,{'source_file','iteration','n_branch_crossings','old_rel_E2','old_rel_E3','new_rel_E2','new_rel_E3'}), ...
  fullfile(D,'OLD_DEFECT_REPRODUCTION.csv'));
fprintf('\nWP5 OLD defect reproduced: max old rel E2=%.4e E3=%.4e (E1=%.4e) over %d states, %d with crossings\n', ...
  max(T7.old_rel_E2),max(T7.old_rel_E3),max(T7.old_rel_E1),height(T7),nnz(T7.n_branch_crossings>0));
fprintf('WP7 AMENDED: max new rel E2=%.4e E3=%.4e (E1=%.4e)  -> reduction factor E2 %.3g\n', ...
  max(T7.new_rel_E2),max(T7.new_rel_E3),max(T7.new_rel_E1),max(T7.old_rel_E2)/max(T7.new_rel_E2));

%% ================= WP6 double-ULP stability, OLD vs NEW
P=load(fullfile(repo,'analysis','iteration_efficiency_phase2b_precision','qualification_runs', ...
    's1_transition_96x12_h320_paired_states.mat'),'x_double','pairIterations');
sel=[]; for i=1:numel(P.pairIterations)
  if nnz(abs(double(P.x_double(:,i))-0.1)<1e-9)>0, sel(end+1)=i; end %#ok<AGROW>
end
r6={};
for t=1:numel(sel)
  i=sel(t); x=double(P.x_double(:,i)); at=abs(x-0.1)<1e-9;
  xm=x; xm(at)=0.1-eps(0.1); xp=x; xp(at)=0.1+eps(0.1);
  qom=Q(old(xm,96,12)); qop=Q(old(xp,96,12));
  qnm=Q(new(xm,96,12)); qnp=Q(new(xp,96,12));
  do=abs(qop-qom)./abs(qom); dn=abs(qnp-qnm)./abs(qnm);
  r6(end+1,:)={P.pairIterations(i),nnz(at),2*eps(0.1),do(1),do(2),do(3),dn(1),dn(2),dn(3)}; %#ok<AGROW>
end
T6=cell2table(r6,'VariableNames',{'iteration','n_elements_at_branch','density_perturbation', ...
  'old_rel_dE1','old_rel_dE2','old_rel_dE3','new_rel_dE1','new_rel_dE2','new_rel_dE3'});
writetable(T6,fullfile(D,'EQ4A_DOUBLE_ULP_STABILITY.csv'));
fprintf('\nWP6 one-double-ULP: OLD max E2=%.4e E3=%.4e | NEW max E2=%.4e E3=%.4e | reduction %.3g\n', ...
  max(T6.old_rel_dE2),max(T6.old_rel_dE3),max(T6.new_rel_dE2),max(T6.new_rel_dE3), ...
  max(T6.old_rel_dE2)/max(T6.new_rel_dE2));
save(fullfile(D,'scripts','wp4_7.mat'),'T6','T7','-v7.3');
