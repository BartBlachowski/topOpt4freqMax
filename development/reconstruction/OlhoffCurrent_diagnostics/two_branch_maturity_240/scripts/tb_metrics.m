function tb_metrics(A)
%TB_METRICS  Emit METRICS.json (Phases 9-14, 20) including the frozen criteria.
here = fileparts(mfilename('fullpath')); study = fileparts(here);
M = struct();
M.study = 'two_branch_maturity_240';
M.repo_head_at_task_start = '7154d8201e9defb06d0d758da866c3769c07179a';
M.impl_tree_sha256 = 'c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c';
M.matlab = '25.2.0.3042426 (R2025b) Update 1'; M.threads = 1;
M.preregistration_sha256 = '62748225253f85f6a2fbc1bad35489a2c201cd45ee64ff279601003354b73abd';
M.preregistration_frozen_utc = '2026-09-08T12:04:53Z';
M.frozen_rule = struct('W',20,'P',20, ...
  'branchA','med20 cos<0 AND med20 net_path<0.5 AND ||drho||_2 >= tol(NE)', ...
  'branchB','||drho||_2 < tol(NE) AND med20 cos>0', ...
  'tol','0.05*sqrt(NE/3200)  ==  RMS(drho) < 8.838835e-04', ...
  'event','first iteration at which either sustained 20-iteration window begins');
M.runD = A.runD;
keys = {'f160','f240','f320','f400'};
for i=1:numel(keys), M.(keys{i}) = A.branch.(keys{i}); end

% ---- preregistered pass criteria, evaluated on the WITHHELD mesh -------
s = A.branch.f240; c = struct();
c.P1_single_branch_no_retune = ~isnan(s.event) && ~strcmp(s.branch,'neither');
c.P2_remUseful_le_5pct  = ~isnan(s.event) && abs(s.remUseful)  <= 5.0;
c.P3_postRelImp_le_25pct= ~isnan(s.event) && s.postRelImp <= 25.0;
c.P4_tail_ge_400        = ~isnan(s.event) && s.tail >= 400;
if ~isnan(s.event)
    if strcmp(s.branch,'A')
        c.P5_not_saturation_artifact = (s.medcos_unsat_window < 0) || (s.boundFrac < 0.10);
    else
        c.P5_not_saturation_artifact = s.boundFrac < 0.10;
    end
else
    c.P5_not_saturation_artifact = false;
end
c.P6_no_third_regime = ~s.term_thirdRegime;
M.criteria = c;
M.criteria_all_pass = all(struct2array(c));

txt = jsonencode(M,'PrettyPrint',true);
fid = fopen(fullfile(study,'METRICS.json'),'w'); fwrite(fid,txt); fclose(fid);
fprintf('[tb_metrics] METRICS.json written (%d bytes); P1-P6 all pass = %d\n', numel(txt), M.criteria_all_pass);
disp(c);
end
