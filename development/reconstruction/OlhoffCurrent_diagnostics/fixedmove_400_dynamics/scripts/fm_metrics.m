function fm_metrics(A)
%FM_METRICS  Emit METRICS.json (Phases 8-13, 17) and the cross-mesh table.
here = fileparts(mfilename('fullpath')); study = fileparts(here);
M = struct();
M.study = 'fixedmove_400_dynamics';
M.repo_head_at_task_start = '7154d8201e9defb06d0d758da866c3769c07179a';
M.impl_tree_sha256 = 'c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c';
M.matlab = '25.2.0.3042426 (R2025b) Update 1';
M.threads = 1;
M.preregistration_sha256 = 'f6e84d8a65a3d8180508693f307a07ec9ae8ada51cf852a4668c087b1aeafcbc';
M.classifier = struct('persistence_P',20,'period2','median q2<1.0 AND median cos<0.0', ...
                      'coherent','median q2>1.5 AND median cos>0.25','W',10,'cap',1200);
M.runC = A.runC;
M.prefix = A.prefix;
M.abandoned = A.abandoned;
M.useful = A.useful;

% ---- cross-mesh table (Phase 13) ---------------------------------------
fixed = {'f160','f320','f400'}; prod = {'p160','p320','p400'};
NE = [3200 12800 20000]; meshes = {'160x20','320x40','400x50'};
rows = {};
for i = 1:3
    tf = A.T.(fixed{i}); tp = A.T.(prod{i});
    kd = tp.ev.firstDescent; ko = tf.C.onset.PERIOD2;
    r = struct();
    r.mesh = meshes{i}; r.NE = NE(i);
    r.production_first_descent = kd;
    r.production_cosTheta   = tp.dyn.cosT(kd);
    r.production_net_path   = tp.dyn.net_ratio(kd);
    r.production_label      = char(tp.C.label(kd));
    r.production_Mnd_at_descent = tp.per.Mnd(kd);
    r.production_boundFrac_at_descent = tp.dyn.boundFrac(kd);
    r.fixedmove_onset = ko;
    r.offset_onset_minus_descent = ko - kd;
    r.Mnd_at_onset = ternary(isnan(ko), NaN, tf.per.Mnd(max(ko,1)));
    u = A.useful.(fixed{i});
    r.remaining_useful_Mnd_pct = getf(u,'remaining_Mnd_pct');
    r.remaining_useful_L1_pct  = getf(u,'remaining_L1_pct');
    r.boundFrac_at_onset       = getf(u,'boundFrac_onset');
    r.cosTheta_raw_at_onset    = getf(u,'cos_raw_onset');
    r.cosTheta_unsat_at_onset  = getf(u,'cos_unsat_onset');
    r.net_path_raw_at_onset    = getf(u,'net_raw_onset');
    r.net_path_unsat_at_onset  = getf(u,'net_unsat_onset');
    r.medcos_raw_onsetwin      = getf(u,'medcos_raw_onsetwin');
    r.medcos_unsat_onsetwin    = getf(u,'medcos_unsat_onsetwin');
    r.post_onset_tail          = getf(u,'tail');
    r.post_onset_Mnd_change    = getf(u,'Mnd_change_after_onset');
    r.post_onset_Mnd_best_improve_rel_pct = getf(u,'Mnd_best_improve_rel_pct');
    r.post_onset_omega1_relpct = getf(u,'omega1_after_onset_relpct');
    r.fixedmove_final_Mnd      = tf.per.Mnd(end);
    r.fixedmove_final_omega1   = tf.per.omega1(end);
    r.production_final_Mnd     = tp.per.Mnd(end);
    r.production_final_omega1  = tp.per.omega1(end);
    % Phase 17: objective vs dynamical maturity
    r.betaStall_first          = tf.ev.firstBetaStall;
    r.betaStall_minus_onset    = tf.ev.firstBetaStall - ko;
    rows{end+1} = r; %#ok<AGROW>
end
M.crossMesh = rows;

% ---- Phase 17 separation summary ---------------------------------------
sep = {};
for i = 1:3
    r = rows{i};
    sep{end+1} = struct('mesh',r.mesh,'NE',r.NE, ...
        'production_descent',r.production_first_descent, ...
        'fixedmove_onset',r.fixedmove_onset, ...
        'separation_iters',r.offset_onset_minus_descent, ...
        'separation_rel_to_onset', r.offset_onset_minus_descent / max(r.fixedmove_onset,1)); %#ok<AGROW>
end
M.objective_vs_dynamical_separation = sep;

% ---- preregistered mechanism criteria ----------------------------------
u4 = A.useful.f400; ko = u4.onset;
c = struct();
c.C1_onset_within_cap   = ~isnan(ko);
c.C2_onset_after_138    = ~isnan(ko) && ko > 138;
c.C3_label_at_138_COHERENT = strcmp(char(A.T.f400.C.label(138)),'COHERENT');
c.C4_remaining_Mnd_le_5pct = ~isnan(ko) && abs(getf(u4,'remaining_Mnd_pct')) <= 5.0;
c.C5_tail_ge_400 = ~isnan(ko) && getf(u4,'tail') >= 400;
c.C5_Mnd_no_improve_gt_5pct = ~isnan(ko) && getf(u4,'Mnd_best_improve_rel_pct') <= 5.0;
c.C6_boundFrac_lt_010 = ~isnan(ko) && getf(u4,'boundFrac_onset') < 0.10;
c.C6_unsat_cos_negative = ~isnan(ko) && getf(u4,'medcos_unsat_onsetwin') < 0;
M.mechanism_criteria = c;
M.mechanism_all_pass = all(struct2array(c));

txt = jsonencode(M,'PrettyPrint',true);
fid = fopen(fullfile(study,'METRICS.json'),'w'); fwrite(fid,txt); fclose(fid);
fprintf('[fm_metrics] METRICS.json written (%d bytes); criteria all pass = %d\n', numel(txt), M.mechanism_all_pass);
end
function v = getf(s,f), if isfield(s,f), v = s.(f); else, v = NaN; end, end
function o = ternary(c,a,b), if c, o=a; else, o=b; end, end
