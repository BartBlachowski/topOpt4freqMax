function tm_metrics()
%TM_METRICS  Emit METRICS.json for the Phase A (terminal) topology-maturity study.
here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
load(fullfile(study,'evidence','phaseA_stats.mat'),'A');
ms = {'m160x20','m320x40'};
names = A.m160x20.names;
levels = [0.40 0.30 0.20 0.15 0.10 0.05];

M = struct();
M.study = 'topology_maturity_transition';
M.phase_reached = 'A';
M.terminal_verdict = 'TOPOLOGY_MATURITY_SIGNAL_NOT_IDENTIFIED';
M.window_W = 10;
M.solver_runs_executed = 0;
M.meshes_with_retained_trajectories = {'160x20','320x40'};
M.meshes_without_any_trajectory = {'400x50'};

for i=1:2
    r = A.(ms{i});
    s = struct();
    s.mesh = r.mesh; s.NE = r.NE; s.iterations = r.nK;
    s.production_first_descent = r.kP;
    s.armPU_first_divergence   = r.firstDivergence;
    s.mature_reference_iter    = r.kM;
    s.Mnd_initial = r.Mnd(1); s.Mnd_at_kP = r.Mnd_kP; s.Mnd_at_mature = r.Mnd_kM;
    s.remaining_topology_evolution_Mnd_pct = 100*(r.Mnd_kP-r.Mnd_kM)/(r.Mnd(1)-r.Mnd_kM);
    s.remaining_evolution_L1_endpoint_pct  = 100*r.remAtkP;
    s.remaining_evolution_L1_path_pct      = 100*r.remPathAtkP;
    s.omega1_at_kP = r.om1_kP; s.omega1_at_mature = r.om1_kM;
    s.omega1_forgone_pct = 100*(r.om1_kM/r.om1_kP-1);
    s.volume_at_kP = r.volume(r.kP);
    % regime
    c2 = r.coherence2(max(80,r.kP):min(r.nK,400));
    s.median_two_step_ratio_after_kP = median(c2(~isnan(c2)));
    s.regime = 'undetermined';
    if s.median_two_step_ratio_after_kP < 1, s.regime = 'period-2 limit cycle'; else, s.regime = 'coherent convergent descent'; end
    s.elements_at_move_bound_at_kP = NaN;
    % limit-cycle onset: first k with d2/d1 < 1 held for N consecutive iterations
    cc2 = r.coherence2(:); onset = struct();
    for N = [5 10 20]
        first = NaN;
        for k = (N+1):numel(cc2)
            seg = cc2(k-N+1:k);
            if all(~isnan(seg)) && all(seg < 1), first = k; break; end
        end
        onset.(sprintf('N%d',N)) = first;
    end
    s.limit_cycle_onset = onset;
    s.two_step_ratio_at_kP = cc2(r.kP);
    % candidate values at kP
    cv = struct();
    for n=1:numel(names), cv.(names{n}) = r.C.(names{n})(r.kP); end
    s.candidate_values_at_kP = cv;
    M.(ms{i}) = s;
end

% mesh ratio of each candidate at matched true-maturity levels
CR = struct();
for n=1:numel(names)
    v = nan(2,numel(levels));
    for i=1:2
        r = A.(ms{i});
        for j=1:numel(levels)
            k = find(r.rem<=levels(j),1);
            if ~isempty(k), v(i,j)=r.C.(names{n})(k); end
        end
    end
    ratio = v(1,:)./v(2,:);
    CR.(names{n}) = struct('levels',levels,'m160',v(1,:),'m320',v(2,:),'ratio',ratio, ...
        'ratio_min',min(ratio),'ratio_max',max(ratio), ...
        'ratio_swing', max(ratio)/max(min(ratio),eps), ...
        'value_ratio_at_kP', A.m160x20.C.(names{n})(79)/A.m320x40.C.(names{n})(130));
end
M.candidate_mesh_consistency = CR;

M.ground_truth_label = struct( ...
    'definition','remaining topology evolution = (Mnd(kP)-Mnd(mature))/(Mnd(1)-Mnd(mature))', ...
    'm160x20_pct', M.m160x20.remaining_topology_evolution_Mnd_pct, ...
    'm320x40_pct', M.m320x40.remaining_topology_evolution_Mnd_pct, ...
    'separation_factor', M.m320x40.remaining_topology_evolution_Mnd_pct / M.m160x20.remaining_topology_evolution_Mnd_pct);

txt = jsonencode(M,'PrettyPrint',true);
fid = fopen(fullfile(study,'METRICS.json'),'w'); fwrite(fid,txt); fclose(fid);
fprintf('METRICS.json written (%d bytes)\n', numel(txt));
end
