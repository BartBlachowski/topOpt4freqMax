function sd_verify_sweeps()
%SD_VERIFY_SWEEPS  Part 3: recompute every column of the committed sweep tables
%   from the committed res.mat files (same arithmetic as repro/sweep_table.m),
%   plus per-run consistency checks.  Read-only on the snapshot; writes JSON/CSV
%   into evaluations/.  No optimization is run: one FE assembly and one eigs
%   call per run for the SIMP + eq.(4) re-evaluation.
P = sd_use_source();
maxNumCompThreads(1);
resDir = fullfile(P.snap, 'repro', 'results');
meshes = [160 20; 240 30; 320 40; 400 50; 480 60; 560 70; 640 80; 720 90; 800 100];
sweeps = {'S', 'SWEEP_R06.csv'; 'Rel', 'SWEEP_R13EL.csv'};
rows = {};
for s = 1:size(sweeps,1)
    T = readtable(fullfile(resDir, sweeps{s,2}), 'TextType', 'char');
    for i = 1:size(meshes,1)
        lab = sprintf('%s%dx%d', sweeps{s,1}, meshes(i,1), meshes(i,2));
        L = load(fullfile(resDir, lab, 'res.mat'));
        res = L.res; cfg = L.cfg; out = L.out; h = res.hist;
        flat = olh.config.toLegacy(cfg); mdl = model2D(flat); NE = mdl.nele;
        m4 = struct('model','eq4','q',1,'lowDensityExponent',6,'cutoff',0.1);
        [K,M] = assemble2D(mdl, res.rho, cfg.material.stiffness.p, m4);
        w4 = eigSolve(K, M, 3, 'eigs');
        n = h.nInner;
        it99 = find(h.omega(1,:) >= 0.99*max(h.omega(1,:)), 1);
        rec = struct();
        rec.run = lab; rec.sweep = sweeps{s,1}; rec.nelx = meshes(i,1); rec.nely = meshes(i,2); rec.NE = NE;
        rec.status = res.status; rec.outer = res.nOuter; rec.inner = sum(n); rec.innerPerOuter = mean(n);
        rec.w1_native = res.omega(1); rec.w2_native = res.omega(2); rec.w3_native = res.omega(3);
        rec.w1_eq4 = w4(1); rec.w2_eq4 = w4(2); rec.w3_eq4 = w4(3);
        rec.gap_pct = 100*(w4(2)-w4(1))/w4(1);
        rec.Mnd = 4*mean(res.rho.*(1-res.rho));
        rec.iter99 = it99; rec.wall_s = res.wallclock;
        rec.tEig_per_outer_s = sum(h.tEig)/res.nOuter;
        rec.tOuterExclInner_per_outer_s = (sum(h.tEig)+sum(h.tGrad))/res.nOuter;
        rec.t_per_inner_s = sum(h.tInner)/sum(n);
        rec.innerShare_pct = sum(h.tInner)/res.wallclock*100;
        % ---- table comparison: every numeric column, relative tolerance 1e-12
        r = T(T.nelx == meshes(i,1), :);
        cols = {'NE','outer','inner','innerPerOuter','w1_native','w2_native','w3_native', ...
            'w1_eq4','w2_eq4','w3_eq4','gap_pct','Mnd','iter99','wall_s','tEig_per_outer_s', ...
            'tOuterExclInner_per_outer_s','t_per_inner_s','innerShare_pct'};
        worst = 0; worstCol = '';
        for c = 1:numel(cols)
            a = rec.(cols{c}); b = r.(cols{c});
            d = abs(a-b)/max(abs(b), 1e-300);
            if d > worst, worst = d; worstCol = cols{c}; end
        end
        rec.table_status_match = strcmp(r.status{1}, res.status);
        rec.table_max_rel_diff = worst; rec.table_worst_col = worstCol;
        rec.table_pass = rec.table_status_match && worst <= 1e-12;
        % ---- run-internal consistency
        rec.eps = cfg.stop.tolerance;
        rec.final_dxNorm2 = h.dxNorm2(end);
        rec.log_converged = any(contains(res.log, 'converged at outer iteration'));
        rec.natural_stop_consistent = (rec.log_converged && rec.final_dxNorm2 < rec.eps && res.nOuter < cfg.runtime.maxOuter) ...
            || (~rec.log_converged && res.nOuter == cfg.runtime.maxOuter && strcmp(res.status,'CAP_HIT'));
        rec.summary_match = strcmp(out.status, res.status) && out.nOuter == res.nOuter && ...
            out.innerTotal == h.cumInner(end) && out.omega1 == res.omega(1) && out.Mnd == rec.Mnd;
        rec.aux_Mnd_final_match = abs(res.aux.Mnd(end) - rec.Mnd) == 0;
        rec.stiffness_model = cfg.material.stiffness.model;
        rec.linearBelow = cfg.material.stiffness.linearBelow;
        rec.mass_model = cfg.material.mass.model;
        rec.radiusPhysical = cfg.filter.radiusPhysical;
        rec.radiusElements = cfg.filter.radiusElements;
        if ~isempty(cfg.filter.radiusPhysical)
            rec.rmin_el = cfg.filter.radiusPhysical/(cfg.domain.b/cfg.domain.mesh.nely);
        else
            rec.rmin_el = cfg.filter.radiusElements;
        end
        rec.move_policy = cfg.move.policy; rec.move_initial = cfg.move.initial; rec.move_minimum = cfg.move.minimum;
        rec.grow = cfg.move.adaptive.grow; rec.shrink = cfg.move.adaptive.shrink;
        rec.preset = cfg.provenance.preset;
        rec.maxOuter = cfg.runtime.maxOuter; rec.singleThread = cfg.runtime.singleThread;
        rec.inner_tol = cfg.optimizer.inner.tolerance; rec.inner_min = cfg.optimizer.inner.minIterations;
        rec.inner_max = cfg.optimizer.inner.maxIterations;
        rec.innerNonConv = sum(~h.innerConv);
        rec.innerMax = max(n); rec.innerMin = min(n);
        w1 = h.omega(1,:);
        rec.spike_events = sum(w1(2:end) < 0.7*w1(1:end-1));
        rec.N_unique = unique(h.N);
        rec.rho_sha256 = sd_sha256_double(res.rho);
        rows{end+1} = rec; %#ok<AGROW>
        fprintf('%-11s %-9s outer %3d  eq4 %.1f/%.1f gap %.2f%% Mnd %.3f  table %d (%.1e %s) nat %d summ %d spikes %d rmin %.2f el %s/%s\n', ...
            lab, res.status, res.nOuter, w4(1), w4(2), rec.gap_pct, rec.Mnd, rec.table_pass, worst, worstCol, ...
            rec.natural_stop_consistent, rec.summary_match, rec.spike_events, rec.rmin_el, rec.stiffness_model, rec.mass_model);
    end
end
fid = fopen(fullfile(P.eval, 'sweep_verification.json'), 'w');
fprintf(fid, '%s\n', jsonencode([rows{:}], 'PrettyPrint', true)); fclose(fid);
end
