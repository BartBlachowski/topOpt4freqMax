function T = evaluate_runs(labels)
%EVALUATE_RUNS  Common evaluation of every recorded run's FINAL design.
%
%   T = EVALUATE_RUNS()            every repro/results/*/res.mat
%   T = EVALUATE_RUNS({'B1_...'})  a subset
%
%   For each run the final density field is re-analysed in the SAME FE model
%   under two reference mass models, independent of what the run optimised
%   with:  eq. (4) with the printed cut-off 0.1, and eq. (2) linear mass.
%   Both are Du-Olhoff models.  Also reports M_nd, grey fraction, iterations,
%   natural convergence, and the run's own native frequencies.
%   Writes repro/results/SUMMARY.md and SUMMARY.csv.

root = fileparts(fileparts(mfilename('fullpath')));
run(fullfile(root,'setpaths.m')); maxNumCompThreads(1);
resDir = fullfile(root,'repro','results');
if nargin < 1 || isempty(labels)
    d = dir(fullfile(resDir,'*','res.mat'));
    labels = cellfun(@(p) p(numel(resDir)+2:end), {d.folder}, 'UniformOutput', false);
    labels = labels(~startsWith(labels,'smoke'));
end
rows = {};
for i = 1:numel(labels)
    L = load(fullfile(resDir, labels{i}, 'res.mat'));
    res = L.res; cfg = L.cfg; out = L.out;
    flat = olh.config.toLegacy(cfg);
    mdl = model2D(flat);
    p = cfg.material.stiffness.p;
    m4  = struct('model','eq4','q',1,'lowDensityExponent',6,'cutoff',0.1);
    m2  = struct('model','eq2','q',1,'lowDensityExponent',6,'cutoff',0.1);
    [K,M] = assemble2D(mdl, res.rho, p, m4);  w4 = eigSolve(K, M, 3, 'eigs');
    [K,M] = assemble2D(mdl, res.rho, p, m2);  w2 = eigSolve(K, M, 3, 'eigs');
    h = res.hist;
    it99 = find(h.omega(1,:) >= 0.99*max(h.omega(1,:)), 1);   % first iteration within 1% of the run's peak omega_1
    rows(end+1,:) = {labels{i}, cfg.domain.mesh.nelx, cfg.domain.mesh.nely, ...
        cfg.material.mass.model, cfg.material.mass.cutoff, cfg.move.initial, ...
        local_radius(cfg), res.status, res.nOuter, h.cumInner(end), ...
        res.omega(1), res.omega(2), res.omega(3), ...
        w4(1), w4(2), w4(3), 100*(w4(2)-w4(1))/w4(1), ...
        w2(1), w2(2), w2(3), ...
        out.Mnd, out.greyFraction, out.coalescenceIter, out.wall_s, it99}; %#ok<AGROW>
    r = rows(end,:);
    fprintf('%-32s %4dx%-3d %-5s cut %.2g d0 %.3g R %.3g el | %-9s outer %3d inner %5d | native %.1f/%.1f/%.1f | eq4(0.1) %.1f/%.1f/%.1f gap %.2f%% | lin %.1f/%.1f/%.1f | Mnd %.3f | it99 %d\n', ...
        r{[1:10 11:13 14:17 18:20 21 25]});
end
T = cell2table(rows, 'VariableNames', {'run','nelx','nely','massModel','cutoff','move0', ...
    'rminEl','status','outer','inner','w1_native','w2_native','w3_native', ...
    'w1_eq4','w2_eq4','w3_eq4','gap_eq4_pct','w1_lin','w2_lin','w3_lin','Mnd','greyFrac','coalIter','wall_s','iter99'});
writetable(T, fullfile(resDir,'SUMMARY.csv'));
fid = fopen(fullfile(resDir,'SUMMARY.md'),'w');
fprintf(fid,'| run | mesh | mass | cut-off | box | r_min [el] | status | outer | inner | native w1/w2/w3 | eq.(4) w1/w2/w3 | gap eq.(4) | linear w1/w2/w3 | M_nd | coalesce@ | wall [s] | omega1 within 1%% of peak @ |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n');
for i = 1:size(rows,1)
    r = rows(i,:);
    fprintf(fid,'| %s | %dx%d | %s | %.2g | %.3g | %.2f | %s | %d | %d | %.1f / %.1f / %.1f | %.1f / %.1f / %.1f | %.2f%% | %.1f / %.1f / %.1f | %.3f | %s | %.0f | %d |\n', ...
        r{1}, r{2}, r{3}, r{4}, r{5}, r{6}, r{7}, r{8}, r{9}, r{10}, r{11}, r{12}, r{13}, r{14}, r{15}, r{16}, r{17}, r{18}, r{19}, r{20}, r{21}, num2str(r{23}), r{24}, r{25});
end
fprintf(fid,'\nPaper (Fig. 3a/4a/5): 174.7 / 174.7 / 284.9, bimodal.\n');
fclose(fid);
fprintf('wrote %s\n', fullfile(resDir,'SUMMARY.md'));
end

function r = local_radius(cfg)
if ~isempty(cfg.filter.radiusPhysical)
    r = cfg.filter.radiusPhysical/(cfg.domain.b/cfg.domain.mesh.nely);
else
    r = cfg.filter.radiusElements;
end
end
