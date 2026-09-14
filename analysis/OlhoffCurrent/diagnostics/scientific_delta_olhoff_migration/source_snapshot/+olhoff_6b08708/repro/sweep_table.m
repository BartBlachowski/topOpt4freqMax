function T = sweep_table(prefix, outfile, footer)
%SWEEP_TABLE  One row per mesh of a sweep: results and per-iteration costs.
%
%   T = SWEEP_TABLE('S', 'repro/results/SWEEP_R06.md', footerText)
%   Rows are the runs repro/results/<prefix><nelx>x<nely>/res.mat.  Frequencies
%   are re-evaluated under SIMP + eq. (4) (cut-off 0.1) so that rows optimised
%   with different interpolations stay comparable; native values are kept too.

root = fileparts(fileparts(mfilename('fullpath')));
run(fullfile(root,'setpaths.m')); maxNumCompThreads(1);
resDir = fullfile(root,'repro','results');
d = dir(fullfile(resDir,[prefix '*x*'])); d = d([d.isdir]);
rows = {};
for i = 1:numel(d)
    f = fullfile(d(i).folder, d(i).name, 'res.mat');
    if ~isfile(f), continue; end
    L = load(f); res = L.res; cfg = L.cfg;
    flat = olh.config.toLegacy(cfg); mdl = model2D(flat); NE = mdl.nele;
    m4 = struct('model','eq4','q',1,'lowDensityExponent',6,'cutoff',0.1);
    [K,M] = assemble2D(mdl, res.rho, cfg.material.stiffness.p, m4); w4 = eigSolve(K,M,3,'eigs');
    h = res.hist; n = h.nInner;
    it99 = find(h.omega(1,:) >= 0.99*max(h.omega(1,:)), 1);
    rows(end+1,:) = {cfg.domain.mesh.nelx, cfg.domain.mesh.nely, NE, res.status, res.nOuter, sum(n), mean(n), ...
        res.omega(1), res.omega(2), res.omega(3), w4(1), w4(2), w4(3), 100*(w4(2)-w4(1))/w4(1), ...
        4*mean(res.rho.*(1-res.rho)), it99, res.wallclock, sum(h.tEig)/res.nOuter, ...
        (sum(h.tEig)+sum(h.tGrad))/res.nOuter, sum(h.tInner)/sum(n), sum(h.tInner)/res.wallclock*100}; %#ok<AGROW>
end
[~,o] = sort(cell2mat(rows(:,3))); rows = rows(o,:);
T = cell2table(rows, 'VariableNames', {'nelx','nely','NE','status','outer','inner','innerPerOuter', ...
    'w1_native','w2_native','w3_native','w1_eq4','w2_eq4','w3_eq4','gap_pct','Mnd','iter99', ...
    'wall_s','tEig_per_outer_s','tOuterExclInner_per_outer_s','t_per_inner_s','innerShare_pct'});
writetable(T, [outfile(1:end-3) '.csv']);
fid = fopen(outfile,'w');
fprintf(fid,'| mesh | NE | status | outer | inner | inner/outer | w1/w2/w3 (SIMP+eq.4) | gap | M_nd | w1 settled @ | wall [s] | eig/outer [s] | inner iter [s] | inner share |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n');
for i = 1:size(rows,1)
    r = rows(i,:);
    fprintf(fid,'| %dx%d | %d | %s | %d | %d | %.1f | %.1f / %.1f / %.1f | %.1f%% | %.3f | %d | %.0f | %.3f | %.3f | %.0f%% |\n', ...
        r{1}, r{2}, r{3}, r{4}, r{5}, r{6}, r{7}, r{11}, r{12}, r{13}, r{14}, r{15}, r{16}, r{17}, r{18}, r{20}, r{21});
end
if nargin < 3, footer = 'Preset duOlhoffAdaptivePedersen, physical radius 0.06, single thread.'; end
fprintf(fid,'\n%s Paper: 174.7 / 174.7 / 284.9.\n', footer);
fclose(fid);
disp(T); fprintf('wrote %s\n', outfile);
end
