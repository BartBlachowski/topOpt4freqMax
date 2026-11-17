function r3_fig8_fixed(csvFixed, metaJson, outDir, researchRepo)
%R3_FIG8_FIXED  Figure 8 from the fixed-assembly Table 2: stage time (Time 1 + Time 2)
%   versus N_e with the paper's plotting code (confbench_complexity_plots.m core:
%   fit_complexity_model + plot_table1_complexity of the research repository, same
%   titles, labels, colours and fixed exponent 1.5).  Du-Olhoff: full-precision stage
%   times of the PAPER campaign (complexity_stage_time_metadata.json); Yuksel-Yilmaz and
%   proposed: medians of 5 runs (table2_fixed.csv).
addpath(fullfile(researchRepo, 'examples', 'Performance', 'conference_bench'));
meta = jsondecode(fileread(metaJson));
Ne = meta.elements(:);
T = readtable(csvFixed, 'TextType', 'string');
t = NaN(numel(Ne), 3);  t(:,1) = meta.stage_time_s(:,1);
names = ["Yuksel-Yilmaz", "Proposed"];
for m = 1:2
    s = T(T.Method == names(m), :);
    for r = 1:numel(Ne)
        k = find(s.Ne == Ne(r));  t(r, m+1) = s.StageTime_s(k);
    end
end
labels = {'Du-Olhoff', 'Yuksel-Yilmaz', 'Proposed'};  fitMask = true(size(t));
[Cf, pf, R2f, nf] = fit_complexity_model(Ne, t, 'free');
[Cx, px, R2x, nx] = fit_complexity_model(Ne, t, 'fixed', 1.5);
for m = 1:3
    fprintf('Figure 8 %-14s free: C = %.4e  b = %.4f  R2 = %.4f (n=%d) | fixed 1.5: C = %.4e  R2 = %.4f\n', ...
        labels{m}, Cf(m), pf(m), R2f(m), nf(m), Cx(m), R2x(m));
end
plot_table1_complexity(Ne, labels, t, Cf, pf, outDir, 'table1_complexity_fit', ...
    'Stage-time complexity fit:  T(N_e) = C \cdot N_e^{exp}', fitMask, 'Stage time T = Time 1 + Time 2 (s)');
plot_table1_complexity(Ne, labels, t, Cx, px, outDir, 'table1_complexity_fit_fixedexp', ...
    sprintf('Stage-time fixed-exponent fit (C estimated only):  T(N_e) = C \\cdot N_e^{%.2f}', 1.5), fitMask, ...
    'Stage time T = Time 1 + Time 2 (s)');
fid = fopen(fullfile(outDir, 'fig8_fits.csv'), 'w');
fprintf(fid, 'method,C_free,b_free,R2_free,C_fixed15,R2_fixed15\n');
for m = 1:3, fprintf(fid, '%s,%.6e,%.6f,%.6f,%.6e,%.6f\n', labels{m}, Cf(m), pf(m), R2f(m), Cx(m), R2x(m)); end
fclose(fid);
end
