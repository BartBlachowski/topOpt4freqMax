function r3_table2_fixed(nRep, meshIdx)
%R3_TABLE2_FIXED  Table 2 rerun with the lower-triangle assemble_KM: proposed and
%   Yuksel-Yilmaz, simply supported beam, V_f = 0.5, native settings, stagnation_x
%   (W = 10, 1e-2), single thread.  Per mesh and method one discarded warm-up and nRep
%   (default 5) timed runs.  Per run all Table 2 columns are recorded:
%   Count 1, Count 2 (proposed: 1 and n_II; YY: n1 and n2), Time 1, Time 2 (proposed:
%   reference eigenanalysis and static loop; YY: Stage 1 and Stage 2), Stage time =
%   Time 1 + Time 2, and Total = tic/toc around the solver call plus the final modal
%   analysis (common evaluator, 2 modes), as in r3_w2_timing.
%   Output: results/r3/table2_fixed_runs.csv (the medians are formed afterwards).
if nargin < 1 || isempty(nRep), nRep = 5; end
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
maxNumCompThreads(1);
out = results_dir('r3');
meshes = [160 20; 240 30; 320 40; 400 50; 480 60; 560 70; 640 80; 720 90; 800 100];
if nargin < 2, meshIdx = []; end
if ~isempty(meshIdx), meshes = meshes(meshIdx, :); end
sfx = '';  if ~isempty(meshIdx), sfx = sprintf('_%d', meshIdx); end
fn = fullfile(out, ['table2_fixed_runs' sfx '.csv']);
fid = fopen(fn, 'w');
fprintf(fid, 'mesh,Ne,method,run,count1,count2,time1_s,time2_s,stage_s,total_s,solver_total_s,omega1,omega2\n');
for k = 1:size(meshes, 1)
    prob = problem_ss_beam(meshes(k,1), meshes(k,2));
    for m = 1:2
        for r = 0:nRep
            t0 = tic;
            if m == 1, R = proposed_freq(prob); else, R = yuksel_freq(prob); end
            o = structural_modes(prob, R.x, 2);
            tw = toc(t0);
            if m == 1
                name = 'proposed';  c = [1 R.iterations];  t = [R.time.reference R.time.loop];
            else
                name = 'yuksel';  c = R.iterations(1:2);  t = [R.time.stage1 R.time.stage2];
            end
            fprintf('%4dx%-3d %-8s run %d  count %d+%d  T1 %.3f T2 %.3f stage %.3f total %.3f  omega %.4f %.4f\n', ...
                meshes(k,:), name, r, c, t, sum(t), tw, o);
            if r == 0, continue; end
            fprintf(fid, '%dx%d,%d,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', meshes(k,:), prod(meshes(k,:)), ...
                name, r, c, t, sum(t), tw, R.time.total, o);
        end
    end
end
fclose(fid);
end
