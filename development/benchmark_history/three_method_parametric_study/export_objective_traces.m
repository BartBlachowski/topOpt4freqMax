function export_objective_traces()
%EXPORT_OBJECTIVE_TRACES  WP12/WP22 per-iteration native objective and cost.
%
%   Writes one long-format CSV holding, for every Stage A run, the native
%   objective and the measured cumulative optimization-loop time at each
%   iteration.  Native objectives differ by method and are NOT comparable
%   across methods: Olhoff maximises omega_1 while Yuksel and Proposed
%   minimise an inertial compliance.  The column `direction` records which,
%   and the within-method progress fraction defined in the report normalises
%   both to "fraction of the total improvement this run eventually made".

repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
study = fullfile(repo,'analysis','three_method_parametric_study');
out = fullfile(study,'results');
if ~exist(out,'dir'), mkdir(out); end
fid = fopen(fullfile(out,'objective_traces.csv'),'w');
fprintf(fid,'run_id,method,pass,direction,iter,stage,objective,loop_time_s,eig_time_cum_s,d_inf,d_rms,grayness\n');

f = dir(fullfile(study,'raw','stage_a','olhoff_move_*.mat'));
for i = 1:numel(f)
    S = load(fullfile(f(i).folder,f(i).name)); r = S.record;
    if ~strcmp(r.status,'COMPLETED_OBSERVER'), continue; end
    n = r.n_iter;
    loopCum = cumsum(r.hist.tEig(1:n)+r.hist.tGrad(1:n)+r.hist.tInner(1:n));
    eigCum  = cumsum(r.hist.tEig(1:n));
    for k = 1:n
        fprintf(fid,'%s,Olhoff,stage_a,maximize,%d,1,%.10g,%.6f,%.6f,%.10g,%.10g,NaN\n', ...
            r.run_id,k,r.hist.omega(1,k),loopCum(k),eigCum(k), ...
            r.hist.dxOuter(k),r.telemetry.d_rms(k));
    end
end

for pass = {'stage_a','stage_a_v2'}
    d = fullfile(study,'raw',pass{1});
    if ~exist(d,'dir'), continue; end
    g = dir(fullfile(d,'*.mat'));
    for i = 1:numel(g)
        if startsWith(g(i).name,'olhoff'), continue; end
        S = load(fullfile(g(i).folder,g(i).name)); r = S.record;
        if ~strcmp(r.status,'COMPLETED_OBSERVER'), continue; end
        h = r.telemetry.history;
        for k = 1:h.n
            fprintf(fid,'%s,%s,%s,minimize,%d,%d,%.10g,%.6f,NaN,%.10g,%.10g,%.10g\n', ...
                r.run_id,r.method,pass{1},h.iter(k),h.stage(k),h.objective(k), ...
                h.elapsed_s(k),h.d_inf(k),h.d_rms(k),h.grayness(k));
        end
    end
end
fclose(fid);
fprintf('wrote objective_traces.csv\n');
end
