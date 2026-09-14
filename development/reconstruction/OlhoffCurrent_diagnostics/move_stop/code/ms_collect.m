function [runs, A] = ms_collect()
%MS_COLLECT  Load every diagnostic run, analyse, write METRICS.json and figures.
repo = '/Users/piotrek/Programming/topOpt4freqMax';
base = fullfile(repo,'analysis','OlhoffCurrent','diagnostics','move_stop');
addpath(fullfile(base,'code'));
addpath(fullfile(repo,'analysis','OlhoffCurrent'));

want = {'baseline',160,20; 'baseline',320,40; 'fixedmove',160,20; 'fixedmove',320,40};
runs = [];
for i = 1:size(want,1)
    f = fullfile(base,'runs', sprintf('%s_%dx%d.mat', want{i,1}, want{i,2}, want{i,3}));
    if exist(f,'file') ~= 2
        fprintf('  MISSING: %s\n', f); continue
    end
    S = load(f, 'out');
    if isempty(runs); runs = S.out; else; runs(end+1) = S.out; end %#ok<AGROW>
end
fprintf('loaded %d run(s)\n', numel(runs));

A = ms_analyze(runs);

% ---- METRICS.json ------------------------------------------------------
M = struct();
M.schema = 'olhoff_move_stop_diagnostic/1';
M.generated = char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ssXXX','TimeZone','local'));
M.implementation = 'analysis/OlhoffCurrent';
M.production_preset = 'duOlhoffFixedPenaltySensitivityFiltered';
M.matlab = version;
M.threads = maxNumCompThreads();
M.source_tree_sha256 = runs(1).sourceTree;
M.main_repo_commit = local_git(repo);
M.runs = struct('key',{},'arm',{},'label',{},'mesh',{},'NE',{},'status',{}, ...
    'converged',{},'nOuter',{},'innerTotal',{},'omega1',{},'omega2',{},'gap12',{}, ...
    'volume',{},'Mnd_pct',{},'gray_fraction',{},'mid_fraction',{}, ...
    'tolOuter',{},'epsRMS',{},'l2_final',{},'l2_over_tol',{},'maxAbs_final',{}, ...
    'nActive_final',{},'wall_s',{},'cfgHash',{});
for i = 1:numel(runs)
    r = runs(i); P = r.per;
    M.runs(end+1) = struct('key', sprintf('%s_%dx%d', r.arm, r.mesh(1), r.mesh(2)), ...
        'arm', r.arm, 'label', r.label, 'mesh', r.mesh, 'NE', r.NE, ...
        'status', r.status, 'converged', r.converged, 'nOuter', r.nOuter, ...
        'innerTotal', r.innerTotal, 'omega1', r.omega(1), 'omega2', r.omega(2), ...
        'gap12', P.gap12(end), 'volume', r.volume_final, 'Mnd_pct', r.Mnd_final, ...
        'gray_fraction', r.gray_final, 'mid_fraction', r.mid_final, ...
        'tolOuter', r.tolOuter, 'epsRMS', r.epsRMS, 'l2_final', P.l2(end), ...
        'l2_over_tol', P.l2(end)/r.tolOuter, 'maxAbs_final', P.maxAbs(end), ...
        'nActive_final', P.nActive(end,:), 'wall_s', r.wall_s, 'cfgHash', r.cfgHash);
end
M.transitions = A.transitions;
M.activeSet   = A.activeSet;
M.grayness    = A.grayness;
M.stability   = A.stability;
fid = fopen(fullfile(base,'METRICS.json'),'w');
fprintf(fid,'%s\n', jsonencode(M,'PrettyPrint',true)); fclose(fid);
fprintf('wrote METRICS.json\n');

% ---- per-iteration CSVs (full recorders, for re-analysis) --------------
for i = 1:numel(runs)
    r = runs(i); P = r.per;
    T = table(P.outer, P.omega1, P.omega2, P.gap12, P.volume, P.Mnd, P.gray, P.mid, ...
        P.move, P.stage, P.beta, P.l2, P.rms, P.maxAbs, P.tolOuter, P.epsRMS, ...
        double(P.stopRaw), double(P.settled), double(P.stopAdmitted), double(P.descent), ...
        P.nActive(:,1), P.nActive(:,2), P.nActive(:,3), P.nActive(:,4), ...
        P.nInner, double(P.innerConv), P.multN, P.degen, ...
        'VariableNames', {'outer','omega1','omega2','gap12','volume','Mnd_pct','gray_frac', ...
        'mid_frac','move','stage','beta','l2','rms','maxAbs','tolOuter','epsRMS', ...
        'stopRaw','settledMove','stopAdmitted','moveDescent', ...
        'nActive_epsRMS','nActive_1e4','nActive_1e3','nActive_1e2', ...
        'nInner','innerConv','multN','degen'});
    writetable(T, fullfile(base,'runs', sprintf('%s_%dx%d_iterations.csv', ...
        r.arm, r.mesh(1), r.mesh(2))));
end
fprintf('wrote per-iteration CSVs\n');

ms_figures(runs, fullfile(base,'figures'));
end

function h = local_git(repo)
[st,out] = system(sprintf('git -C "%s" rev-parse HEAD', repo));
if st==0; h = strtrim(out); else; h = 'unknown'; end
end
