function verify_comparator_counts()
%VERIFY_COMPARATOR_COUNTS Re-run representative comparator cases on current code.

thisDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(thisDir));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'examples', 'Performance'));

jsonPath = fullfile(repoRoot, 'examples', 'Performance', 'performance_comparison.json');
data = jsondecode(fileread(jsonPath));
data.postprocessing.visualize_live = false;
data.postprocessing.save_final_image = false;
data.postprocessing.save_snapshot_image = false;
data.postprocessing.save_frequency_iterations = false;
data.optimization.filter.radius = 2;
data.optimization.filter.radius_units = 'element';

cases = {
    'OurApproach', 160, 20;
    'OurApproach', 240, 30;
    'OurApproach', 320, 40;
    'OurApproach', 400, 50;
    'Olhoff',      160, 20;
    'Olhoff',      320, 40;
};

T = table('Size', [size(cases,1), 6], ...
    'VariableTypes', {'string','double','double','double','double','double'}, ...
    'VariableNames', {'Approach','Nelx','Nely','Iterations','Omega1','TimePerIteration'});
for k = 1:size(cases,1)
    data.optimization.approach = cases{k,1};
    data.domain.mesh.nelx = cases{k,2};
    data.domain.mesh.nely = cases{k,3};
    fprintf('[VERIFY] %s %dx%d\n', cases{k,1}, cases{k,2}, cases{k,3});
    commandOutput = evalc( ...
        '[~, omega, tIter, nIter, ~, ~] = run_topopt_from_json(data);'); %#ok<NASGU>
    T.Approach(k) = string(cases{k,1});
    T.Nelx(k) = cases{k,2};
    T.Nely(k) = cases{k,3};
    T.Iterations(k) = nIter;
    T.Omega1(k) = omega(1);
    T.TimePerIteration(k) = tIter;
    writetable(T(1:k,:), fullfile(thisDir, 'results', 'comparator_verification.csv'));
end
disp(T);
end
