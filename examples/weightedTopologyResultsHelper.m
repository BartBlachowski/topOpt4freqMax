function results = weightedTopologyResultsHelper(data,basename,res_str,forms_str)

    runs = { ...
        '1',  1.00, 0.00; ...
        '75', 0.75, 0.25; ...
        '5',  0.50, 0.50; ...
        '25', 0.25, 0.75; ...
        '0',  0.00, 1.00  ...
    };

    results = [];
    for i = 1:size(runs, 1)
        tag = runs{i, 1};
        data.domain.load_cases(1).factor = runs{i, 2};
        data.domain.load_cases(2).factor = runs{i, 3};

        % Ensure approach override uses the correct field spelling.
        data = localSetApproach(data, 'ourApproach');

        [x3, omega3, tIter3, nIter3, mem3] = run_topopt_from_json(data);
        results = [results; packResultRow(x3, omega3, tIter3, nIter3, mem3)];

        outPrefix = string(basename) + "_" + string(tag) + "_" + string(res_str) + string(forms_str);
        localCaptureArtifacts(data, res_str, outPrefix);
    end
end

function row = packResultRow(x, omega, tIter, nIter, mem)
    row = [reshape(x, 1, []), ...
           reshape(omega, 1, []), ...
           reshape(tIter, 1, []), ...
           reshape(nIter, 1, []), ...
           reshape(mem, 1, [])];
end

function data = localSetApproach(data, approachName)
    if isfield(data, 'optimization')
        data.optimization.approach = approachName;
    elseif isfield(data, 'optimisation')
        data.optimisation.approach = approachName;
    else
        data.optimization = struct('approach', approachName);
    end
end

function localCaptureArtifacts(data, resStr, outPrefix)
    approachName = localGetApproachName(data);
    nameSafe = regexprep(char(string(approachName)), '[^\w\-]', '_');

    cwdFolder = pwd;
    repoRoot = localRepoRootFromRunner();
    resultsDir = fullfile(repoRoot, 'results');

    localMoveIfExists( ...
        fullfile(cwdFolder, sprintf('%s_%s.png', nameSafe, resStr)), ...
        fullfile(cwdFolder, char(outPrefix + ".png")));
    localMoveIfExists( ...
        fullfile(cwdFolder, sprintf('%s_%s.fig', nameSafe, resStr)), ...
        fullfile(cwdFolder, char(outPrefix + ".fig")));

    % Figure-6-style frequency history (saved by run_topopt_from_json to <repo>/results).
    freqStem = sprintf('%s_%s_freq_iterations', nameSafe, resStr);
    freqPngMoved = localMoveIfExists( ...
        fullfile(resultsDir, [freqStem '.png']), ...
        fullfile(cwdFolder, char(outPrefix + "_freq_iterations.png")));
    freqFigMoved = localMoveIfExists( ...
        fullfile(resultsDir, [freqStem '.fig']), ...
        fullfile(cwdFolder, char(outPrefix + "_freq_iterations.fig")));
    if ~(freqPngMoved || freqFigMoved)
        warning('weightedTopologyResultsHelper:MissingFreqIterations', ...
            'Frequency-iteration artifacts were not found at %s.', fullfile(resultsDir, freqStem));
    end

    localMoveIfExists( ...
        fullfile(cwdFolder, 'topopt_config_topology_mode_1.png'), ...
        fullfile(cwdFolder, char(outPrefix + "_topopt_config_topology_mode_1.png")));
    localMoveIfExists( ...
        fullfile(cwdFolder, 'topopt_config_topology_mode_2.png'), ...
        fullfile(cwdFolder, char(outPrefix + "_topopt_config_topology_mode_2.png")));
    localMoveIfExists( ...
        fullfile(cwdFolder, 'topopt_config_correlation.csv'), ...
        fullfile(cwdFolder, char(outPrefix + "_correlation.csv")));
end

function tf = localMoveIfExists(srcPath, dstPath)
    tf = false;
    if exist(srcPath, 'file') ~= 2
        return;
    end
    [ok, msg] = movefile(srcPath, dstPath, 'f');
    if ~ok
        warning('weightedTopologyResultsHelper:MoveFailed', ...
            'Failed to move "%s" to "%s": %s', srcPath, dstPath, msg);
        return;
    end
    tf = true;
end

function approachName = localGetApproachName(data)
    approachName = 'ourApproach';
    if isfield(data, 'optimization') && isfield(data.optimization, 'approach') ...
            && ~isempty(data.optimization.approach)
        approachName = char(string(data.optimization.approach));
        return;
    end
    if isfield(data, 'optimisation') && isfield(data.optimisation, 'approach') ...
            && ~isempty(data.optimisation.approach)
        approachName = char(string(data.optimisation.approach));
    end
end

function repoRoot = localRepoRootFromRunner()
    runnerPath = which('run_topopt_from_json');
    if isempty(runnerPath)
        repoRoot = pwd;
        return;
    end
    repoRoot = fileparts(fileparts(fileparts(runnerPath)));
end
