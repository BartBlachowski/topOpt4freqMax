function [x, omega, tIter, nIter, mem_usage] = run_topopt_from_json(jsonInput)
%RUN_TOPOPT_FROM_JSON Run selected topology-optimization approach from JSON task file.
%
%   [x, omega, tIter, nIter, mem_usage] = run_topopt_from_json(jsonInput)
%
% Inputs:
%   jsonInput : either path to JSON task file, or a decoded JSON struct.
%
% Outputs:
%   x         : final density vector (nelx*nely, 1)
%   omega     : first three circular frequencies in rad/s (3,1)
%   tIter     : average time per optimization iteration [s]
%   nIter     : number of optimization iterations executed
%   mem_usage : peak additional memory used during this call [MB]

    ensureCompatHelpersOnPath();

    if nargin < 1 || isempty(jsonInput)
        error('run_topopt_from_json:MissingInput', 'Input is required (JSON filename or decoded struct).');
    end
    jsonSource = '';
    if isstruct(jsonInput)
        if numel(jsonInput) ~= 1
            error('run_topopt_from_json:InvalidInputStruct', ...
                'Decoded JSON input must be a scalar struct.');
        end
        cfg = jsonInput;
    else
        if isstring(jsonInput)
            jsonInput = char(jsonInput);
        end
        if ~ischar(jsonInput)
            error('run_topopt_from_json:InvalidInputType', ...
                'Input must be a JSON filename (char/string) or decoded struct.');
        end
        if ~isfile(jsonInput)
            error('run_topopt_from_json:FileNotFound', 'JSON file not found: %s', jsonInput);
        end
        jsonSource = jsonInput;
        cfg = jsondecode(fileread(jsonInput));
    end

    % -------- Required fields --------
    L = reqNum(cfg, {'domain','size','length'}, 'domain.size.length');
    H = reqNum(cfg, {'domain','size','height'}, 'domain.size.height');
    nelx = reqInt(cfg, {'domain','mesh','nelx'}, 'domain.mesh.nelx');
    nely = reqInt(cfg, {'domain','mesh','nely'}, 'domain.mesh.nely');
    thickness = reqNum(cfg, {'domain','thickness'}, 'domain.thickness');

    E0 = reqNum(cfg, {'material','E'}, 'material.E');
    nu = reqNum(cfg, {'material','nu'}, 'material.nu');
    rho0 = reqNum(cfg, {'material','rho'}, 'material.rho');
    EminRatio = reqNum(cfg, {'void_material','E_min_ratio'}, 'void_material.E_min_ratio');
    rho_min = reqNum(cfg, {'void_material','rho_min'}, 'void_material.rho_min');

    approach = reqStr(cfg, {'optimization','approach'}, 'optimization.approach');
    volfrac = reqNum(cfg, {'optimization','volume_fraction'}, 'optimization.volume_fraction');
    penal = reqNum(cfg, {'optimization','penalization'}, 'optimization.penalization');
    move = reqNum(cfg, {'optimization','move_limit'}, 'optimization.move_limit');
    maxiter = reqInt(cfg, {'optimization','max_iters'}, 'optimization.max_iters');
    convTol = reqNum(cfg, {'optimization','convergence_tol'}, 'optimization.convergence_tol');

    filterType = reqStr(cfg, {'optimization','filter','type'}, 'optimization.filter.type');
    filterRadius = reqNum(cfg, {'optimization','filter','radius'}, 'optimization.filter.radius');
    radiusUnits = lower(reqStr(cfg, {'optimization','filter','radius_units'}, 'optimization.filter.radius_units'));
    filterBC = reqStr(cfg, {'optimization','filter','boundary_condition'}, 'optimization.filter.boundary_condition');

    optimizerType = 'OC';
    if hasFieldPath(cfg, {'optimization','optimizer'})
        optimizerType = upper(strtrim(reqStr(cfg, {'optimization','optimizer'}, 'optimization.optimizer')));
        if ~any(strcmp(optimizerType, {'OC','MMA'}))
            error('run_topopt_from_json:InvalidOptimizer', ...
                'optimization.optimizer must be "OC" or "MMA" (got "%s").', optimizerType);
        end
    end

    supports = reqStructArray(cfg, {'bc','supports'}, 'bc.supports');
    validateSupportEntries(supports);
    [supportCode, ~, ~] = mapSupportsToCode(supports);
    [extraFixedDofs, ~] = supportsToFixedDofs(supports, nelx, nely, L, H);
    [pasS, pasV] = parsePassiveRegions(cfg, nelx, nely, L, H);
    tipMassFrac = parseTipMassFraction(cfg, volfrac, L, H, rho0);
    loadCasesOur = [];
    if hasFieldPath(cfg, {'domain','load_cases'})
        loadCasesOur = validateLoadCases(getFieldPath(cfg, {'domain','load_cases'}), 'domain.load_cases');
    elseif isfield(cfg, 'loads') && ~isempty(cfg.loads)
        legacySingleCase = struct('name', 'legacy', 'factor', 1.0, 'loads', cfg.loads);
        loadCasesOur = validateLoadCases(legacySingleCase, 'loads');
    end

    % --- Reject deprecated keys and parse postprocessing block ---
    rejectDeprecatedKeys(cfg);
    postproc = parsePostprocessingBlock(cfg);
    hasDebugSemiHarmonic = hasFieldPath(cfg, {'optimization','debug_semi_harmonic'});
    debugSemiHarmonic = false;
    if hasDebugSemiHarmonic
        debugSemiHarmonic = parseBool( ...
            getFieldPath(cfg, {'optimization','debug_semi_harmonic'}), ...
            'optimization.debug_semi_harmonic');
    end
    hasSemiHarmonicBaseline = hasFieldPath(cfg, {'optimization','semi_harmonic_baseline'});
    semiHarmonicBaseline = '';
    if hasSemiHarmonicBaseline
        semiHarmonicBaseline = char(string(getFieldPath(cfg, {'optimization','semi_harmonic_baseline'})));
    end
    hasSemiHarmonicRhoSource = hasFieldPath(cfg, {'optimization','semi_harmonic_rho_source'});
    semiHarmonicRhoSource = '';
    if hasSemiHarmonicRhoSource
        semiHarmonicRhoSource = char(string(getFieldPath(cfg, {'optimization','semi_harmonic_rho_source'})));
    end

    % Radius conversion requested by task description.
    dx = L / nelx;
    switch radiusUnits
        case 'element'
            rmin_elem = filterRadius;
            rmin_phys = filterRadius * dx;
        case 'physical'
            rmin_elem = filterRadius / dx;
            rmin_phys = filterRadius;
        otherwise
            error('run_topopt_from_json:InvalidFilterUnits', ...
                'optimization.filter.radius_units must be "element" or "physical".');
    end
    if rmin_elem <= 0 || rmin_phys <= 0
        error('run_topopt_from_json:InvalidFilterRadius', 'Filter radius must be positive.');
    end

    % Validate common numeric ranges.
    assertPositive(L, 'domain.size.length');
    assertPositive(H, 'domain.size.height');
    assertPositive(thickness, 'domain.thickness');
    assertPositive(E0, 'material.E');
    if ~(nu > 0 && nu < 0.5)
        error('run_topopt_from_json:InvalidNu', 'material.nu must be in (0, 0.5).');
    end
    assertPositive(rho0, 'material.rho');
    if ~(EminRatio > 0 && EminRatio <= 1)
        error('run_topopt_from_json:InvalidEminRatio', 'void_material.E_min_ratio must be in (0, 1].');
    end
    assertPositive(rho_min, 'void_material.rho_min');
    if ~(volfrac > 0 && volfrac <= 1)
        error('run_topopt_from_json:InvalidVolumeFraction', 'optimization.volume_fraction must be in (0, 1].');
    end
    assertPositive(penal, 'optimization.penalization');
    assertPositive(move, 'optimization.move_limit');
    if move > 1
        error('run_topopt_from_json:InvalidMove', 'optimization.move_limit must be <= 1.');
    end
    assertPositive(maxiter, 'optimization.max_iters');
    assertPositive(convTol, 'optimization.convergence_tol');

    % This file lives in tools/Matlab/, so repo root is three levels up.
    repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));

    x = [];
    omega = NaN(3,1);
    tIter = NaN;
    nIter = NaN;
    mem_usage = 0;
    Emin = E0 * EminRatio;
    freqIterOmega = [];

    % --- Memory sampling setup (only when 5th output requested) ---
    if nargout >= 5
        baselineRSS = getCurrentRSS_KB();
        setappdata(0, 'topopt_peakRSS_KB', baselineRSS);  % shared mutable store
        samplerTimer = timer('ExecutionMode', 'fixedRate', ...
                             'Period', 0.1, ...
                             'TimerFcn', @(~,~) sampleRSS());
        cleanupObj = onCleanup(@() stopAndDeleteTimer(samplerTimer));
        start(samplerTimer);
    end

    % Start each run with a fresh plotting session so repeated
    % run_topopt_from_json calls open separate figure windows.
    resetTopologyPlotSession();

    switch lower(strtrim(approach))
        case 'olhoff'
            addpath(fullfile(repoRoot, 'OlhoffApproach', 'Matlab'));

            cfgO = struct();
            cfgO.L = L;
            cfgO.H = H;
            cfgO.nelx = nelx;
            cfgO.nely = nely;
            cfgO.t = thickness;
            cfgO.volfrac = volfrac;
            cfgO.penal = penal;
            cfgO.rmin = rmin_phys; % Olhoff solver expects physical units.
            cfgO.maxiter = maxiter;
            cfgO.conv_tol = convTol;
            cfgO.supportType = supportCode;
            cfgO.E0 = E0;
            cfgO.Emin = Emin;
            cfgO.rho0 = rho0;
            cfgO.rho_min = rho_min;
            cfgO.nu = nu;
            cfgO.move = move;
            cfgO.extraFixedDofs = extraFixedDofs;
            cfgO.pasS = pasS;
            cfgO.pasV = pasV;

            optsO = struct('doDiagnostic', true, 'diagnosticOnly', false, 'diagModes', 5);
            optsO.approach_name = approach;
            optsO.save_frq_iterations = postproc.saveFrequencyIterations;
            optsO.visualization_quality = postproc.visualizeQuality;
            if ~isempty(postproc.visualizeLive)
                optsO.visualize_live = postproc.visualizeLive;
            end

            [~, xPhys, diagnostics] = topFreqOptimization_MMA(cfgO, optsO);
            x = xPhys(:);
            if isfield(diagnostics, 'final') && isfield(diagnostics.final, 'omega')
                omega = toVec3(diagnostics.final.omega(:));
            elseif isfield(diagnostics, 'final') && isfield(diagnostics.final, 'freq')
                omega = toVec3(2*pi*diagnostics.final.freq(:));
            end
            if isfield(diagnostics, 't_iter')
                tIter = diagnostics.t_iter;
            end
            if isfield(diagnostics, 'iterations')
                nIter = diagnostics.iterations;
            end
            if postproc.saveFrequencyIterations && isfield(diagnostics, 'freq_iter_omega')
                freqIterOmega = diagnostics.freq_iter_omega;
            end

        case 'yuksel'
            addpath(fullfile(repoRoot, 'YukselApproach', 'Matlab'));

            bcType = mapSupportCodeToYuksel(supportCode);
            [ft, ftBC] = mapFilterToYuksel(filterType, filterBC);

            runCfg = struct();
            runCfg.E0 = E0;
            runCfg.Emin = Emin;
            runCfg.nu = nu;
            runCfg.rho0 = rho0;
            runCfg.rho_min = rho_min;
            runCfg.beamL = L;
            runCfg.beamH = H;
            runCfg.conv_tol = convTol;
            runCfg.approach_name = approach;
            runCfg.save_frq_iterations = postproc.saveFrequencyIterations;
            runCfg.visualization_quality = postproc.visualizeQuality;
            if ~isempty(tipMassFrac)
                runCfg.tipMassFrac = tipMassFrac;
            end
            if ~isempty(postproc.visualizeLive)
                runCfg.visualize_live = postproc.visualizeLive;
            end
            runCfg.extraFixedDofs = extraFixedDofs;
            runCfg.pasS = pasS;
            runCfg.pasV = pasV;

            eta = 0.5;
            beta = 1.0;
            stage1MaxIter = min(maxiter, 200);
            if hasFieldPath(cfg, {'optimization','yuksel','stage1_max_iters'})
                stage1MaxIter = reqInt(cfg, {'optimization','yuksel','stage1_max_iters'}, ...
                    'optimization.yuksel.stage1_max_iters');
            end
            if stage1MaxIter < 1
                error('run_topopt_from_json:InvalidYukselStage1MaxIters', ...
                    'optimization.yuksel.stage1_max_iters must be >= 1.');
            end
            stage1MaxIter = min(stage1MaxIter, maxiter);

            nHistModes = 0;
            if hasFieldPath(cfg, {'optimization','yuksel','mode_history_modes'})
                nHistModes = reqInt(cfg, {'optimization','yuksel','mode_history_modes'}, ...
                    'optimization.yuksel.mode_history_modes');
            end
            if nHistModes < 0
                error('run_topopt_from_json:InvalidYukselModeHistoryModes', ...
                    'optimization.yuksel.mode_history_modes must be >= 0.');
            end

            finalModeCount = 3;
            if hasFieldPath(cfg, {'optimization','yuksel','final_mode_count'})
                finalModeCount = reqInt(cfg, {'optimization','yuksel','final_mode_count'}, ...
                    'optimization.yuksel.final_mode_count');
            end
            if finalModeCount < 1
                error('run_topopt_from_json:InvalidYukselFinalModeCount', ...
                    'optimization.yuksel.final_mode_count must be >= 1.');
            end
            runCfg.final_modes = finalModeCount;
            if hasFieldPath(cfg, {'optimization','yuksel','stage1_tol'})
                runCfg.stage1_tol = reqNum(cfg, {'optimization','yuksel','stage1_tol'}, ...
                    'optimization.yuksel.stage1_tol');
                assertPositive(runCfg.stage1_tol, 'optimization.yuksel.stage1_tol');
            end
            if hasFieldPath(cfg, {'optimization','yuksel','stage2_tol'})
                runCfg.stage2_tol = reqNum(cfg, {'optimization','yuksel','stage2_tol'}, ...
                    'optimization.yuksel.stage2_tol');
                assertPositive(runCfg.stage2_tol, 'optimization.yuksel.stage2_tol');
            end

            [xPhysStage2, ~, info] = top99neo_inertial_freq( ...
                nelx, nely, volfrac, penal, rmin_elem, ft, ftBC, eta, beta, move, ...
                maxiter, stage1MaxIter, bcType, nHistModes, runCfg);

            x = xPhysStage2(:);
            if isfield(info, 'stage2') && isfield(info.stage2, 'omegaFinal') && ~isempty(info.stage2.omegaFinal)
                omega = toVec3(info.stage2.omegaFinal(:));
            elseif isfield(info, 'stage2') && isfield(info.stage2, 'omegaHist') && ~isempty(info.stage2.omegaHist)
                w = info.stage2.omegaHist(end, :);
                omega = toVec3(w(:));
            elseif isfield(info, 'stage2') && isfield(info.stage2, 'omega1')
                omega(1) = info.stage2.omega1;
            end
            if isfield(info, 'timing') && isfield(info.timing, 't_iter')
                tIter = info.timing.t_iter;
            end
            if isfield(info, 'timing') && isfield(info.timing, 'total_iterations')
                nIter = info.timing.total_iterations;
            end
            if postproc.saveFrequencyIterations && isfield(info, 'freq_iter_omega')
                freqIterOmega = info.freq_iter_omega;
            end

        case 'ourapproach'
            addpath(fullfile(repoRoot, 'ourApproach', 'Matlab'));

            ft = mapFilterToOurApproach(filterType);
            runCfg = struct();
            runCfg.E0 = E0;
            runCfg.Emin = Emin;
            runCfg.nu = nu;
            runCfg.rho0 = rho0;
            runCfg.rho_min = rho_min;
            runCfg.move = move;
            runCfg.conv_tol = convTol;
            runCfg.max_iters = maxiter;
            runCfg.supportType = supportCode;
            runCfg.approach_name = approach;
            runCfg.save_frq_iterations = postproc.saveFrequencyIterations;
            runCfg.visualization_quality = postproc.visualizeQuality;
            if ~isempty(postproc.visualizeLive)
                runCfg.visualize_live = postproc.visualizeLive;
            end
            runCfg.extraFixedDofs = extraFixedDofs;
            runCfg.pasS = pasS;
            runCfg.pasV = pasV;
            if hasFieldPath(cfg, {'optimization','harmonic_normalize'})
                runCfg.harmonic_normalize = parseBool( ...
                    getFieldPath(cfg, {'optimization','harmonic_normalize'}), ...
                    'optimization.harmonic_normalize');
            end
            if hasDebugSemiHarmonic
                runCfg.debug_semi_harmonic = debugSemiHarmonic;
            end
            if hasSemiHarmonicBaseline
                runCfg.semi_harmonic_baseline = semiHarmonicBaseline;
            end
            if hasSemiHarmonicRhoSource
                runCfg.semi_harmonic_rho_source = semiHarmonicRhoSource;
            end
            if ~isempty(loadCasesOur)
                runCfg.load_cases = loadCasesOur;
            end
            runCfg.optimizer = optimizerType;

            if postproc.saveFrequencyIterations
                [xOut, fOut, tOut, itOut, infoOur] = topopt_freq( ...
                    nelx, nely, volfrac, penal, rmin_phys, ft, L, H, runCfg);
                if isstruct(infoOur) && isfield(infoOur, 'freq_iter_omega')
                    freqIterOmega = infoOur.freq_iter_omega;
                end
            else
                [xOut, fOut, tOut, itOut] = topopt_freq( ...
                    nelx, nely, volfrac, penal, rmin_phys, ft, L, H, runCfg);
            end
            x = xOut(:);
            omega = toVec3(2*pi*fOut(:)); % ourApproach runner reports Hz.
            tIter = tOut;
            nIter = itOut;

        otherwise
            error('run_topopt_from_json:UnknownApproach', ...
                'Unknown optimization.approach "%s". Use "Olhoff", "Yuksel", or "ourApproach".', approach);
    end

    % --- Finalize memory measurement ---
    if nargout >= 5
        sampleRSS();   % one last sample
        stopAndDeleteTimer(samplerTimer);
        peakRSS = getappdata(0, 'topopt_peakRSS_KB');
        mem_usage = max(0, peakRSS - baselineRSS) / 1024;  % KB -> MB
    end

    if isempty(x)
        error('run_topopt_from_json:NoResult', 'Solver did not return a final topology vector.');
    end
    x = x(:);
    omega = toVec3(omega(:));

    % === POSTPROCESSING — all after timing and memory measurement ===

    if postproc.saveFrequencyIterations
        if ~isempty(freqIterOmega)
            saveFrequencyIterationPlot(freqIterOmega, approach, nelx, nely, repoRoot);
        else
            warning('run_topopt_from_json:MissingFrequencyHistory', ...
                'save_frequency_iterations requested, but no iteration history was returned by "%s".', approach);
        end
    end

    % --- Compute required mode counts for both domains ---
    nTopo    = computeRequiredTopologyModes(postproc);
    nInitial = computeRequiredInitialModes(postproc);

    % --- Topology-domain eigensolve (ONE cached call for all consumers) ---
    modesT = struct('omega', [], 'phi', [], 'free', [], 'ndof', [], 'Mff', [], 'Mfull', []);
    if nTopo > 0
        fprintf('[Eigensolve] Computing topology domain modes (n=%d)...\n', nTopo);
        try
            [modesT.omega, modesT.phi, modesT.free, modesT.ndof, modesT.Mff, modesT.Mfull] = ...
                localSolveTopologyModes(x, nTopo, L, H, nelx, nely, thickness, ...
                    E0, Emin, nu, rho0, rho_min, penal, supportCode, extraFixedDofs);
            fprintf('[Eigensolve] Topology modes done: omega_1 = %.4f rad/s.\n', modesT.omega(1));
        catch eigenErr
            warning('run_topopt_from_json:TopologyEigenSolveFailed', ...
                'Topology eigensolve failed: %s', eigenErr.message);
        end
    end

    % --- Initial-domain eigensolve (ONE cached call for all consumers) ---
    modesI = struct('omega', [], 'phi', [], 'free', [], 'ndof', []);
    if nInitial > 0
        fprintf('[Eigensolve] Computing initial domain modes (n=%d)...\n', nInitial);
        try
            [modesI.omega, modesI.phi, modesI.free, modesI.ndof] = ...
                localSolveReferenceModes(nInitial, L, H, nelx, nely, thickness, ...
                    E0, Emin, nu, rho0, rho_min, volfrac, penal, ...
                    supportCode, extraFixedDofs);
            fprintf('[Eigensolve] Initial modes done: omega_1 = %.4f rad/s.\n', modesI.omega(1));
        catch eigenErr
            warning('run_topopt_from_json:InitialEigenSolveFailed', ...
                'Initial domain eigensolve failed: %s', eigenErr.message);
        end
    end

    % --- Correlation matrix (no eigensolve — uses cached eigenpairs only) ---
    corrMat = [];
    if postproc.correlation.enabled
        if ~isempty(modesI.phi) && ~isempty(modesT.phi) && ~isempty(modesT.Mfull)
            fprintf('[Correlation] Using full-DOF mass matrix (metric: %s).\n', ...
                postproc.correlation.metric);
            assert(modesT.ndof == modesI.ndof, 'run_topopt_from_json:NdofMismatch', ...
                'Initial and topology ndof mismatch (%d vs %d).', modesI.ndof, modesT.ndof);
            PhiFull = zeros(modesI.ndof, size(modesI.phi, 2));
            PhiFull(modesI.free, :) = modesI.phi;
            PsiFull = zeros(modesT.ndof, size(modesT.phi, 2));
            PsiFull(modesT.free, :) = modesT.phi;
            corrMat = computeCorrelationMatrixFullDOF( ...
                PhiFull, PsiFull, modesT.Mfull, postproc.correlation.metric);
            if postproc.correlation.writeBestMatches
                logCorrelationBestMatches(corrMat, modesI.omega, modesT.omega, ...
                    postproc.correlation.bestK);
            end
            if postproc.correlation.writeCsv
                [corrFolder, corrBase] = localResolveModeOutputTarget(jsonSource);
                csvName = postproc.correlation.csvFilename;
                if isempty(csvName)
                    csvName = [corrBase '_correlation.csv'];
                end
                writeCorrelationCSV(corrMat, modesI.omega, modesT.omega, ...
                    fullfile(corrFolder, csvName));
            end
            if postproc.correlation.writeHeatmap
                [heatFolder, heatBase] = localResolveModeOutputTarget(jsonSource);
                writeCorrelationHeatmap(corrMat, modesI.omega, modesT.omega, ...
                    fullfile(heatFolder, [heatBase '_correlation_heatmap.png']));
            end
        else
            warning('run_topopt_from_json:CorrelationMissingModes', ...
                'Correlation enabled but mode data unavailable for one or both domains.');
        end
    end

    % --- Topology snapshot (uses cached topology modes for omega labels) ---
    if postproc.saveFinalImage
        omega1snap = NaN;
        omega2snap = NaN;
        if numel(modesT.omega) >= 1, omega1snap = modesT.omega(1); end
        if numel(modesT.omega) >= 2, omega2snap = modesT.omega(2); end
        saveTopologySnapshot(x, nelx, nely, approach, jsonSource, omega1snap, ...
            postproc.visualizeQuality, omega2snap);
    end

    % --- Topology mode visualizations (optional correlation labels) ---
    if postproc.visualizeTopologyModes.enabled && ~isempty(modesT.omega)
        try
            saveTopologyModeVisualizationsFromCache( ...
                jsonSource, x, postproc.visualizeTopologyModes.count, ...
                L, H, nelx, nely, ...
                modesT.omega, modesT.phi, modesT.free, modesT.ndof, ...
                corrMat, postproc.correlation.useForTopologyModeLabels);
        catch modeErr
            warning('run_topopt_from_json:TopologyModeVisualizationFailed', ...
                'Topology mode visualization failed: %s', modeErr.message);
        end
    end

    % --- Initial-domain mode visualizations (from cache — no eigensolve) ---
    if postproc.visualizeModes.enabled && ~isempty(modesI.omega)
        try
            saveReferenceModeVisualizationsFromCache( ...
                jsonSource, postproc.visualizeModes.count, ...
                L, H, nelx, nely, ...
                modesI.omega, modesI.phi, modesI.free, modesI.ndof);
        catch modeErr
            warning('run_topopt_from_json:ModeVisualizationFailed', ...
                'Reference mode visualization failed: %s', modeErr.message);
        end
    end
end

function saveFrequencyIterationPlot(freqIterOmega, approachName, nelx, nely, repoRoot)
    if isempty(freqIterOmega)
        return;
    end
    if ~isnumeric(freqIterOmega)
        warning('run_topopt_from_json:InvalidFrequencyHistoryType', ...
            'Frequency history must be numeric to save iteration plot.');
        return;
    end

    freqIterOmega = double(freqIterOmega);
    nIter = size(freqIterOmega, 1);
    if nIter < 1
        return;
    end
    if size(freqIterOmega, 2) < 3
        tmp = NaN(nIter, 3);
        tmp(:,1:size(freqIterOmega,2)) = freqIterOmega;
        freqIterOmega = tmp;
    else
        freqIterOmega = freqIterOmega(:,1:3);
    end

    resultsDir = fullfile(repoRoot, 'results');
    if exist(resultsDir, 'dir') ~= 7
        mkdir(resultsDir);
    end

    nameRaw = char(string(approachName));
    nameSafe = regexprep(nameRaw, '[^\w\-]', '_');
    outPath = fullfile(resultsDir, sprintf('%s_%dx%d_freq_iterations.png', nameSafe, nelx, nely));

    fig = figure('Color', 'white', 'Visible', 'off');
    ax = axes('Parent', fig);
    hold(ax, 'on');
    colors = [0.0000, 0.4470, 0.7410; ...
              0.8500, 0.3250, 0.0980; ...
              0.4660, 0.6740, 0.1880];
    xIter = (1:nIter)';
    for j = 1:3
        plot(ax, xIter, freqIterOmega(:,j), '-', 'LineWidth', 1.6, ...
            'Color', colors(j,:), 'DisplayName', sprintf('\\omega_%d', j));
    end

    xlabel(ax, 'Outer iteration');
    ylabel(ax, 'Frequency (rad/s)');
    title(ax, sprintf('%s frequency history', nameRaw), 'Interpreter', 'none');
    grid(ax, 'on');
    box(ax, 'on');
    % MATLAB requires strictly increasing limits; handle single-iteration runs.
    if nIter == 1
        xlim(ax, [0.5, 1.5]);
    else
        xlim(ax, [1, nIter]);
    end
    legend(ax, 'Location', 'best');

    exportgraphics(fig, outPath, 'Resolution', 180, 'BackgroundColor', 'white');
    figPath = fullfile(resultsDir, sprintf('%s_%dx%d_freq_iterations.fig', nameSafe, nelx, nely));
    savefig(fig, figPath);
    close(fig);
    fprintf('Saved frequency iteration plot: %s\n', outPath);
    fprintf('Saved frequency iteration figure: %s\n', figPath);
end

function [supportCode, leftType, rightType] = mapSupportsToCode(supports)
    leftType = '';
    rightType = '';
    for k = 1:numel(supports)
        s = supports(k);
        if ~isfield(s, 'type') || isempty(s.type)
            continue;
        end
        t = lower(strtrim(char(s.type)));
        % Only hinge/clamp with a string left/right location map to supportCode.
        if ~any(strcmp(t, {'hinge', 'clamp'}))
            continue;
        end
        if ~isfield(s, 'location') || isempty(s.location) || ...
                (~ischar(s.location) && ~isstring(s.location))
            continue;
        end
        loc = lower(strtrim(char(s.location)));
        if contains(loc, 'left')
            leftType = t;
        elseif contains(loc, 'right')
            rightType = t;
        end
    end

    if strcmp(leftType, 'clamp') && strcmp(rightType, 'clamp')
        supportCode = 'CC';
    elseif strcmp(leftType, 'clamp') && strcmp(rightType, 'hinge')
        supportCode = 'CS';
    elseif strcmp(leftType, 'hinge') && strcmp(rightType, 'hinge')
        supportCode = 'SS';
    elseif strcmp(leftType, 'clamp') && isempty(rightType)
        supportCode = 'CF';
    elseif isempty(leftType) && isempty(rightType)
        % No hinge/clamp entries — all BCs come from extraFixedDofs.
        supportCode = 'NONE';
    else
        error('run_topopt_from_json:UnsupportedSupportCombo', ...
            ['Unsupported support combination from bc.supports. ', ...
             'Expected clamp+clamp, clamp+hinge, hinge+hinge, left clamp only, or purely new-style entries.']);
    end
end

function bcType = mapSupportCodeToYuksel(code)
    switch upper(code)
        case 'SS'
            bcType = 'simply';
        case 'CS'
            bcType = 'fixedPinned';
        case 'CF'
            bcType = 'cantilever';
        case 'NONE'
            bcType = 'none';  % no standard hinge/clamp — extraFixedDofs carries all BCs
        otherwise
            error('run_topopt_from_json:UnsupportedYukselBC', ...
                'Yuksel approach supports SS/CS/CF/NONE mappings only (not "%s").', code);
    end
end

function [ft, ftBC] = mapFilterToYuksel(filterType, filterBC)
    t = lower(strtrim(filterType));
    switch t
        case 'sensitivity'
            ft = 1;
        case 'density'
            ft = 2;
        case 'density_projection'
            ft = 3;
        otherwise
            error('run_topopt_from_json:UnsupportedYukselFilter', ...
                'Unsupported optimization.filter.type "%s" for Yuksel approach.', filterType);
    end

    bc = lower(strtrim(filterBC));
    switch bc
        case {'symmetric', 'reflect'}
            ftBC = 'N';
        otherwise
            ftBC = '0';
    end
end

function ft = mapFilterToOurApproach(filterType)
    t = lower(strtrim(filterType));
    switch t
        case 'sensitivity'
            ft = 0;
        case {'density', 'density_projection'}
            ft = 1;
        otherwise
            error('run_topopt_from_json:UnsupportedOurFilter', ...
                'Unsupported optimization.filter.type "%s" for ourApproach.', filterType);
    end
end

function tipMassFrac = parseTipMassFraction(cfg, volfrac, L, H, rho0)
    tipMassFrac = [];
    if ~hasFieldPath(cfg, {'bc','concentrated_masses'})
        return;
    end
    masses = getFieldPath(cfg, {'bc','concentrated_masses'});
    if isempty(masses)
        return;
    end
    if isstruct(masses)
        masses = masses(:);
    else
        error('run_topopt_from_json:InvalidMassList', 'bc.concentrated_masses must be an array of objects.');
    end

    permittedMass = volfrac * L * H * rho0;
    for i = 1:numel(masses)
        m = masses(i);
        if ~isfield(m, 'enabled') || ~parseBool(m.enabled, 'bc.concentrated_masses[].enabled')
            continue;
        end
        if ~isfield(m, 'value_type') || ~isfield(m, 'value')
            continue;
        end
        vt = lower(strtrim(char(m.value_type)));
        v = m.value;
        if ~isnumeric(v) || ~isscalar(v) || ~(v > 0)
            continue;
        end
        switch vt
            case 'fraction_of_permitted_mass'
                tipMassFrac = v;
                return;
            case 'absolute'
                if permittedMass > 0
                    tipMassFrac = v / permittedMass;
                    return;
                end
        end
    end
end

function saveTopologySnapshot(x, nelx, nely, approachName, jsonSource, omega1, visualizationQuality, omega2)
    if nargin < 5 || isempty(jsonSource)
        folder = pwd;
    else
        [folder, ~, ~] = fileparts(jsonSource);
        if isempty(folder)
            folder = pwd;
        end
    end
    if nargin < 4 || isempty(approachName)
        approachName = 'topopt';
    end
    if nargin < 6 || isempty(omega1) || ~isfinite(omega1)
        omega1 = NaN;
    end
    if nargin < 7
        visualizationQuality = 'regular';
    end
    if nargin < 8 || isempty(omega2) || ~isscalar(omega2) || ~isfinite(omega2)
        omega2 = NaN;
    end

    nameRaw  = char(string(approachName));
    nameSafe = regexprep(nameRaw, '[^\w\-]', '_');
    baseName = fullfile(folder, sprintf('%s_%dx%d', nameSafe, nelx, nely));

    % Build a visible figure so the renderer is fully active before saving.
    fig = figure('Color', 'white', 'Visible', 'on');
    ax  = axes('Parent', fig);
    img = buildTopologyDisplayImage(x, nelx, nely, visualizationQuality, true);
    imagesc(ax, 1 - img, 'Interpolation', 'nearest');
    axis(ax, 'equal'); axis(ax, 'tight');
    set(ax, 'YDir', 'normal');
    set(ax, 'XColor', 'none', 'YColor', 'none');  % hide ticks/lines; keep title visible
    colormap(ax, gray(256));
    clim(ax, [0 1]);

    if isfinite(omega1)
        titleStr = sprintf('%s  |  %dx%d  |  \\omega_1 = %.2f rad/s', ...
            nameRaw, nelx, nely, omega1);
        if isfinite(omega2)
            titleStr = sprintf('%s  |  \\omega_2 = %.2f rad/s', ...
                titleStr, omega2);
        end
    else
        titleStr = sprintf('%s  |  %dx%d', nameRaw, nelx, nely);
    end
    title(ax, titleStr, 'Interpreter', 'tex', 'FontSize', 10);

    drawnow;   % flush renderer so both PNG and FIG capture the full content

    % PNG: export full figure (title included).
    exportgraphics(fig, [baseName '.png'], 'Resolution', 160, 'BackgroundColor', 'white');

    % FIG: save while the figure is visible so the file is self-contained.
    savefig(fig, [baseName '.fig']);

    close(fig);
    fprintf('Saved topology image: %s  (.png / .fig)\n', baseName);
end

function saveReferenceModeVisualizationsFromCache( ...
    jsonSource, nModes, ...
    L, H, nelx, nely, ...
    omegaVals, modeShapes, free, ndof)
% Save initial-domain eigenmodes from precomputed eigenpairs (no eigensolve).
    if nModes <= 0 || isempty(omegaVals)
        return;
    end

    [outFolder, jsonBaseName] = localResolveModeOutputTarget(jsonSource);

    nAvail = min(nModes, numel(omegaVals));
    if nAvail < nModes
        warning('run_topopt_from_json:ModeVisualizationTruncated', ...
            'Requested %d initial modes, but only %d were computed.', nModes, nAvail);
    end
    for i = 1:nAvail
        modeFull = zeros(ndof, 1);
        modeFull(free) = modeShapes(:, i);
        localSaveSingleModePlot( ...
            modeFull, L, H, nelx, nely, ...
            omegaVals(i), i, outFolder, jsonBaseName);
    end
end

function saveTopologyModeVisualizationsFromCache( ...
    jsonSource, xDens, nModes, ...
    L, H, nelx, nely, ...
    omegaVals, modeShapes, free, ndof, ...
    corrMat, useCorrelationLabels)
% Save final-topology eigenmodes using precomputed eigenpairs (no eigensolve).
% corrMat (optional, nI x nT): correlation matrix; empty [] disables labels.
% useCorrelationLabels (optional bool): embed best-match initial mode in plot title.
    if nargin < 12, corrMat = []; end
    if nargin < 13, useCorrelationLabels = false; end
    if nModes <= 0 || isempty(omegaVals)
        return;
    end

    [outFolder, jsonBaseName] = localResolveModeOutputTarget(jsonSource);

    nAvail = min(nModes, numel(omegaVals));
    if nAvail < nModes
        warning('run_topopt_from_json:TopologyModeVisualizationTruncated', ...
            'Requested %d topology modes, but only %d could be computed.', nModes, nAvail);
    end
    for i = 1:nAvail
        corrLabel = '';
        if useCorrelationLabels && ~isempty(corrMat) && i <= size(corrMat, 2)
            [bestVal, bestInitIdx] = max(corrMat(:, i));
            corrLabel = sprintf('best init. mode = %d (corr=%.3f)', bestInitIdx, bestVal);
        end
        modeFull = zeros(ndof, 1);
        modeFull(free) = modeShapes(:, i);
        localSaveTopologyModePlot( ...
            xDens, modeFull, L, H, nelx, nely, ...
            omegaVals, i, outFolder, jsonBaseName, corrLabel);
    end
end

function [omegaVals, modeShapes, free, ndof, Mff, Mfull] = localSolveTopologyModes( ...
    xDens, nModes, L, H, nelx, nely, thickness, ...
    E0, Emin, nu, rho0, rho_min, penal, ...
    supportCode, extraFixedDofs)
% Solve generalised eigenvalue problem K*phi = lambda*M*phi for optimised topology.
% Returns both Mff (restricted to free DOFs) and Mfull (all DOFs).
    nNode = (nelx + 1) * (nely + 1);
    ndof = 2 * nNode;

    [KE, ME] = localQ4ElementMatrices(L / nelx, H / nely, nu, thickness);
    [iK, jK] = localBuildQ4AssemblyIndices(nelx, nely);

    xEl = max(0, min(1, xDens(:)));  % ensure [0,1]
    Ee = Emin + (xEl .^ penal) * (E0 - Emin);
    rhoEl = rho_min + xEl * (rho0 - rho_min);

    sK = reshape(KE(:) * Ee', [], 1);
    sM = reshape(ME(:) * rhoEl', [], 1);
    K = sparse(iK, jK, sK, ndof, ndof);
    M = sparse(iK, jK, sM, ndof, ndof);
    K = (K + K') / 2;
    M = (M + M') / 2;

    fixed = localBuildFixedDofsFromSupportCode(supportCode, nelx, nely);
    if ~isempty(extraFixedDofs)
        fixed = unique([fixed(:); double(extraFixedDofs(:))]);
    end
    fixed = unique(fixed(:));
    fixed = fixed(fixed >= 1 & fixed <= ndof);

    free = setdiff((1:ndof)', fixed);
    if isempty(free)
        error('run_topopt_from_json:NoFreeDofsForTopologyModes', ...
            'No free DOFs available for topology mode visualization.');
    end

    Mfull = M;
    Mff = Mfull(free, free);
    nReq = min(max(1, round(double(nModes))), numel(free));
    if numel(free) > 1
        nReq = min(nReq, numel(free) - 1);
    end
    [omegaVals, modeShapes] = localSolveLowestModes(K(free, free), Mff, nReq);
end

function localSaveTopologyModePlot( ...
    xDens, modeFull, L, H, nelx, nely, omegaAll, modeIdx, outFolder, jsonBaseName, corrLabel)
% Plot optimized topology (light gray, undeformed) overlaid with the
% deformed eigenmode in transparent light blue — solid elements only.
% omegaAll: full vector of computed omega values; title shows the mode's omega.
% corrLabel (optional): extra string appended to title, e.g. 'best init. mode = 2 (corr=0.91)'.
    if nargin < 11, corrLabel = ''; end
    nNodes = (nelx + 1) * (nely + 1);
    if numel(modeFull) ~= 2 * nNodes
        error('run_topopt_from_json:InvalidTopologyModeVector', ...
            'Mode vector length does not match mesh DOF count.');
    end

    dx = L / nelx;
    dy = H / nely;

    % --- Topology background (undeformed, light gray via direct RGB) ---
    densEl  = reshape(xDens(:), nely, nelx);      % (nely x nelx), row1=bottom
    grayVal = 1.0 - densEl * 0.45;                % void→1.0 white, solid→0.55 gray
    rgbImg  = repmat(grayVal, [1, 1, 3]);          % (nely x nelx x 3)
    xCenters = ((0:nelx-1) + 0.5) * dx;
    yCenters = ((0:nely-1) + 0.5) * dy;

    % --- Full Q4 element connectivity (matches xDens column-major order) ---
    % Element el (1-based): ely = mod(el-1,nely), elx = floor((el-1)/nely)
    [ELX_g, ELY_g] = meshgrid(0:nelx-1, 0:nely-1);  % (nely x nelx), ely varies first
    ELX = ELX_g(:);  ELY = ELY_g(:);
    n1 = ELY + ELX    *(nely+1) + 1;   % LL
    n2 = ELY + (ELX+1)*(nely+1) + 1;   % LR
    n3 = n2 + 1;                         % UR
    n4 = n1 + 1;                         % UL
    allFaces = [n1, n2, n3, n4];         % nEl x 4

    % --- Solid element mask (threshold 0.5) ---
    rho_plot_thresh = 0.5;
    solidElem  = (xDens(:) >= rho_plot_thresh);
    solidFaces = allFaces(solidElem, :);    % nSolid x 4
    solidNodes = unique(solidFaces(:));     % node indices touching solid elements

    % --- Undeformed nodal coordinates (flat vectors) ---
    xGrid = repmat((0:nelx) * dx, nely+1, 1);
    yGrid = repmat((0:nely)' * dy, 1, nelx+1);
    X = xGrid(:);
    Y = yGrid(:);

    % --- Eigenmode: normalise using solid nodes only ---
    ux = reshape(modeFull(1:2:end), nely+1, nelx+1);
    uy = reshape(modeFull(2:2:end), nely+1, nelx+1);
    ux = ux(:);  uy = uy(:);
    umax = max(hypot(ux(solidNodes), uy(solidNodes)));
    if isfinite(umax) && umax > 0
        uxN = ux / umax;
        uyN = uy / umax;
    else
        uxN = ux;
        uyN = uy;
    end

    % --- Scale from solid-region spatial extent ---
    Lsolid = max(X(solidNodes)) - min(X(solidNodes));
    Hsolid = max(Y(solidNodes)) - min(Y(solidNodes));
    scale  = 0.05 * max(max(Lsolid, Hsolid), eps);

    % --- Deformed vertices (only solid nodes displaced) ---
    V0 = [X, Y];
    Vd = V0;
    Vd(solidNodes, 1) = X(solidNodes) + scale * uxN(solidNodes);
    Vd(solidNodes, 2) = Y(solidNodes) + scale * uyN(solidNodes);

    % --- Plot ---
    fig = figure('Color', 'white', 'Visible', 'on');
    ax  = axes('Parent', fig);
    hold(ax, 'on');

    % Topology background: direct RGB, no colormap.
    image(ax, xCenters, yCenters, rgbImg);
    set(ax, 'YDir', 'normal');

    % Deformed solid-element overlay: transparent light blue.
    patch(ax, 'Vertices', Vd, 'Faces', solidFaces, ...
        'FaceColor', [0.45, 0.70, 1.00], 'FaceAlpha', 0.28, ...
        'EdgeColor', [0.10, 0.40, 0.80], 'EdgeAlpha', 0.65, 'LineWidth', 0.5);

    axis(ax, 'equal');
    set(ax, 'YDir', 'normal', 'XTick', [], 'YTick', []);
    box(ax, 'off');

    % Axis limits: full domain + 5 % padding.
    pad = 0.05 * max(L, H);
    xlim(ax, [min([0; Vd(solidNodes,1)]) - pad,  max([L; Vd(solidNodes,1)]) + pad]);
    ylim(ax, [min([0; Vd(solidNodes,2)]) - pad,  max([H; Vd(solidNodes,2)]) + pad]);

    % --- Title: omega for the displayed mode, plus optional correlation label ---
    if modeIdx <= numel(omegaAll) && isfinite(omegaAll(modeIdx))
        omegaModeStr = sprintf('\\omega_%d = %.3f rad/s', modeIdx, omegaAll(modeIdx));
        titleStr = sprintf('%s | Topology Mode %d | %s', jsonBaseName, modeIdx, omegaModeStr);
    else
        titleStr = sprintf('%s | Topology Mode %d', jsonBaseName, modeIdx);
    end
    if ~isempty(corrLabel)
        titleStr = [titleStr ' | ' corrLabel];
    end
    title(ax, titleStr, 'Interpreter', 'tex', 'FontSize', 10);

    drawnow;
    outPath = fullfile(outFolder, ...
        sprintf('%s_topology_mode_%d.png', jsonBaseName, modeIdx));
    exportgraphics(fig, outPath, 'Resolution', 180, 'BackgroundColor', 'white');
    close(fig);
    fprintf('Saved topology mode image: %s\n', outPath);
end

function [outFolder, jsonBaseName] = localResolveModeOutputTarget(jsonSource)
    if nargin < 1 || isempty(jsonSource)
        outFolder = pwd;
        jsonBaseName = 'topopt_config';
        return;
    end

    [outFolder, jsonBaseName, ~] = fileparts(char(jsonSource));
    if isempty(outFolder)
        outFolder = pwd;
    end
    if isempty(jsonBaseName)
        jsonBaseName = 'topopt_config';
    end
end

function [omegaVals, modeShapes, free, ndof] = localSolveReferenceModes( ...
    nModes, L, H, nelx, nely, thickness, ...
    E0, Emin, nu, rho0, rho_min, volfrac, penal, ...
    supportCode, extraFixedDofs)
    nNode = (nelx + 1) * (nely + 1);
    ndof = 2 * nNode;
    nEl = nelx * nely;

    [KE, ME] = localQ4ElementMatrices(L / nelx, H / nely, nu, thickness);
    [iK, jK] = localBuildQ4AssemblyIndices(nelx, nely);

    % Uniform reference field across all methods (no topology pattern).
    xRef = max(0, min(1, volfrac));
    Ee = Emin + xRef^penal * (E0 - Emin);
    rhoRef = rho_min + xRef * (rho0 - rho_min);

    sK = reshape(KE(:) * (Ee * ones(1, nEl)), [], 1);
    sM = reshape(ME(:) * (rhoRef * ones(1, nEl)), [], 1);
    K = sparse(iK, jK, sK, ndof, ndof);
    M = sparse(iK, jK, sM, ndof, ndof);
    K = (K + K') / 2;
    M = (M + M') / 2;

    fixed = localBuildFixedDofsFromSupportCode(supportCode, nelx, nely);
    if ~isempty(extraFixedDofs)
        fixed = unique([fixed(:); double(extraFixedDofs(:))]);
    end
    fixed = unique(fixed(:));
    fixed = fixed(fixed >= 1 & fixed <= ndof);

    free = setdiff((1:ndof)', fixed);
    if isempty(free)
        error('run_topopt_from_json:NoFreeDofsForModes', ...
            'No free DOFs available for initial mode visualization.');
    end

    nReq = min(max(1, round(double(nModes))), numel(free));
    if numel(free) > 1
        % eigs requires k < n, and dense full eig is too expensive for large meshes.
        nReq = min(nReq, numel(free) - 1);
    end
    [omegaVals, modeShapes] = localSolveLowestModes(K(free, free), M(free, free), nReq);
end

function [iK, jK] = localBuildQ4AssemblyIndices(nelx, nely)
    nEl = nelx * nely;
    edofMat = zeros(nEl, 8);
    for elx = 0:nelx-1
        for ely = 0:nely-1
            el = ely + elx * nely + 1;
            n1 = (nely + 1) * elx + ely;      % LL (0-based)
            n2 = (nely + 1) * (elx + 1) + ely;  % LR
            n3 = n2 + 1;                        % UR
            n4 = n1 + 1;                        % UL
            edofMat(el, :) = [2*n1+1, 2*n1+2, ...
                              2*n2+1, 2*n2+2, ...
                              2*n3+1, 2*n3+2, ...
                              2*n4+1, 2*n4+2];
        end
    end
    iK = reshape(kron(edofMat, ones(1, 8))', [], 1);
    jK = reshape(kron(edofMat, ones(8, 1))', [], 1);
end

function [KE, ME] = localQ4ElementMatrices(hx, hy, nu, thickness)
    if nargin < 4 || isempty(thickness)
        thickness = 1.0;
    end

    D = (1 / (1 - nu^2)) * [1, nu, 0; nu, 1, 0; 0, 0, 0.5 * (1 - nu)];
    invJ = [2 / hx, 0; 0, 2 / hy];
    detJ = 0.25 * hx * hy;
    gp = 1 / sqrt(3);
    gaussPts = [-gp, gp];

    KE = zeros(8, 8);
    for xi = gaussPts
        for eta = gaussPts
            dN_dxi = 0.25 * [-(1-eta), (1-eta), (1+eta), -(1+eta)];
            dN_deta = 0.25 * [-(1-xi), -(1+xi), (1+xi), (1-xi)];
            dN_xy = invJ * [dN_dxi; dN_deta];
            dN_dx = dN_xy(1, :);
            dN_dy = dN_xy(2, :);

            B = zeros(3, 8);
            B(1, 1:2:end) = dN_dx;
            B(2, 2:2:end) = dN_dy;
            B(3, 1:2:end) = dN_dy;
            B(3, 2:2:end) = dN_dx;
            KE = KE + (B' * D * B) * detJ;
        end
    end
    KE = thickness * KE;

    area = hx * hy;
    Ms = (area / 36) * [4, 2, 1, 2; ...
                        2, 4, 2, 1; ...
                        1, 2, 4, 2; ...
                        2, 1, 2, 4];
    ME = thickness * kron(Ms, eye(2));
end

function fixed = localBuildFixedDofsFromSupportCode(supportCode, nelx, nely)
    jMid = round(nely / 2);  % 0-based mid-height node index
    nL = jMid;
    nR = nelx * (nely + 1) + jMid;
    leftNodes = (0:nely)';
    rightNodes = nelx * (nely + 1) + (0:nely)';

    switch upper(string(supportCode))
        case "SS"
            fixed = [2*nL+1; 2*nL+2; 2*nR+1; 2*nR+2];
        case "CS"
            fixed = [2*leftNodes+1; 2*leftNodes+2; 2*nR+1; 2*nR+2];
        case "CC"
            fixed = [2*leftNodes+1; 2*leftNodes+2; ...
                     2*rightNodes+1; 2*rightNodes+2];
        case {"CF", "CANTILEVER"}
            fixed = [2*leftNodes+1; 2*leftNodes+2];
        case "NONE"
            fixed = zeros(0, 1);
        otherwise
            error('run_topopt_from_json:UnsupportedSupportCombo', ...
                'Unsupported support code "%s" for mode visualization.', string(supportCode));
    end
    fixed = unique(double(fixed(:)));
end

function [omegaVals, modeShapes] = localSolveLowestModes(Kff, Mff, nReq)
    nFree = size(Kff, 1);
    nReq = min(nReq, nFree);
    if nReq < 1
        error('run_topopt_from_json:NoRequestedModes', ...
            'Requested mode count must be >= 1.');
    end

    if nReq >= nFree
        [modeShapes, dVec] = eig(full(Kff), full(Mff), 'vector');
    else
        eigOpts = localReferenceModeEigOptions();
        try
            [modeShapes, D] = eigs(Kff, Mff, nReq, 'smallestabs', eigOpts);
        catch
            try
                [modeShapes, D] = eigs(Kff, Mff, nReq, 'sm', eigOpts);
            catch
                [modeShapes, D] = eigs(Kff, Mff, nReq, 'SM', eigOpts);
            end
        end
        dVec = diag(D);
    end

    lam = real(dVec(:));
    if isempty(lam) || ~any(isfinite(lam))
        error('run_topopt_from_json:ModeSolveFailed', ...
            'Unable to compute finite eigenvalues for mode visualization.');
    end
    [lamSorted, ord] = sort(lam, 'ascend');
    finiteMask = isfinite(lamSorted);
    ord = ord(finiteMask);
    lamSorted = lamSorted(finiteMask);
    keep = min(nReq, numel(ord));
    ord = ord(1:keep);
    lam = lamSorted(1:keep);
    modeShapes = real(modeShapes(:, ord));
    lam = max(lam, 0);
    omegaVals = sqrt(lam);

    % Keep deterministic sign orientation for plotting.
    for k = 1:size(modeShapes, 2)
        phi = modeShapes(:, k);
        [~, idx] = max(abs(phi));
        if ~isempty(idx) && phi(idx) < 0
            modeShapes(:, k) = -phi;
        end
    end
end

function opts = localReferenceModeEigOptions()
    % For matrix inputs, eigs ignores function-handle-only flags like issym/isreal.
    opts = struct('tol', 1e-8, 'maxit', 800, 'disp', 0);
end

function localSaveSingleModePlot( ...
    modeFull, L, H, nelx, nely, omegaVal, modeIdx, outFolder, jsonBaseName)
    nNodes = (nelx + 1) * (nely + 1);
    if numel(modeFull) ~= 2 * nNodes
        error('run_topopt_from_json:InvalidModeVector', ...
            'Mode vector length does not match mesh DOF count.');
    end

    xGrid = repmat((0:nelx) * (L / nelx), nely + 1, 1);
    yGrid = repmat((0:nely)' * (H / nely), 1, nelx + 1);
    ux = reshape(modeFull(1:2:end), nely + 1, nelx + 1);
    uy = reshape(modeFull(2:2:end), nely + 1, nelx + 1);

    maxDisp = max(hypot(ux(:), uy(:)));
    targetDisp = 0.075 * max(L, H);  % target: 7.5% of domain size
    if isfinite(maxDisp) && maxDisp > 0
        scale = targetDisp / maxDisp;
    else
        scale = 1.0;
    end

    xDef = xGrid + scale * ux;
    yDef = yGrid + scale * uy;

    fig = figure('Color', 'white', 'Visible', 'off');
    ax = axes('Parent', fig);
    hold(ax, 'on');

    % Undeformed domain outline in very light gray.
    plot(ax, [0, L, L, 0, 0], [0, 0, H, H, 0], '-', ...
        'Color', [0.92, 0.92, 0.92], 'LineWidth', 1.5);

    [rowIdx, colIdx] = localModePlotLineIndices(nely, nelx);
    for r = rowIdx
        plot(ax, xDef(r, :), yDef(r, :), '-', ...
            'Color', [0.05, 0.32, 0.68], 'LineWidth', 0.7);
    end
    for c = colIdx
        plot(ax, xDef(:, c), yDef(:, c), '-', ...
            'Color', [0.05, 0.32, 0.68], 'LineWidth', 0.7);
    end

    axis(ax, 'equal');
    xMin = min([0; xDef(:)]);
    xMax = max([L; xDef(:)]);
    yMin = min([0; yDef(:)]);
    yMax = max([H; yDef(:)]);
    pad = 0.03 * max([L, H, eps]);
    xlim(ax, [xMin - pad, xMax + pad]);
    ylim(ax, [yMin - pad, yMax + pad]);
    set(ax, 'YDir', 'normal', 'XTick', [], 'YTick', []);
    box(ax, 'on');

    title(ax, sprintf('%s | Mode %d | \\omega = %.3f rad/s', ...
        jsonBaseName, modeIdx, omegaVal), ...
        'Interpreter', 'tex', 'FontSize', 10);

    outPath = fullfile(outFolder, sprintf('%s_mode_%d.png', jsonBaseName, modeIdx));
    exportgraphics(fig, outPath, 'Resolution', 180, 'BackgroundColor', 'white');
    close(fig);
    fprintf('Saved mode shape image: %s\n', outPath);
end

function [rowIdx, colIdx] = localModePlotLineIndices(nely, nelx)
    maxLinesPerDirection = 120;
    rowStride = max(1, ceil((nely + 1) / maxLinesPerDirection));
    colStride = max(1, ceil((nelx + 1) / maxLinesPerDirection));
    % Keep one common stride so visible cells preserve mesh aspect ratio.
    % For square elements (e.g. 8x1 with 400x50), this avoids stretched cells.
    commonStride = max(rowStride, colStride);
    rowStride = commonStride;
    colStride = commonStride;
    rowIdx = unique([1:rowStride:(nely + 1), nely + 1]);
    colIdx = unique([1:colStride:(nelx + 1), nelx + 1]);
end

function ensureCompatHelpersOnPath()
    if exist('matlab.internal.math.checkInputName', 'file') == 2
        return;
    end
    thisDir = fileparts(mfilename('fullpath'));
    compatDir = fullfile(thisDir, 'compat');
    if exist(compatDir, 'dir') == 7
        addpath(compatDir);
    end
end

function v = reqNum(s, path, label)
    v = getFieldPath(s, path);
    if ~isnumeric(v) || ~isscalar(v) || ~isfinite(v)
        error('run_topopt_from_json:MissingOrInvalidField', 'Required numeric field "%s" is missing/invalid.', label);
    end
end

function v = reqInt(s, path, label)
    v = reqNum(s, path, label);
    if abs(v - round(v)) > 0
        error('run_topopt_from_json:InvalidIntegerField', 'Field "%s" must be an integer.', label);
    end
    v = round(v);
end

function v = reqStr(s, path, label)
    v = getFieldPath(s, path);
    if isstring(v) && isscalar(v), v = char(v); end
    if ~ischar(v) || isempty(strtrim(v))
        error('run_topopt_from_json:MissingOrInvalidField', 'Required string field "%s" is missing/invalid.', label);
    end
end

function a = reqStructArray(s, path, label)
    a = getFieldPath(s, path);
    % jsondecode returns a cell array when JSON objects have different field sets.
    % Normalize to a uniform struct array by padding missing fields with [].
    if iscell(a)
        a = normalizeCellOfStructs(a, label);
    end
    if ~isstruct(a)
        error('run_topopt_from_json:MissingOrInvalidField', 'Required object-array field "%s" is missing/invalid.', label);
    end
    a = a(:);
    if isempty(a)
        error('run_topopt_from_json:MissingOrInvalidField', 'Field "%s" must contain at least one entry.', label);
    end
end

function a = normalizeCellOfStructs(c, label)
% Convert a cell array of structs (jsondecode output for mixed-field objects)
% into a uniform struct array by collecting all field names and padding
% missing fields with [].
    if isempty(c)
        error('run_topopt_from_json:MissingOrInvalidField', ...
            'Field "%s" must contain at least one entry.', label);
    end
    allFields = {};
    for k = 1:numel(c)
        if ~isstruct(c{k})
            error('run_topopt_from_json:MissingOrInvalidField', ...
                'Each entry in "%s" must be a JSON object.', label);
        end
        allFields = union(allFields, fieldnames(c{k}), 'stable');
    end
    % Build the uniform struct array one element at a time.
    a(numel(c), 1) = struct();
    for k = 1:numel(c)
        for fi = 1:numel(allFields)
            fn = allFields{fi};
            if isfield(c{k}, fn)
                a(k).(fn) = c{k}.(fn);
            else
                a(k).(fn) = [];
            end
        end
    end
end

function b = parseBool(v, label)
    if islogical(v) && isscalar(v)
        b = v;
        return;
    end
    if isnumeric(v) && isscalar(v) && isfinite(v) && (v == 0 || v == 1)
        b = logical(v);
        return;
    end
    error('run_topopt_from_json:InvalidBoolean', ...
        'Field "%s" must be a JSON boolean true/false.', label);
end

function quality = parseVisualizationQuality(v, label)
    if isstring(v) && isscalar(v)
        v = char(v);
    end
    if ischar(v)
        key = lower(strtrim(v));
        if isempty(key)
            quality = 'regular';
            return;
        end
        if any(strcmp(key, {'regular', 'smooth'}))
            quality = key;
            return;
        end
    end
    error('run_topopt_from_json:InvalidVisualizationQuality', ...
        'Field "%s" must be "regular" or "smooth".', label);
end

function out = toVec3(v)
    out = NaN(3,1);
    if isempty(v), return; end
    v = v(:);
    n = min(3, numel(v));
    out(1:n) = v(1:n);
end

function assertPositive(v, label)
    if ~(isnumeric(v) && isscalar(v) && isfinite(v) && v > 0)
        error('run_topopt_from_json:InvalidPositiveField', 'Field "%s" must be > 0.', label);
    end
end

function tf = hasFieldPath(s, path)
    tf = true;
    cur = s;
    for k = 1:numel(path)
        key = path{k};
        if ~isstruct(cur) || ~isfield(cur, key)
            tf = false;
            return;
        end
        cur = cur.(key);
    end
end

function v = getFieldPath(s, path)
    cur = s;
    for k = 1:numel(path)
        key = path{k};
        if ~isstruct(cur) || ~isfield(cur, key)
            error('run_topopt_from_json:MissingField', 'Missing required field "%s".', strjoin(path, '.'));
        end
        cur = cur.(key);
    end
    v = cur;
end

function rssKB = getCurrentRSS_KB()
%GETCURRENTRSS_KB  Resident set size of current MATLAB process in KB.
    if ismac || isunix
        pid = feature('getpid');
        [status, result] = system(sprintf('ps -o rss= -p %d', pid));
        if status == 0
            rssKB = str2double(strtrim(result));
            if isnan(rssKB), rssKB = 0; end
        else
            rssKB = 0;
        end
    else
        % Windows: use memory() if available
        try
            m = memory;
            rssKB = m.MemUsedMATLAB / 1024;  % bytes -> KB
        catch
            rssKB = 0;
        end
    end
end

function sampleRSS()
%SAMPLERSS  Update peak RSS in appdata if current RSS exceeds stored peak.
    currentRSS = getCurrentRSS_KB();
    peak = getappdata(0, 'topopt_peakRSS_KB');
    if currentRSS > peak
        setappdata(0, 'topopt_peakRSS_KB', currentRSS);
    end
end

function stopAndDeleteTimer(t)
    try
        stop(t);
    catch
    end
    try
        delete(t);
    catch
    end
end

function resetTopologyPlotSession()
% Clear persistent figure handles used by shared topology plot helper.
if exist('plotTopology', 'file') == 2
    clear('plotTopology');
end
end

function validateSupportEntries(supports)
% VALIDATESUPPORTENTRIES  Type-specific validation of bc.supports entries.
for k = 1:numel(supports)
    s = supports(k);
    prefix = sprintf('bc.supports[%d]', k);
    if ~isfield(s, 'type') || isempty(s.type)
        error('run_topopt_from_json:InvalidSupportEntry', ...
            '%s: missing required field "type".', prefix);
    end
    t = lower(strtrim(char(s.type)));
    switch t
        case {'hinge', 'clamp'}
            if ~isfield(s, 'location') || isempty(s.location)
                error('run_topopt_from_json:InvalidSupportEntry', ...
                    '%s (type=%s): missing required string field "location".', prefix, t);
            end
        case 'vertical_line'
            if ~isfield(s, 'x') || isempty(s.x) || ~isnumeric(s.x) || ~isscalar(s.x)
                error('run_topopt_from_json:InvalidSupportEntry', ...
                    '%s (type=vertical_line): "x" must be a numeric scalar.', prefix);
            end
            if ~isfield(s, 'dofs') || isempty(s.dofs)
                error('run_topopt_from_json:InvalidSupportEntry', ...
                    '%s (type=vertical_line): missing required field "dofs".', prefix);
            end
            if isfield(s, 'tol') && ~isempty(s.tol)
                if ~isnumeric(s.tol) || ~isscalar(s.tol) || s.tol <= 0
                    error('run_topopt_from_json:InvalidSupportEntry', ...
                        '%s (type=vertical_line): "tol" must be a positive numeric scalar.', prefix);
                end
            end
        case 'horizontal_line'
            if ~isfield(s, 'y') || isempty(s.y) || ~isnumeric(s.y) || ~isscalar(s.y)
                error('run_topopt_from_json:InvalidSupportEntry', ...
                    '%s (type=horizontal_line): "y" must be a numeric scalar.', prefix);
            end
            if ~isfield(s, 'dofs') || isempty(s.dofs)
                error('run_topopt_from_json:InvalidSupportEntry', ...
                    '%s (type=horizontal_line): missing required field "dofs".', prefix);
            end
            if isfield(s, 'tol') && ~isempty(s.tol)
                if ~isnumeric(s.tol) || ~isscalar(s.tol) || s.tol <= 0
                    error('run_topopt_from_json:InvalidSupportEntry', ...
                        '%s (type=horizontal_line): "tol" must be a positive numeric scalar.', prefix);
                end
            end
        case 'closest_point'
            if ~isfield(s, 'location') || isempty(s.location) || ...
                    ~isnumeric(s.location) || numel(s.location) ~= 2
                error('run_topopt_from_json:InvalidSupportEntry', ...
                    '%s (type=closest_point): "location" must be a numeric [x,y] pair.', prefix);
            end
            if ~isfield(s, 'dofs') || isempty(s.dofs)
                error('run_topopt_from_json:InvalidSupportEntry', ...
                    '%s (type=closest_point): missing required field "dofs".', prefix);
            end
        otherwise
            warning('run_topopt_from_json:UnknownSupportType', ...
                '%s: unknown support type "%s" — entry will be ignored.', prefix, t);
    end
end
end

% supportsToFixedDofs lives in tools/supportsToFixedDofs.m (public standalone file).

function rejectDeprecatedKeys(cfg)
% Throw a clear error when deprecated optimization-block visualization keys are present.
    if hasFieldPath(cfg, {'optimization'})
        optCfg = getFieldPath(cfg, {'optimization'});
        if isstruct(optCfg)
            optFields = fieldnames(optCfg);
            for i = 1:numel(optFields)
                key = optFields{i};
                if startsWith(lower(key), 'visualise_')
                    replacement = ['visualize_' key(numel('visualise_') + 1:end)];
                    error('run_topopt_from_json:DeprecatedKey', ...
                        'Deprecated key "optimization.%s" found. Use "postprocessing.%s" (use visualize_* spelling).', ...
                        key, replacement);
                end
            end
        end
    end
    deprecated = { ...
        {'optimization','visualization_quality'},    'postprocessing.visualize_quality'; ...
        {'optimization','save_snapshot_image'},      'postprocessing.save_snapshot_image'; ...
        {'optimization','save_final_image'},         'postprocessing.save_final_image'; ...
        {'optimization','save_frq_iterations'},      'postprocessing.save_frequency_iterations'; ...
        {'save_frq_iterations'},                     'postprocessing.save_frequency_iterations'; ...
    };
    for i = 1:size(deprecated, 1)
        if hasFieldPath(cfg, deprecated{i,1})
            error('run_topopt_from_json:DeprecatedKey', ...
                'Deprecated key "%s" found. Use "%s" instead.', ...
                strjoin(deprecated{i,1}, '.'), deprecated{i,2});
        end
    end
end

function postproc = parsePostprocessingBlock(cfg)
% Parse the top-level "postprocessing" block and return a postproc struct.
    postproc = struct();

    if hasFieldPath(cfg, {'postprocessing','compute_modes'})
        postproc.computeModes = reqInt(cfg, {'postprocessing','compute_modes'}, 'postprocessing.compute_modes');
        if postproc.computeModes < 1
            error('run_topopt_from_json:InvalidComputeModes', 'postprocessing.compute_modes must be >= 1.');
        end
    else
        postproc.computeModes = 3;
    end

    if hasFieldPath(cfg, {'postprocessing','compute_modes_initial'})
        v = reqInt(cfg, {'postprocessing','compute_modes_initial'}, 'postprocessing.compute_modes_initial');
        if v < 0
            error('run_topopt_from_json:InvalidComputeModesInitial', ...
                'postprocessing.compute_modes_initial must be >= 0.');
        end
        postproc.computeModesInitial = v;
    else
        postproc.computeModesInitial = 0;  % derived by computeRequiredInitialModes()
    end

    if hasFieldPath(cfg, {'postprocessing','visualize_live'})
        postproc.visualizeLive = parseBool(getFieldPath(cfg, {'postprocessing','visualize_live'}), ...
            'postprocessing.visualize_live');
    else
        postproc.visualizeLive = [];  % empty: solver uses its own default
    end

    if hasFieldPath(cfg, {'postprocessing','visualize_quality'})
        postproc.visualizeQuality = parseVisualizationQuality( ...
            getFieldPath(cfg, {'postprocessing','visualize_quality'}), ...
            'postprocessing.visualize_quality');
    else
        postproc.visualizeQuality = 'regular';
    end

    if hasFieldPath(cfg, {'postprocessing','visualize_modes'})
        vm = getFieldPath(cfg, {'postprocessing','visualize_modes'});
        if ~isstruct(vm) || ~isfield(vm,'enabled') || ~isfield(vm,'count')
            error('run_topopt_from_json:InvalidVisualizeModes', ...
                'postprocessing.visualize_modes must be {"enabled": bool, "count": int}.');
        end
        postproc.visualizeModes.enabled = parseBool(vm.enabled, 'postprocessing.visualize_modes.enabled');
        postproc.visualizeModes.count   = max(0, round(double(vm.count)));
    else
        postproc.visualizeModes = struct('enabled', false, 'count', 0);
    end

    if hasFieldPath(cfg, {'postprocessing','visualize_topology_modes'})
        vtm = getFieldPath(cfg, {'postprocessing','visualize_topology_modes'});
        if ~isstruct(vtm) || ~isfield(vtm,'enabled') || ~isfield(vtm,'count')
            error('run_topopt_from_json:InvalidVisualizeTopologyModes', ...
                'postprocessing.visualize_topology_modes must be {"enabled": bool, "count": int}.');
        end
        postproc.visualizeTopologyModes.enabled = parseBool(vtm.enabled, 'postprocessing.visualize_topology_modes.enabled');
        postproc.visualizeTopologyModes.count   = max(0, round(double(vtm.count)));
    else
        postproc.visualizeTopologyModes = struct('enabled', false, 'count', 0);
    end

    if hasFieldPath(cfg, {'postprocessing','write_correlation_table'})
        postproc.writeCorrelationTable = parseBool( ...
            getFieldPath(cfg, {'postprocessing','write_correlation_table'}), ...
            'postprocessing.write_correlation_table');
    else
        postproc.writeCorrelationTable = false;
    end

    if hasFieldPath(cfg, {'postprocessing','save_final_image'})
        postproc.saveFinalImage = parseBool( ...
            getFieldPath(cfg, {'postprocessing','save_final_image'}), ...
            'postprocessing.save_final_image');
    else
        postproc.saveFinalImage = false;
    end

    if hasFieldPath(cfg, {'postprocessing','save_frequency_iterations'})
        postproc.saveFrequencyIterations = parseBool( ...
            getFieldPath(cfg, {'postprocessing','save_frequency_iterations'}), ...
            'postprocessing.save_frequency_iterations');
    else
        postproc.saveFrequencyIterations = false;
    end

    % ---- Correlation block ----
    % Activated either by write_correlation_table:true (legacy shorthand) or by an
    % explicit "correlation": { "enabled": true, ... } block.
    corrDef = struct( ...
        'enabled',                  false, ...
        'initialCount',             0, ...
        'topologyCount',            0, ...
        'metric',                   'mass_inner_product', ...
        'writeCsv',                 true, ...
        'csvFilename',              '', ...
        'writeBestMatches',         true, ...
        'bestK',                    3, ...
        'useForTopologyModeLabels', false, ...
        'writeHeatmap',             false);

    if hasFieldPath(cfg, {'postprocessing','correlation'})
        corrCfg = getFieldPath(cfg, {'postprocessing','correlation'});
        if ~isstruct(corrCfg)
            error('run_topopt_from_json:InvalidCorrelation', ...
                'postprocessing.correlation must be a JSON object.');
        end
        if ~isfield(corrCfg, 'enabled')
            error('run_topopt_from_json:InvalidCorrelation', ...
                'postprocessing.correlation.enabled is required.');
        end
        corrDef.enabled = parseBool(corrCfg.enabled, 'postprocessing.correlation.enabled');
        if isfield(corrCfg, 'initial_count')
            corrDef.initialCount = max(0, round(double(corrCfg.initial_count)));
        end
        if isfield(corrCfg, 'topology_count')
            corrDef.topologyCount = max(0, round(double(corrCfg.topology_count)));
        end
        if isfield(corrCfg, 'metric')
            corrDef.metric = lower(strtrim(char(string(corrCfg.metric))));
        end
        if isfield(corrCfg, 'write_csv')
            corrDef.writeCsv = parseBool(corrCfg.write_csv, 'postprocessing.correlation.write_csv');
        end
        if isfield(corrCfg, 'csv_filename') && ~isempty(corrCfg.csv_filename)
            corrDef.csvFilename = strtrim(char(string(corrCfg.csv_filename)));
        end
        if isfield(corrCfg, 'write_best_matches')
            corrDef.writeBestMatches = parseBool(corrCfg.write_best_matches, ...
                'postprocessing.correlation.write_best_matches');
        end
        if isfield(corrCfg, 'best_k')
            corrDef.bestK = max(1, round(double(corrCfg.best_k)));
        end
        if isfield(corrCfg, 'use_for_topology_mode_labels')
            corrDef.useForTopologyModeLabels = parseBool(corrCfg.use_for_topology_mode_labels, ...
                'postprocessing.correlation.use_for_topology_mode_labels');
        end
        if isfield(corrCfg, 'write_heatmap')
            corrDef.writeHeatmap = parseBool(corrCfg.write_heatmap, ...
                'postprocessing.correlation.write_heatmap');
        end
    elseif postproc.writeCorrelationTable
        % write_correlation_table: true is a legacy shorthand for the correlation block.
        corrDef.enabled          = true;
        corrDef.writeCsv         = true;
        corrDef.writeBestMatches = true;
    end
    postproc.correlation = corrDef;
end

function n = computeRequiredTopologyModes(postproc)
% Compute the number of topology-domain modes needed for all postprocessing consumers.
% Enforces n >= 2 when save_final_image is requested (title shows omega_1 and omega_2).
% Also accounts for correlation.topology_count when correlation is enabled.
    n = postproc.computeModes;
    if postproc.visualizeTopologyModes.enabled
        n = max(n, postproc.visualizeTopologyModes.count);
    end
    if postproc.saveFinalImage
        n = max(n, 2);  % topology snapshot title always shows omega_1 and omega_2
    end
    if postproc.correlation.enabled
        cTopo = postproc.correlation.topologyCount;
        if cTopo == 0, cTopo = postproc.computeModes; end
        n = max(n, cTopo);
    end
    if n > postproc.computeModes
        fprintf('[Postprocessing] Topology compute_modes: %d -> %d to satisfy requested outputs.\n', ...
            postproc.computeModes, n);
    end
end

function n = computeRequiredInitialModes(postproc)
% Compute the number of initial-domain (reference) modes needed for all consumers.
% Accounts for visualize_modes.count and correlation.initial_count.
    n = postproc.computeModesInitial;
    if postproc.visualizeModes.enabled
        n = max(n, postproc.visualizeModes.count);
    end
    if postproc.correlation.enabled
        cInit = postproc.correlation.initialCount;
        if cInit == 0
            cInit = postproc.computeModesInitial;
            if cInit == 0, cInit = postproc.computeModes; end
        end
        n = max(n, cInit);
    end
    % Legacy flag without new correlation block: fall back to topology count.
    if postproc.writeCorrelationTable && ~postproc.correlation.enabled && n == 0
        n = postproc.computeModes;
        fprintf('[Postprocessing] Initial domain modes set to %d for correlation table.\n', n);
    end
end

% =========================================================================
%  Correlation utilities
% =========================================================================

function C = computeCorrelationMatrixFullDOF(PhiFull, PsiFull, Mfull, metric)
%COMPUTECORRELATIONMATRIXFULLDOF  Full-DOF mass-inner-product correlation.
%
%   PhiFull : (ndof x nI) initial-domain eigenvectors embedded in full DOFs
%   PsiFull : (ndof x nT) topology-domain eigenvectors embedded in full DOFs
%   Mfull   : (ndof x ndof) topology mass matrix on full DOFs
%   metric  : "mass_inner_product" or "mac"
    PhiFull = double(PhiFull);
    PsiFull = double(PsiFull);

    MPhi = Mfull * PhiFull;
    MPsi = Mfull * PsiFull;
    crossVals = PhiFull' * MPsi;

    normPhi = sqrt(max(0, sum(PhiFull .* MPhi, 1)));
    normPsi = sqrt(max(0, sum(PsiFull .* MPsi, 1)));
    denom = normPhi' * normPsi;

    C0 = abs(crossVals) ./ max(denom, eps);
    C0 = min(max(C0, 0), 1);
    C0(~isfinite(C0)) = 0;

    metricKey = lower(strtrim(char(string(metric))));
    if strcmp(metricKey, 'mass_inner_product')
        C = C0;
    elseif strcmp(metricKey, 'mac')
        C = C0 .^ 2;
    else
        error('run_topopt_from_json:InvalidCorrelationMetric', ...
            'Unsupported postprocessing.correlation.metric "%s". Supported values: "mass_inner_product", "mac".', ...
            metricKey);
    end
end

function logCorrelationBestMatches(C, omegaInit, omegaTopo, bestK)
%LOGCORRELATIONBESTMATCHES  Print best-match report to the command window.
    [nI, nT] = size(C);
    bestK = min(bestK, nT);
    fprintf('[Correlation] Best matches per initial mode (top %d of %d topology modes):\n', bestK, nT);
    for i = 1:nI
        row = C(i, :);
        [sortedVals, sortedIdx] = sort(row, 'descend');
        parts = cell(1, bestK);
        for k = 1:bestK
            j = sortedIdx(k);
            if j <= numel(omegaTopo)
                parts{k} = sprintf('topo_%d/%.1f rad\\s (corr=%.3f)', j, omegaTopo(j), sortedVals(k));
            else
                parts{k} = sprintf('topo_%d (corr=%.3f)', j, sortedVals(k));
            end
        end
        fprintf('[Correlation] init_mode_%d (omega=%.3f rad/s): %s\n', ...
            i, omegaInit(i), strjoin(parts, ', '));
    end
    % Compact one-liner: best single match per initial mode.
    parts = cell(1, nI);
    for i = 1:nI
        [bestVal, bestJ] = max(C(i, :));
        parts{i} = sprintf('init_%d->topo_%d(%.3f)', i, bestJ, bestVal);
    end
    fprintf('[Correlation] Compact: %s\n', strjoin(parts, '  '));
end

function writeCorrelationCSV(C, omegaInit, omegaTopo, csvPath)
%WRITECORRELATIONCSV  Write correlation matrix to a CSV file.
%   Header row: column per topology mode with its frequency label.
%   First column: initial mode row labels with frequencies.
    [nI, nT] = size(C);
    fid = fopen(csvPath, 'w');
    if fid < 0
        warning('run_topopt_from_json:CorrelationCSVWriteFailed', ...
            'Cannot open for writing: %s', csvPath);
        return;
    end
    % Header: first cell blank, then topology mode + frequency labels.
    fprintf(fid, 'initial_mode');
    for j = 1:nT
        if j <= numel(omegaTopo)
            fprintf(fid, ',topology_mode_%d(%.4f_rad_per_s)', j, omegaTopo(j));
        else
            fprintf(fid, ',topology_mode_%d', j);
        end
    end
    fprintf(fid, '\n');
    % Data rows: initial mode + frequency label, then correlation values.
    for i = 1:nI
        if i <= numel(omegaInit)
            fprintf(fid, 'initial_mode_%d(%.4f_rad_per_s)', i, omegaInit(i));
        else
            fprintf(fid, 'initial_mode_%d', i);
        end
        for j = 1:nT
            fprintf(fid, ',%.6f', C(i, j));
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
    fprintf('[Postprocessing] Saved correlation CSV: %s\n', csvPath);
end

function writeCorrelationHeatmap(C, omegaInit, omegaTopo, outPath)
%WRITECORRELATIONHEATMAP  Save a correlation matrix heatmap as PNG (no extra toolboxes).
    [nI, nT] = size(C);
    fig = figure('Color', 'white', 'Visible', 'off');
    ax  = axes('Parent', fig);
    imagesc(ax, C);
    colormap(ax, parula(256));
    clim(ax, [0, 1]);
    colorbar(ax);

    % Axis labels: show mode index and frequency where available.
    xLabels = cell(1, nT);
    for j = 1:nT
        if j <= numel(omegaTopo)
            xLabels{j} = sprintf('%d\n(%.1f)', j, omegaTopo(j));
        else
            xLabels{j} = num2str(j);
        end
    end
    yLabels = cell(1, nI);
    for i = 1:nI
        if i <= numel(omegaInit)
            yLabels{i} = sprintf('%d (%.1f)', i, omegaInit(i));
        else
            yLabels{i} = num2str(i);
        end
    end

    set(ax, 'XTick', 1:nT, 'XTickLabel', xLabels, ...
            'YTick', 1:nI, 'YTickLabel', yLabels);
    set(ax, 'YDir', 'normal');
    xlabel(ax, 'Topology mode  (index / omega [rad/s])');
    ylabel(ax, 'Initial mode  (index / omega [rad/s])');
    title(ax, sprintf('Mode correlation matrix (%d initial × %d topology)', nI, nT), ...
        'Interpreter', 'none');
    axis(ax, 'tight');

    % Annotate cells with values when the matrix is small enough.
    if nI * nT <= 100
        for i = 1:nI
            for j = 1:nT
                text(ax, j, i, sprintf('%.2f', C(i,j)), ...
                    'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', 'k');
            end
        end
    end
    drawnow;
    exportgraphics(fig, outPath, 'Resolution', 150, 'BackgroundColor', 'white');
    close(fig);
    fprintf('[Postprocessing] Saved correlation heatmap: %s\n', outPath);
end
