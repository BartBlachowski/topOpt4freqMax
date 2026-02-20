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

    approach = reqStr(cfg, {'optimisation','approach'}, 'optimisation.approach');
    volfrac = reqNum(cfg, {'optimisation','volume_fraction'}, 'optimisation.volume_fraction');
    penal = reqNum(cfg, {'optimisation','penalization'}, 'optimisation.penalization');
    move = reqNum(cfg, {'optimisation','move_limit'}, 'optimisation.move_limit');
    maxiter = reqInt(cfg, {'optimisation','max_iters'}, 'optimisation.max_iters');
    convTol = reqNum(cfg, {'optimisation','convergence_tol'}, 'optimisation.convergence_tol');

    filterType = reqStr(cfg, {'optimisation','filter','type'}, 'optimisation.filter.type');
    filterRadius = reqNum(cfg, {'optimisation','filter','radius'}, 'optimisation.filter.radius');
    radiusUnits = lower(reqStr(cfg, {'optimisation','filter','radius_units'}, 'optimisation.filter.radius_units'));
    filterBC = reqStr(cfg, {'optimisation','filter','boundary_condition'}, 'optimisation.filter.boundary_condition');

    supports = reqStructArray(cfg, {'bc','supports'}, 'bc.supports');
    [supportCode, ~, ~] = mapSupportsToCode(supports);
    tipMassFrac = parseTipMassFraction(cfg, volfrac, L, H, rho0);

    % Optional visualization flags (default = keep legacy behavior).
    hasVisualise = hasFieldPath(cfg, {'optimisation','visualise_live'});
    if hasVisualise
        visualiseLive = parseBool(getFieldPath(cfg, {'optimisation','visualise_live'}), 'optimisation.visualise_live');
    else
        visualiseLive = [];
    end
    hasSaveFinalImage = hasFieldPath(cfg, {'optimisation','save_final_image'});
    if hasSaveFinalImage
        saveFinalImage = parseBool(getFieldPath(cfg, {'optimisation','save_final_image'}), 'optimisation.save_final_image');
    else
        % Backward compatibility with older field name.
        hasSnapshot = hasFieldPath(cfg, {'optimisation','save_snapshot_image'});
        if hasSnapshot
            saveFinalImage = parseBool(getFieldPath(cfg, {'optimisation','save_snapshot_image'}), 'optimisation.save_snapshot_image');
        else
            saveFinalImage = false;
        end
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
                'optimisation.filter.radius_units must be "element" or "physical".');
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
        error('run_topopt_from_json:InvalidVolumeFraction', 'optimisation.volume_fraction must be in (0, 1].');
    end
    assertPositive(penal, 'optimisation.penalization');
    assertPositive(move, 'optimisation.move_limit');
    if move > 1
        error('run_topopt_from_json:InvalidMove', 'optimisation.move_limit must be <= 1.');
    end
    assertPositive(maxiter, 'optimisation.max_iters');
    assertPositive(convTol, 'optimisation.convergence_tol');

    repoRoot = fileparts(fileparts(mfilename('fullpath')));

    x = [];
    omega = NaN(3,1);
    tIter = NaN;
    nIter = NaN;
    mem_usage = 0;
    Emin = E0 * EminRatio;

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

            optsO = struct('doDiagnostic', true, 'diagnosticOnly', false, 'diagModes', 5);
            optsO.approach_name = approach;
            if hasVisualise
                optsO.visualise_live = visualiseLive;
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
            if ~isempty(tipMassFrac)
                runCfg.tipMassFrac = tipMassFrac;
            end
            if hasVisualise
                runCfg.visualise_live = visualiseLive;
            end

            eta = 0.5;
            beta = 1.0;
            stage1MaxIter = min(maxiter, 200);
            if hasFieldPath(cfg, {'optimisation','yuksel','stage1_max_iters'})
                stage1MaxIter = reqInt(cfg, {'optimisation','yuksel','stage1_max_iters'}, ...
                    'optimisation.yuksel.stage1_max_iters');
            end
            if stage1MaxIter < 1
                error('run_topopt_from_json:InvalidYukselStage1MaxIters', ...
                    'optimisation.yuksel.stage1_max_iters must be >= 1.');
            end
            stage1MaxIter = min(stage1MaxIter, maxiter);

            nHistModes = 0;
            if hasFieldPath(cfg, {'optimisation','yuksel','mode_history_modes'})
                nHistModes = reqInt(cfg, {'optimisation','yuksel','mode_history_modes'}, ...
                    'optimisation.yuksel.mode_history_modes');
            end
            if nHistModes < 0
                error('run_topopt_from_json:InvalidYukselModeHistoryModes', ...
                    'optimisation.yuksel.mode_history_modes must be >= 0.');
            end

            finalModeCount = 3;
            if hasFieldPath(cfg, {'optimisation','yuksel','final_mode_count'})
                finalModeCount = reqInt(cfg, {'optimisation','yuksel','final_mode_count'}, ...
                    'optimisation.yuksel.final_mode_count');
            end
            if finalModeCount < 1
                error('run_topopt_from_json:InvalidYukselFinalModeCount', ...
                    'optimisation.yuksel.final_mode_count must be >= 1.');
            end
            runCfg.final_modes = finalModeCount;
            if hasFieldPath(cfg, {'optimisation','yuksel','stage1_tol'})
                runCfg.stage1_tol = reqNum(cfg, {'optimisation','yuksel','stage1_tol'}, ...
                    'optimisation.yuksel.stage1_tol');
                assertPositive(runCfg.stage1_tol, 'optimisation.yuksel.stage1_tol');
            end
            if hasFieldPath(cfg, {'optimisation','yuksel','stage2_tol'})
                runCfg.stage2_tol = reqNum(cfg, {'optimisation','yuksel','stage2_tol'}, ...
                    'optimisation.yuksel.stage2_tol');
                assertPositive(runCfg.stage2_tol, 'optimisation.yuksel.stage2_tol');
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
            if hasVisualise
                runCfg.visualise_live = visualiseLive;
            end

            [xOut, fOut, tOut, itOut] = topopt_freq(nelx, nely, volfrac, penal, rmin_phys, ft, L, H, runCfg);
            x = xOut(:);
            omega = toVec3(2*pi*fOut(:)); % ourApproach runner reports Hz.
            tIter = tOut;
            nIter = itOut;

        otherwise
            error('run_topopt_from_json:UnknownApproach', ...
                'Unknown optimisation.approach "%s". Use "Olhoff", "Yuksel", or "ourApproach".', approach);
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

    if saveFinalImage
        saveTopologySnapshot(x, nelx, nely, approach, jsonSource);
    end
end

function [supportCode, leftType, rightType] = mapSupportsToCode(supports)
    leftType = '';
    rightType = '';
    for k = 1:numel(supports)
        s = supports(k);
        if ~isfield(s, 'type') || ~isfield(s, 'location')
            error('run_topopt_from_json:InvalidSupportEntry', ...
                'Each bc.supports entry must include "type" and "location".');
        end
        t = lower(strtrim(char(s.type)));
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
    else
        error('run_topopt_from_json:UnsupportedSupportCombo', ...
            ['Unsupported support combination from bc.supports. ', ...
             'Expected clamp+clamp, clamp+hinge, hinge+hinge, or left clamp only.']);
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
        otherwise
            error('run_topopt_from_json:UnsupportedYukselBC', ...
                'Yuksel approach supports SS/CS/CF mappings only (not "%s").', code);
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
                'Unsupported optimisation.filter.type "%s" for Yuksel approach.', filterType);
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
                'Unsupported optimisation.filter.type "%s" for ourApproach.', filterType);
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

function saveTopologySnapshot(x, nelx, nely, approachName, jsonSource)
    if nargin < 5 || isempty(jsonSource)
        folder = pwd;
    else
        [folder, ~, ~] = fileparts(jsonSource);
    end
    if nargin < 4 || isempty(approachName)
        approachName = 'topopt';
    end
    nameRaw = char(string(approachName));
    nameSafe = regexprep(nameRaw, '[^\w\-]', '_');
    baseName = fullfile(folder, sprintf('%s_%dx%d', nameSafe, nelx, nely));
    fig = figure('Visible', 'off');
    imagesc(1 - reshape(x, nely, nelx));
    axis equal tight off;
    colormap(gray(256));
    exportgraphics(gca, [baseName '.png'], 'Resolution', 160);
    savefig(fig, [baseName '.fig']);
    close(fig);
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
    if ~isstruct(a)
        error('run_topopt_from_json:MissingOrInvalidField', 'Required object-array field "%s" is missing/invalid.', label);
    end
    a = a(:);
    if isempty(a)
        error('run_topopt_from_json:MissingOrInvalidField', 'Field "%s" must contain at least one entry.', label);
    end
end

function b = parseBool(v, label)
    if islogical(v) && isscalar(v)
        b = v;
        return;
    end
    if isnumeric(v) && isscalar(v)
        b = v ~= 0;
        return;
    end
    if isstring(v) && isscalar(v), v = char(v); end
    if ischar(v)
        t = lower(strtrim(v));
        if any(strcmp(t, {'true','yes','y','1','on'}))
            b = true;
            return;
        elseif any(strcmp(t, {'false','no','n','0','off'}))
            b = false;
            return;
        end
    end
    error('run_topopt_from_json:InvalidBoolean', 'Field "%s" must be boolean-like.', label);
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
