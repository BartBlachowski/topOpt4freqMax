function loadCases = validateLoadCases(loadCasesRaw, basePath)
%VALIDATELOADCASES Validate and normalize domain.load_cases JSON entries.
%
%   loadCases = validateLoadCases(loadCasesRaw)
%   loadCases = validateLoadCases(loadCasesRaw, basePath)
%
% Normalized output format:
%   loadCases(i).name   : char
%   loadCases(i).factor : double scalar >= 0 (default 1.0)
%   loadCases(i).loads  : struct array with fields:
%       .type     : 'self_weight' | 'closest_node' | 'harmonic'
%       .factor   : scalar (required for self_weight, optional otherwise; default 1)
%       .location : 1x2 double (closest_node only)
%       .force    : 1x2 double (closest_node only)
%       .mode     : integer >= 1 (harmonic only)

if nargin < 2 || isempty(basePath)
    basePath = 'domain.load_cases';
end

cases = localNormalizeStructArray(loadCasesRaw, basePath);
if isempty(cases)
    error('validateLoadCases:EmptyLoadCases', ...
        '%s must contain at least one case.', basePath);
end

nCases = numel(cases);
loadCases = repmat(struct('name', '', 'factor', 1.0, 'loads', struct([])), nCases, 1);

for i = 1:nCases
    c = cases(i);
    casePath = sprintf('%s[%d]', basePath, i);

    % Optional case name.
    caseName = sprintf('case%d', i);
    if isfield(c, 'name') && ~isempty(c.name)
        if isstring(c.name) && isscalar(c.name)
            c.name = char(c.name);
        end
        if ~ischar(c.name)
            error('validateLoadCases:InvalidCaseName', ...
                '%s.name must be a string.', casePath);
        end
        caseName = char(c.name);
    end

    % Optional case factor.
    caseFactor = 1.0;
    if isfield(c, 'factor') && ~isempty(c.factor)
        if ~isnumeric(c.factor) || ~isscalar(c.factor) || ~isfinite(c.factor) || c.factor < 0
            error('validateLoadCases:InvalidCaseFactor', ...
                '%s.factor must be a numeric scalar >= 0.', casePath);
        end
        caseFactor = double(c.factor);
    end

    if ~isfield(c, 'loads') || isempty(c.loads)
        error('validateLoadCases:MissingLoads', ...
            '%s.loads must be a non-empty array.', casePath);
    end
    loadsRaw = localNormalizeStructArray(c.loads, sprintf('%s.loads', casePath));
    if isempty(loadsRaw)
        error('validateLoadCases:EmptyLoads', ...
            '%s.loads must contain at least one load.', casePath);
    end

    nLoads = numel(loadsRaw);
    loadsNorm = repmat(struct( ...
        'type', '', ...
        'factor', [], ...
        'location', [], ...
        'force', [], ...
        'mode', [], ...
        'update_after', []), nLoads, 1);

    for j = 1:nLoads
        ld = loadsRaw(j);
        loadPath = sprintf('%s.loads[%d]', casePath, j);

        if ~isfield(ld, 'type') || isempty(ld.type)
            error('validateLoadCases:MissingLoadType', ...
                '%s.type is required.', loadPath);
        end
        if isstring(ld.type) && isscalar(ld.type)
            ld.type = char(ld.type);
        end
        if ~ischar(ld.type)
            error('validateLoadCases:InvalidLoadType', ...
                '%s.type must be a string.', loadPath);
        end

        loadType = lower(strtrim(ld.type));
        loadsNorm(j).type = loadType;

        switch loadType
            case 'self_weight'
                if ~isfield(ld, 'factor') || isempty(ld.factor)
                    error('validateLoadCases:MissingSelfWeightFactor', ...
                        '%s.factor is required for self_weight.', loadPath);
                end
                if ~isnumeric(ld.factor) || ~isscalar(ld.factor) || ~isfinite(ld.factor)
                    error('validateLoadCases:InvalidSelfWeightFactor', ...
                        '%s.factor must be a numeric scalar.', loadPath);
                end
                loadsNorm(j).factor = double(ld.factor);

            case 'closest_node'
                loadFactor = 1.0;
                if isfield(ld, 'factor') && ~isempty(ld.factor)
                    if ~isnumeric(ld.factor) || ~isscalar(ld.factor) || ~isfinite(ld.factor)
                        error('validateLoadCases:InvalidClosestNodeFactor', ...
                            '%s.factor must be a numeric scalar when provided.', loadPath);
                    end
                    loadFactor = double(ld.factor);
                end
                if ~isfield(ld, 'location') || isempty(ld.location)
                    error('validateLoadCases:MissingClosestNodeLocation', ...
                        '%s.location is required for closest_node.', loadPath);
                end
                loc = double(ld.location(:));
                if numel(loc) ~= 2 || any(~isfinite(loc))
                    error('validateLoadCases:InvalidClosestNodeLocation', ...
                        '%s.location must be a numeric [x,y] pair.', loadPath);
                end

                if ~isfield(ld, 'force') || isempty(ld.force)
                    error('validateLoadCases:MissingClosestNodeForce', ...
                        '%s.force is required for closest_node.', loadPath);
                end
                frc = double(ld.force(:));
                if numel(frc) ~= 2 || any(~isfinite(frc))
                    error('validateLoadCases:InvalidClosestNodeForce', ...
                        '%s.force must be a numeric [Fx,Fy] pair.', loadPath);
                end

                loadsNorm(j).location = reshape(loc, 1, 2);
                loadsNorm(j).force = reshape(frc, 1, 2);
                loadsNorm(j).factor = loadFactor;

            case 'harmonic'
                loadFactor = 1.0;
                if isfield(ld, 'factor') && ~isempty(ld.factor)
                    if ~isnumeric(ld.factor) || ~isscalar(ld.factor) || ~isfinite(ld.factor)
                        error('validateLoadCases:InvalidHarmonicFactor', ...
                            '%s.factor must be a numeric scalar when provided.', loadPath);
                    end
                    loadFactor = double(ld.factor);
                end
                if ~isfield(ld, 'mode') || isempty(ld.mode)
                    error('validateLoadCases:MissingHarmonicMode', ...
                        '%s.mode is required for harmonic.', loadPath);
                end
                if ~isnumeric(ld.mode) || ~isscalar(ld.mode) || ~isfinite(ld.mode) || ld.mode < 1 || ld.mode ~= round(ld.mode)
                    error('validateLoadCases:InvalidHarmonicMode', ...
                        '%s.mode must be an integer >= 1.', loadPath);
                end
                loadsNorm(j).mode = round(double(ld.mode));
                % update_after: optional nonneg integer (default 1 = recompute every iteration).
                % 0 means compute once at iteration 1 and freeze.
                updateAfter = 1;
                if isfield(ld, 'update_after') && ~isempty(ld.update_after)
                    ua = ld.update_after;
                    if ~isnumeric(ua) || ~isscalar(ua) || ~isfinite(double(ua)) || ua < 0 || ua ~= round(ua)
                        error('validateLoadCases:InvalidUpdateAfter', ...
                            '%s.update_after must be a nonneg integer scalar.', loadPath);
                    end
                    updateAfter = round(double(ua));
                end
                loadsNorm(j).update_after = updateAfter;
                loadsNorm(j).factor = loadFactor;

            otherwise
                error('validateLoadCases:UnsupportedLoadType', ...
                    ['%s.type="%s" is not supported. ', ...
                     'Supported types: self_weight, closest_node, harmonic.'], ...
                    loadPath, loadType);
        end
    end

    loadCases(i).name = caseName;
    loadCases(i).factor = caseFactor;
    loadCases(i).loads = loadsNorm;
end
end

function arr = localNormalizeStructArray(raw, label)
% jsondecode can return mixed-field object arrays as cell arrays.
if iscell(raw)
    if isempty(raw)
        arr = struct([]);
        return;
    end
    allFields = {};
    for k = 1:numel(raw)
        if ~isstruct(raw{k}) || numel(raw{k}) ~= 1
            error('validateLoadCases:InvalidObjectArray', ...
                '%s[%d] must be a JSON object.', label, k);
        end
        allFields = union(allFields, fieldnames(raw{k}), 'stable');
    end
    arr(numel(raw), 1) = struct();
    for k = 1:numel(raw)
        for fi = 1:numel(allFields)
            fn = allFields{fi};
            if isfield(raw{k}, fn)
                arr(k).(fn) = raw{k}.(fn);
            else
                arr(k).(fn) = [];
            end
        end
    end
elseif isstruct(raw)
    arr = raw(:);
else
    error('validateLoadCases:InvalidObjectArray', ...
        '%s must be a JSON object array.', label);
end
end
