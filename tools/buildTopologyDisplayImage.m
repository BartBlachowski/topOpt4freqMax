function img = buildTopologyDisplayImage(xPhys, nelx, nely, visualizationQuality, isFinalPlot)
%BUILDTOPOLOGYDISPLAYIMAGE Build display-only topology image.
%
% Regular mode returns the native element grid.
% Smooth mode applies projection + upsampling for final plots only.

if nargin < 4 || isempty(visualizationQuality)
    visualizationQuality = 'regular';
end
if nargin < 5 || isempty(isFinalPlot)
    isFinalPlot = false;
end

quality = localParseVisualizationQuality(visualizationQuality);
img = reshape(xPhys, nely, nelx);
if ~(isFinalPlot && strcmp(quality, 'smooth'))
    return;
end

% Display-only enhancement (does not modify xPhys used by the solver).
beta = 12;
eta = 0.5;
upscale = 8;
useBinaryForDisplay = true;

xProj = 1 ./ (1 + exp(-beta * (img - eta)));
if useBinaryForDisplay
    xDisp = double(xProj >= 0.5);
else
    xDisp = xProj;
end

if exist('imresize', 'file') == 2
    img = imresize(xDisp, upscale, 'bicubic');
else
    [xGrid, yGrid] = meshgrid(1:nelx, 1:nely);
    [xq, yq] = meshgrid(linspace(1, nelx, nelx * upscale), ...
                        linspace(1, nely, nely * upscale));
    img = interp2(xGrid, yGrid, xDisp, xq, yq, 'cubic');
end
img = max(0, min(1, img));
end

function quality = localParseVisualizationQuality(value)
if isstring(value) && isscalar(value)
    value = char(value);
end
if ischar(value)
    key = lower(strtrim(value));
    if isempty(key)
        quality = 'regular';
        return;
    end
    if any(strcmp(key, {'regular', 'smooth'}))
        quality = key;
        return;
    end
end
error('buildTopologyDisplayImage:InvalidVisualizationQuality', ...
    'visualization_quality must be "regular" or "smooth".');
end
