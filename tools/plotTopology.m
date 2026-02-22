function plotTopology(xPhys, nelx, nely, titleStr, doDrawNow, visualizationQuality, isFinalPlot)
%PLOTTOPOLOGY Plot/update topology in a single reusable figure.
%
%   plotTopology(xPhys, nelx, nely, titleStr, doDrawNow, visualizationQuality, isFinalPlot)
%
% visualizationQuality: 'regular' | 'smooth' (default: 'regular')
% isFinalPlot          : when false (default), always uses regular display

persistent figHandle axHandle imgHandle

if nargin < 5 || isempty(doDrawNow)
    doDrawNow = false;
end
if nargin < 6 || isempty(visualizationQuality)
    visualizationQuality = 'regular';
end
if nargin < 7 || isempty(isFinalPlot)
    isFinalPlot = false;
end

img = buildTopologyDisplayImage(xPhys, nelx, nely, visualizationQuality, isFinalPlot);

if isempty(figHandle) || ~isgraphics(figHandle, 'figure')
    figHandle = figure('Name', 'Topology', 'NumberTitle', 'off');
    axHandle = axes('Parent', figHandle);
    axes(axHandle); %#ok<LAXES>
    imgHandle = imagesc(1-img, 'Interpolation', 'nearest');
else
    if isempty(axHandle) || ~isgraphics(axHandle, 'axes')
        axHandle = axes('Parent', figHandle);
        imgHandle = [];
    end
    axes(axHandle); %#ok<LAXES>
    if isempty(imgHandle) || ~isgraphics(imgHandle, 'image')
        imgHandle = imagesc(1-img, 'Interpolation', 'nearest');
    else
        set(imgHandle, 'CData', 1-img);
    end
end

axis equal off
set(axHandle, 'YDir', 'normal');
colormap(gray(256));
caxis([0 1]);
title(axHandle, titleStr, 'Interpreter', 'none');

if doDrawNow
    drawnow;
end
end
