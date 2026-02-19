function plotTopology(xPhys, nelx, nely, titleStr, doDrawNow)
%PLOTTOPOLOGY Plot/update topology in a single reusable figure.
%
%   plotTopology(xPhys, nelx, nely, titleStr, doDrawNow)

persistent figHandle axHandle imgHandle

if nargin < 5 || isempty(doDrawNow)
    doDrawNow = false;
end

img = reshape(xPhys, nely, nelx);

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
colormap(gray(256));
caxis([0 1]);
title(axHandle, titleStr, 'Interpreter', 'none');

if doDrawNow
    drawnow;
end
end
