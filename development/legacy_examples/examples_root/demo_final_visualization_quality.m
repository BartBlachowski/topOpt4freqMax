% Demo for final-only visualization quality (regular vs smooth).
% This script does not run optimization; it only exercises topology plotting.

clear; clc; close all;

repoRoot = fileparts(fileparts(mfilename('fullpath')));
toolsDir = fullfile(repoRoot, 'tools');
if exist(toolsDir, 'dir') == 7
    addpath(toolsDir);
end

nelx = 80;
nely = 24;

% Synthetic topology with gray transitions and sharp regions.
[xGrid, yGrid] = meshgrid(linspace(0, 1, nelx), linspace(0, 1, nely));
xBase = 0.45 + 0.20 * sin(4 * pi * xGrid) .* cos(2 * pi * yGrid);
xRing = (((xGrid - 0.30).^2 + (yGrid - 0.60).^2) < 0.06^2);
xBar = (abs(yGrid - 0.35) < 0.05) & (xGrid > 0.45) & (xGrid < 0.85);
X = xBase + 0.35 * double(xRing) + 0.25 * double(xBar);
X = max(0, min(1, X));
xPhys = X(:);

% Sanity check: live/iteration rendering must stay regular.
imgLive = buildTopologyDisplayImage(xPhys, nelx, nely, 'smooth', false);
imgExpected = reshape(xPhys, nely, nelx);
if ~isequal(size(imgLive), [nely, nelx]) || norm(imgLive(:) - imgExpected(:), inf) > 1e-12
    error('Live plotting sanity check failed: non-final rendering changed.');
end

% Sanity check: final smooth mode must differ from final regular mode.
imgRegular = buildTopologyDisplayImage(xPhys, nelx, nely, 'regular', true);
imgSmooth = buildTopologyDisplayImage(xPhys, nelx, nely, 'smooth', true);
if isequal(size(imgSmooth), size(imgRegular))
    error('Final smooth sanity check failed: smooth image was not upsampled.');
end

% Required demo calls: final plotting function in regular and smooth modes.
clear plotTopology;
plotTopology(xPhys, nelx, nely, 'FINAL REGULAR', true, 'regular', true);
set(gcf, 'Position', [80, 420, 820, 260]);

clear plotTopology;
plotTopology(xPhys, nelx, nely, 'FINAL SMOOTH', true, 'smooth', true);
set(gcf, 'Position', [80, 100, 820, 260]);

fprintf('Demo complete: FINAL REGULAR and FINAL SMOOTH figures generated.\n');
fprintf('Sanity checks passed: live plotting remains regular, final smooth is enhanced.\n');
