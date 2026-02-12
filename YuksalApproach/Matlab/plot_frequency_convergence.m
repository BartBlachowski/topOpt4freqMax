function plot_frequency_convergence(info, caseLabel)
%PLOT_FREQUENCY_CONVERGENCE Plot Figure-6-style omega histories.
%
% Inputs
%   info      : struct returned by top99neo_inertial_freq with mode histories enabled
%   caseLabel : text label for titles

if nargin < 2 || isempty(caseLabel)
    caseLabel = 'Yuksel benchmark';
end

if ~isfield(info, 'stage1') || ~isfield(info, 'stage2') || ...
   ~isfield(info.stage1, 'omegaHist') || ~isfield(info.stage2, 'omegaHist') || ...
   isempty(info.stage1.omegaHist) || isempty(info.stage2.omegaHist)
    warning('No omega history found. Call top99neo_inertial_freq with nHistModes >= 3.');
    return;
end

om = [info.stage1.omegaHist; info.stage2.omegaHist];
nIt = size(om,1);
it = 1:nIt;
iSplit = size(info.stage1.omegaHist,1) + 0.5;

figure('Name',sprintf('%s - Frequency Convergence', caseLabel),'Color','w');
hold on;
h = [];
h(end+1) = plot(it, om(:,1), 'b-',  'LineWidth', 2.0);
if size(om,2) >= 2
    h(end+1) = plot(it, om(:,2), 'r:',  'LineWidth', 2.0);
end
if size(om,2) >= 3
    h(end+1) = plot(it, om(:,3), 'k--', 'LineWidth', 2.0);
end
grid on;
xlabel('iteration');
ylabel('natural frequency (rad/s)');

legendEntries = {'\omega_1', '\omega_2', '\omega_3'};
legendEntries = legendEntries(1:min(size(om,2),3));
legend(h, legendEntries, 'Interpreter', 'tex', 'Location', 'northeast');
title(sprintf('%s: Convergence history', caseLabel), 'Interpreter', 'none');

yBase = om(:,1);
yBase = yBase(isfinite(yBase));
if isempty(yBase)
    yPos = 0;
else
    yPos = min(yBase) + 0.05 * max(1, max(yBase)-min(yBase));
end
yl = ylim;
plot([iSplit iSplit], yl, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.0, 'HandleVisibility', 'off');
text(max(2,round(0.15*nIt)), yPos, 'Stage 1', 'Color', [0.2 0.2 0.2], 'FontWeight', 'bold');
text(min(nIt-5,round(0.65*nIt)), yPos, 'Stage 2', 'Color', [0.2 0.2 0.2], 'FontWeight', 'bold');

end
