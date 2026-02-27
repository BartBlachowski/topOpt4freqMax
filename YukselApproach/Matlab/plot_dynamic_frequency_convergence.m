function plot_dynamic_frequency_convergence(omegaHist, caseLabel)
%PLOT_DYNAMIC_FREQUENCY_CONVERGENCE Plot dynamic-code omega history (Figure 6 style).
%
% Inputs
%   omegaHist : [nIter x nModes] matrix with natural circular frequencies (rad/s)
%   caseLabel : title prefix

if nargin < 2 || isempty(caseLabel)
    caseLabel = 'Yuksel dynamic benchmark';
end

if isempty(omegaHist) || ~isnumeric(omegaHist)
    warning('Empty omega history provided to plot_dynamic_frequency_convergence.');
    return;
end

nIter = size(omegaHist,1);
it = 0:(nIter-1);

figure('Name',sprintf('%s - Figure 6 history', caseLabel),'Color','w');
theme("light");
hold on;

plot(it, omegaHist(:,1), 'b-', 'LineWidth', 2.0);
if size(omegaHist,2) >= 2
    plot(it, omegaHist(:,2), 'r:', 'LineWidth', 2.0);
end
if size(omegaHist,2) >= 3
    plot(it, omegaHist(:,3), 'k--', 'LineWidth', 2.0);
end

xlabel('iteration');
ylabel('natural frequency (rad/s)');
grid on;
box on;
xlim([0 max(200,nIter-1)]);

legendEntries = {'\omega_1', '\omega_2', '\omega_3'};
legendEntries = legendEntries(1:min(3,size(omegaHist,2)));
legend(legendEntries, 'Interpreter', 'tex', 'Location', 'northeast');
title(sprintf('%s: Convergence history (dynamic code)', caseLabel), 'Interpreter', 'none');

end
