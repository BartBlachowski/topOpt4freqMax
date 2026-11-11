function out = rebuild_summaries(which, outRoot)
%REBUILD_SUMMARIES  Regenerate <case>/summary.csv from the saved per-alpha runs.
%
%   rebuild_summaries('both')            % or 'clamped' / 'building'
%   T = rebuild_summaries('clamped', outRoot)
%
%   Reads outRoot/<case>/alpha_*/result.mat and <case>_correlation.csv written
%   by run_alpha_sweep; no optimization is re-run.  Columns:
%     omega1, omega2          lowest two STRUCTURAL topology modes, taken from
%                             the correlation CSV header topology_mode_1/2
%                             (these are the frequencies to report)
%     omega1_raw, omega2_raw  the runner's returned omega(1:2): the raw final
%                             spectrum, which on designs with void-localized
%                             modes among the lowest three is NOT structural
%     iterations, stop_reason
%     phiK_best_mode/MAC/omega  best-MAC topology mode for solid-domain mode K
%   Rows are sorted by decreasing alpha.  With one case, returns its table;
%   with 'both', a struct with one table per case.

if nargin < 1 || isempty(which), which = 'both'; end
if nargin < 2 || isempty(outRoot)
    outRoot = fullfile(fileparts(mfilename('fullpath')), 'results');
end
switch lower(which)
    case {'clamped', 'building'}, cases = {lower(which)};
    case 'both', cases = {'clamped', 'building'};
    otherwise, error('rebuild_summaries:BadCase', 'Use clamped, building or both.');
end

out = struct();
for ic = 1:numel(cases)
    name = cases{ic};
    d = dir(fullfile(outRoot, name, 'alpha_*'));
    d = d([d.isdir]);
    rows = {};
    for k = 1:numel(d)
        runDir = fullfile(outRoot, name, d(k).name);
        matPath = fullfile(runDir, 'result.mat');
        if exist(matPath, 'file') ~= 2
            warning('rebuild_summaries:NoResult', 'Skipping %s (no result.mat).', runDir);
            continue;
        end
        R = load(matPath, 'a', 'omega', 'nIter', 'telemetry');
        stopReason = 'N/A';
        if isfield(R, 'telemetry') && isfield(R.telemetry, 'stopping') ...
                && isfield(R.telemetry.stopping, 'stop_reason')
            stopReason = char(string(R.telemetry.stopping.stop_reason));
        end
        [wTopo, best] = readCorrelation(fullfile(runDir, [name '_correlation.csv']));
        rows(end+1, :) = {R.a, wTopo(1), wTopo(2), R.omega(1), R.omega(2), R.nIter, stopReason, ...
            best(1).mode, best(1).mac, best(1).omega, ...
            best(2).mode, best(2).mac, best(2).omega}; %#ok<AGROW>
    end
    T = cell2table(rows, 'VariableNames', {'alpha', 'omega1', 'omega2', ...
        'omega1_raw', 'omega2_raw', 'iterations', 'stop_reason', ...
        'phi1_best_mode', 'phi1_MAC', 'phi1_omega', 'phi2_best_mode', 'phi2_MAC', 'phi2_omega'});
    T = sortrows(T, 'alpha', 'descend');
    writetable(T, fullfile(outRoot, name, 'summary.csv'));
    disp(T);
    out.(name) = T;
end
if isscalar(cases), out = out.(cases{1}); end
end

function [w, best] = readCorrelation(csvPath)
%READCORRELATION  Topology-mode frequencies (header) and best MAC per solid mode.
w = NaN(1, 2);
best = repmat(struct('mode', NaN, 'mac', NaN, 'omega', NaN), 2, 1);
if exist(csvPath, 'file') ~= 2
    warning('rebuild_summaries:NoCsv', 'Correlation CSV not found: %s', csvPath);
    return;
end
lines = splitlines(strtrim(fileread(csvPath)));
hdr = strsplit(lines{1}, ',');
w = nan(1, numel(hdr) - 2);
for k = 3:numel(hdr)
    tok = regexp(hdr{k}, '\(([\d.eE+-]+)_rad', 'tokens', 'once');
    if ~isempty(tok), w(k-2) = str2double(tok{1}); end
end
for i = 1:2
    v = str2double(strsplit(lines{i+1}, ','));
    macs = v(3:end);
    [m, j] = max(macs);
    best(i).mode = j; best(i).mac = m; best(i).omega = w(j);
end
end
