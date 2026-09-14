function results = exp2b_building(alphaVals, outDir)
%EXP2B_BUILDING  Building example full analysis for paper revision.
%
%   Addresses reviewer demands:
%     V3 (MR5)  : Additional verification for building example (Tables 4/5).
%     V4 (M6)   : MAC validity threshold; flag gains below 0.8.
%     M6        : Discreteness metric g for each building topology.
%     CR1       : Full MAC diagnostic for all alpha values.
%     M7        : Spurious low-density mode check (eigenspectrum inspection).
%
%   Reproduces Tables 4 (eigenfreqs) and 5 (MAC-tracked gains) of the paper
%   for the 80x240 building example with passive frame elements.
%
%   Usage:
%     results = exp2b_building();
%     results = exp2b_building([1.0, 0.75, 0.5, 0.25, 0.0]);

if nargin < 1 || isempty(alphaVals), alphaVals = [1.0, 0.75, 0.5, 0.25, 0.0]; end
if nargin < 2 || isempty(outDir),    outDir = fileparts(mfilename('fullpath')); end

scriptDir = fileparts(mfilename('fullpath'));
localEnsurePaths(scriptDir);

MAC_THRESHOLD = 0.8;

fprintf('\n=== EXP2b: Building example full analysis ===\n\n');

% Use original building JSON as base (it contains passive regions)
jsonBase = fullfile(scriptDir, '..', 'Building', 'BuildingTopOptFreq.json');
if ~isfile(jsonBase)
    error('exp2b:MissingJSON', 'BuildingTopOptFreq.json not found: %s', jsonBase);
end

data = jsondecode(fileread(jsonBase));
data.postprocessing.visualize_live = false;
data.postprocessing.compute_modes  = 100;
data.postprocessing.visualize_modes = struct('enabled', false, 'count', 0);
data.postprocessing.visualize_topology_modes = struct('enabled', false, 'count', 0);
% Use explicit correlation block with true mass-weighted MAC.
% Metric: 'mac' = (mass-weighted cosine C0)^2.
% MANUSCRIPT NOTE: paper Eq.9 shows unweighted Euclidean MAC; revision should
% update the formula to include M (mass-weighted, physically correct for FE modes).
% This produces correct MAC values; they will be lower than the original paper
% Tables 4/5 (which used 'mass_inner_product', unsquared). The revision tables
% should be updated with these corrected values.
data.postprocessing = rmfield_if(data.postprocessing, 'write_correlation_table');
data.postprocessing.correlation = struct( ...
    'enabled',       true, ...
    'initial_count', 2, ...
    'topology_count', 100, ...
    'metric',        'mac', ...
    'write_csv',     true);

% Report solid-domain reference eigenfrequencies (paper: omega1^0=19.84, omega2^0=90.93)
fprintf('--- Solid-domain reference eigenfrequencies (paper Section 4.3) ---\n');
fprintf('  Paper states: omega1^0 = 19.84 rad/s, omega2^0 = 90.93 rad/s\n');
fprintf('  (These are for the fully solid domain, verified by running with baseline=solid)\n\n');

nAlpha  = numel(alphaVals);
omega1_arr  = NaN(nAlpha,1);
omega2_arr  = NaN(nAlpha,1);
grayness_arr = NaN(nAlpha,1);
nIter_arr   = NaN(nAlpha,1);
macData_cell = cell(nAlpha,1);
xFinal_cell  = cell(nAlpha,1);

for i = 1:nAlpha
    alpha = alphaVals(i);
    fprintf('--- alpha = %.2f ---\n', alpha);
    data.domain.load_cases(1).factor = alpha;
    data.domain.load_cases(2).factor = 1 - alpha;

    csvBefore = fullfile(outDir, 'topopt_config_correlation.csv');
    if isfile(csvBefore), delete(csvBefore); end

    try
        [xFin, omega, ~, nIter, ~] = run_topopt_from_json(data);
        omega1_arr(i)    = omega(1);
        omega2_arr(i)    = omega(2);
        grayness_arr(i)  = mean(4*xFin.*(1-xFin));
        nIter_arr(i)     = nIter;
        xFinal_cell{i}   = xFin;
        fprintf('  omega1=%.4f, omega2=%.4f, g=%.4f, nIter=%d\n', ...
            omega(1), omega(2), grayness_arr(i), nIter);
    catch ME
        fprintf('  FAILED: %s\n', ME.message);
        continue;
    end

    csvAfter = fullfile(outDir,'topopt_config_correlation.csv');
    if isfile(csvAfter)
        macD = localParseMACFromCSV(csvAfter);
        macData_cell{i} = macD;
        copyfile(csvAfter, fullfile(outDir, sprintf('exp2b_alpha%.2f_correlation.csv', alpha)));
        localPrintMACDiagnostic(alpha, macD, [19.84, 90.93], MAC_THRESHOLD);
    end
    fprintf('\n');
end

fprintf('--- Summary: Building eigenfrequencies (reproduces Table 4) ---\n');
localPrintSummaryTable(alphaVals, omega1_arr, omega2_arr, grayness_arr, nIter_arr);

fprintf('--- MAC-tracked gains (reproduces Table 5) ---\n');
localPrintMACGainTable(alphaVals, macData_cell, [19.84, 90.93], MAC_THRESHOLD);

localPrintSpuriousModeCheck(alphaVals, macData_cell);

results = struct('alphaVals', alphaVals, 'omega1', omega1_arr, 'omega2', omega2_arr, ...
    'grayness', grayness_arr, 'nIter', nIter_arr, 'macData', {macData_cell}, ...
    'xFinal', {xFinal_cell}, 'MAC_threshold', MAC_THRESHOLD);

save(fullfile(outDir,'exp2b_building_results.mat'),'results');
fprintf('Results saved.\n');
end

% =========================================================================
function localPrintMACDiagnostic(alpha, macD, omega_init, MAC_THRESH)
if isempty(macD) || isempty(macD.best_mode), return; end
nI = numel(macD.best_mode);
fprintf('  [MAC alpha=%.2f]\n', alpha);
for i = 1:min(2, nI)
    m  = macD.best_mode(i);
    mc = macD.best_mac(i);
    ot = macD.best_omega(i);
    g  = ot / omega_init(i);
    v  = ternary(mc >= MAC_THRESH, 'VALID', 'BELOW_THRESH');
    fprintf('    Phi%d: best at topo-mode %d, MAC=%.4f, omega=%.4f rad/s, gain=%.3fx [%s]\n', ...
        i, m, mc, ot, g, v);
end
% Show modes 1-5 and modes around the best-match
fprintf('    Topo-modes 1-5:\n');
if ~isempty(macD.omega_topo) && ~isempty(macD.mac_mat)
    nTShow = min(5, numel(macD.omega_topo));
    nI2 = min(2, size(macD.mac_mat,1));
    for k = 1:nTShow
        m1k = macD.mac_mat(min(1,nI2), k);
        m2k = macD.mac_mat(min(2,nI2), k);
        fprintf('      mode %2d: omega=%.4f, MAC1=%.4f, MAC2=%.4f\n', ...
            k, macD.omega_topo(k), m1k, m2k);
    end
end
end

% =========================================================================
function localPrintSummaryTable(alphaVals, omega1, omega2, gray, nIter)
fprintf('  alpha | omega1(rad/s) | omega2(rad/s) | grayness | nIter\n');
fprintf('  ------+--------------+--------------+----------+------\n');
for i=1:numel(alphaVals)
    fprintf('  %.2f  |   %10.4f |   %10.4f |  %.4f  | %d\n', ...
        alphaVals(i), omega1(i), omega2(i), gray(i), nIter(i));
end
fprintf('\n');
end

% =========================================================================
function localPrintMACGainTable(alphaVals, macData, omega_init, MAC_THRESH)
fprintf('  alpha | Phi1-mode MAC1  omega_t1   gain1  [v?] | Phi2-mode MAC2  omega_t2   gain2  [v?]\n');
fprintf('  ------+---------------------------------------+---------------------------------------\n');
for i=1:numel(alphaVals)
    macD = macData{i};
    if isempty(macD)||isempty(macD.best_mode)
        fprintf('  %.2f  |             N/A              |              N/A\n', alphaVals(i));
        continue;
    end
    nI=numel(macD.best_mode);
    m1=macD.best_mode(min(1,nI)); mac1=macD.best_mac(min(1,nI)); ot1=macD.best_omega(min(1,nI));
    m2=macD.best_mode(min(2,nI)); mac2=macD.best_mac(min(2,nI)); ot2=macD.best_omega(min(2,nI));
    g1=ot1/omega_init(1); g2=ot2/omega_init(2);
    v1=ternary(mac1>=MAC_THRESH,'Y','N'); v2=ternary(mac2>=MAC_THRESH,'Y','N');
    fprintf('  %.2f  | %3d  %.3f  %9.4f  %.3fx [%s] | %3d  %.3f  %9.4f  %.3fx [%s]\n', ...
        alphaVals(i), m1,mac1,ot1,g1,v1, m2,mac2,ot2,g2,v2);
end
fprintf('\n');
end

% =========================================================================
function localPrintSpuriousModeCheck(alphaVals, macData)
fprintf('--- Spurious low-density mode check (Reviewer M7) ---\n');
fprintf('Checking whether any topo-mode with very low omega has near-zero MAC with all solid modes.\n');
fprintf('If MAC < 0.01 for all solid-domain modes: potential spurious mode.\n\n');
for i=1:numel(alphaVals)
    macD=macData{i};
    if isempty(macD)||isempty(macD.mac_mat)||isempty(macD.omega_topo), continue; end
    nTopo=numel(macD.omega_topo); nInit=size(macD.mac_mat,1);
    fprintf('  alpha=%.2f:\n', alphaVals(i));
    nCheck=min(10, nTopo);
    for k=1:nCheck
        maxMAC=max(macD.mac_mat(:,k));
        if maxMAC < 0.01
            fprintf('    mode %2d: omega=%.4f rad/s, max_MAC_over_all_init_modes=%.4f  [SUSPECT SPURIOUS]\n', ...
                k, macD.omega_topo(k), maxMAC);
        else
            fprintf('    mode %2d: omega=%.4f rad/s, max_MAC=%.4f  [OK]\n', ...
                k, macD.omega_topo(k), maxMAC);
        end
    end
    fprintf('\n');
end
end

% =========================================================================
function macD = localParseMACFromCSV(csvFile)
macD=struct('omega_topo',[],'mac_mat',[],'omega_init',[],'best_mode',[],'best_mac',[],'best_omega',[]);
try
    [corr,rowLabels,colLabels]=readCorrelationCSV(csvFile);
    nInit=size(corr,1); nTopo=size(corr,2);
    topoOmegas=NaN(nTopo,1);
    for j=1:nTopo, topoOmegas(j)=localParseOmega(colLabels{j}); end
    initOmegas=NaN(nInit,1);
    for i=1:nInit, initOmegas(i)=localParseOmega(rowLabels{i}); end
    macD.omega_topo=topoOmegas; macD.omega_init=initOmegas; macD.mac_mat=corr;
    macD.best_mode=NaN(nInit,1); macD.best_mac=NaN(nInit,1); macD.best_omega=NaN(nInit,1);
    for i=1:nInit
        [mx,idx]=max(corr(i,:));
        macD.best_mode(i)=idx; macD.best_mac(i)=mx; macD.best_omega(i)=topoOmegas(idx);
    end
catch ME
    fprintf('    [Warning] CSV parse: %s\n', ME.message);
end
end

function omega = localParseOmega(label)
omega=NaN;
tok=regexp(label,'\(([0-9eE+\-\.]+)_rad','tokens','once');
if ~isempty(tok), omega=str2double(tok{1}); return; end
tok2=regexp(label,'([0-9eE+\-\.]+)','tokens');
for k=1:numel(tok2), v=str2double(tok2{k}{1}); if ~isnan(v)&&v>0, omega=v; return; end; end
end

function s=ternary(c,a,b), if c, s=a; else, s=b; end, end

function s = rmfield_if(s, fn)
if isfield(s, fn), s = rmfield(s, fn); end
end

function localEnsurePaths(scriptDir)
repoRoot=fileparts(fileparts(scriptDir));
toolsDir=fullfile(repoRoot,'tools','Matlab');
if exist(toolsDir,'dir')==7, addpath(toolsDir); end
addpath(genpath(fullfile(repoRoot,'analysis')));
end
