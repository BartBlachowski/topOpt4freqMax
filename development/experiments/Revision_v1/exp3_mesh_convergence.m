function results = exp3_mesh_convergence(alphaVals, outDir)
%EXP3_MESH_CONVERGENCE  Mesh convergence study for the clamped-beam example.
%
%   Addresses reviewer demands:
%     V3 (MR5) : Provide mesh convergence study (min. 2 mesh sizes).
%     C2       : Compare semi_harmonic (paper) vs harmonic Eq.7 at two meshes.
%     V4       : MAC validity threshold enforced.
%
%   Runs all ALPHAVALS at two mesh resolutions for BOTH formulations:
%     - semi_harmonic (paper implementation)
%     - harmonic Eq.7 (solid baseline, exact Eq.7)
%   at meshes: 200x25 (coarse) and 400x50 (reference).
%
%   Usage:
%     results = exp3_mesh_convergence();
%     results = exp3_mesh_convergence([1.0, 0.75, 0.5, 0.25, 0.0]);

if nargin < 1 || isempty(alphaVals), alphaVals = [1.0, 0.75, 0.5, 0.25, 0.0]; end
if nargin < 2 || isempty(outDir),    outDir = fileparts(mfilename('fullpath')); end

scriptDir = fileparts(mfilename('fullpath'));
localEnsurePaths(scriptDir);

% Metric: JSONs use metric='mac' = (mass-weighted cosine)^2, matching paper Eq.9.
% Values will be lower than original paper tables (which used mass_inner_product).
MAC_THRESHOLD = 0.8;

fprintf('\n=== EXP3: Mesh convergence study (clamped beam) ===\n\n');

meshConfigs = {
    struct('json_sh',  fullfile(scriptDir,'clamped_beam_400x50.json'), ...
           'json_eq7', fullfile(scriptDir,'clamped_beam_harmonic_eq7_400x50.json'), ...
           'label', '400x50', 'nelx', 400, 'nely', 50);
    struct('json_sh',  fullfile(scriptDir,'clamped_beam_200x25.json'), ...
           'json_eq7', fullfile(scriptDir,'clamped_beam_harmonic_eq7_200x25.json'), ...
           'label', '200x25', 'nelx', 200, 'nely', 25);
};
nMeshes = numel(meshConfigs);
nAlpha  = numel(alphaVals);

formLabels = {'semi_harmonic (paper)', 'harmonic Eq.7 (solid)'};
nForms = 2;

% omega_init_all(mesh, 2)
% res_all{mesh}{form}.omega1(alpha), .grayness, .macData
omega_init_all = NaN(nMeshes, 2);
res_all = cell(nMeshes, nForms);
for mi = 1:nMeshes
    for f = 1:nForms
        res_all{mi,f} = struct('omega1', NaN(nAlpha,1), 'omega2', NaN(nAlpha,1), ...
            'grayness', NaN(nAlpha,1), 'nIter', NaN(nAlpha,1), ...
            'macData', cell(nAlpha,1));
    end
end

for mi = 1:nMeshes
    mc = meshConfigs{mi};
    fprintf('=== Mesh: %s ===\n', mc.label);

    % Initial-domain (x=volfrac) eigenfrequencies for this mesh.
    % The semi_harmonic JSON has correlation enabled; a 1-iteration probe extracts
    % the initial-domain omegas from the CSV row labels (these match the paper's
    % reported omega_0 values, e.g. 145.44 rad/s for 400x50 clamped at VF=0.5).
    fprintf('  Probing initial-domain eigenfrequencies (1-iter run)...\n');
    om_init = localSolidEigenpairs(mc.json_sh, 2);
    omega_init_all(mi,:) = om_init(1:2)';
    fprintf('  omega1^0=%.4f, omega2^0=%.4f rad/s\n\n', om_init(1), om_init(2));

    jsonPaths = {mc.json_sh, mc.json_eq7};
    for f = 1:nForms
        jp = jsonPaths{f};
        if ~isfile(jp)
            fprintf('  [Skipped] %s: file not found\n', jp);
            continue;
        end
        fprintf('  Formulation: %s\n', formLabels{f});
        data = jsondecode(fileread(jp));
        data.postprocessing.visualize_live = false;

        for i = 1:nAlpha
            alpha = alphaVals(i);
            fprintf('    alpha=%.2f ... ', alpha);
            data.domain.load_cases(1).factor = alpha;
            data.domain.load_cases(2).factor = 1 - alpha;

            csvBefore = fullfile(outDir, 'topopt_config_correlation.csv');
            if isfile(csvBefore), delete(csvBefore); end
            try
                [xFin, omega, ~, nIt, ~] = run_topopt_from_json(data);
                res_all{mi,f}.omega1(i)   = omega(1);
                res_all{mi,f}.omega2(i)   = omega(2);
                res_all{mi,f}.grayness(i) = mean(4*xFin.*(1-xFin));
                res_all{mi,f}.nIter(i)    = nIt;
                fprintf('done (omega1=%.2f, g=%.4f)\n', omega(1), res_all{mi,f}.grayness(i));
            catch ME
                fprintf('FAILED: %s\n', ME.message); continue;
            end
            csvAfter = fullfile(outDir,'topopt_config_correlation.csv');
            if isfile(csvAfter)
                macD = localParseMACFromCSV(csvAfter);
                res_all{mi,f}.macData{i} = macD;
                tag = sprintf('exp3_%s_%s_alpha%.2f', mc.label, regexprep(formLabels{f},'[^A-Za-z0-9]','_'), alpha);
                copyfile(csvAfter, fullfile(outDir,[tag '_correlation.csv']));
            end
        end
        fprintf('\n');
    end
end

localPrintConvergenceTable(alphaVals, omega_init_all, meshConfigs, res_all, formLabels, MAC_THRESHOLD);
localPrintMACGainConvergence(alphaVals, omega_init_all, meshConfigs, res_all, formLabels, MAC_THRESHOLD);

results = struct('alphaVals',alphaVals,'omega_init_all',omega_init_all, ...
    'meshConfigs',{meshConfigs},'res_all',{res_all},'formLabels',{formLabels}, ...
    'MAC_threshold',MAC_THRESHOLD);
save(fullfile(outDir,'exp3_mesh_convergence_results.mat'),'results');
fprintf('Results saved.\n');
end

% =========================================================================
function localPrintConvergenceTable(alphaVals,omega_init_all,meshConfigs,res_all,formLabels,MAC_THRESH)
nMeshes=numel(meshConfigs); nForms=2; nAlpha=numel(alphaVals);
sep=repmat('-',1,110);
for f=1:nForms
    fprintf('\n%s\n', sep);
    fprintf('Formulation: %s -- omega1 and grayness across meshes\n',formLabels{f});
    fprintf('%s\n',sep);
    fprintf('%-6s','alpha');
    for mi=1:nMeshes
        fprintf('  | %-8s omega1  g', meshConfigs{mi}.label);
    end
    fprintf('\n%s\n',sep);
    for i=1:nAlpha
        fprintf('%.2f  ',alphaVals(i));
        for mi=1:nMeshes
            o1=res_all{mi,f}.omega1(i); g=res_all{mi,f}.grayness(i);
            fprintf('  | %8.3f %6.4f', o1, g);
        end
        fprintf('\n');
    end
end
end

% =========================================================================
function localPrintMACGainConvergence(alphaVals,omega_init_all,meshConfigs,res_all,formLabels,MAC_THRESH)
nMeshes=numel(meshConfigs); nForms=2;
sep=repmat('-',1,120);
fprintf('\n%s\n',sep);
fprintf('MAC-tracked gains and validity across meshes and formulations\n');
fprintf('%s\n',sep);
fprintf('%-6s  %-25s','alpha','formulation');
for mi=1:nMeshes
    fprintf('  | %-9s gain1  MAC1  [v?] gain2  MAC2  [v?]',meshConfigs{mi}.label);
end
fprintf('\n%s\n',sep);

for i=1:numel(alphaVals)
    for f=1:nForms
        fprintf('%.2f    %-25s', alphaVals(i), formLabels{f});
        for mi=1:nMeshes
            macD=res_all{mi,f}.macData{i};
            om0=omega_init_all(mi,:);
            if isempty(macD)||isempty(macD.best_mode)
                fprintf('  |     N/A    N/A    N/A   N/A    N/A    N/A');
            else
                nI=numel(macD.best_mode);
                m1=macD.best_mode(min(1,nI)); mac1=macD.best_mac(min(1,nI)); ot1=macD.best_omega(min(1,nI));
                m2=macD.best_mode(min(2,nI)); mac2=macD.best_mac(min(2,nI)); ot2=macD.best_omega(min(2,nI));
                g1=ot1/om0(1); g2=ot2/om0(2);
                v1=ternary(mac1>=MAC_THRESH,'Y','N');
                v2=ternary(mac2>=MAC_THRESH,'Y','N');
                fprintf('  | %5.2fx %5.3f  [%s]  %5.2fx %5.3f  [%s]',g1,mac1,v1,g2,mac2,v2);
            end
        end
        fprintf('\n');
    end
    fprintf('%s\n',sep);
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
    fprintf('    [Warning] CSV parse error: %s\n',ME.message);
end
end

function omega = localParseOmega(label)
omega=NaN;
tok=regexp(label,'\(([0-9eE+\-\.]+)_rad','tokens','once');
if ~isempty(tok), omega=str2double(tok{1}); return; end
tok2=regexp(label,'([0-9eE+\-\.]+)','tokens');
for k=1:numel(tok2), v=str2double(tok2{k}{1}); if ~isnan(v)&&v>0, omega=v; return; end; end
end

function omegas = localSolidEigenpairs(jsonPath, nModes)
% Return initial-domain eigenfrequencies (x = volfrac) from a correlation CSV.
% These are the omega_0 reference values: the paper reports 145.44 and 362.67
% rad/s for the 400x50 clamped beam at volfrac=0.5.
%
% Method: run 1 iteration using the given JSON (which has correlation enabled),
% then read the CSV row labels which are tagged with the initial-domain omegas.
omegas = NaN(nModes, 1);
if ~isfile(jsonPath)
    fprintf('    [Warning] localSolidEigenpairs: JSON not found: %s\n', jsonPath);
    return;
end
data = jsondecode(fileread(jsonPath));
data.optimization.max_iters     = 1;
data.optimization.convergence_tol = 10.0;   % ensure 1 iter completes
data.domain.load_cases(1).factor = 1.0;
data.domain.load_cases(2).factor = 0.0;
data.postprocessing.visualize_live       = false;
data.postprocessing.save_final_image     = false;
data.postprocessing.save_snapshot_image  = false;
data.postprocessing.save_frequency_iterations = false;

csvBefore = fullfile(pwd, 'topopt_config_correlation.csv');
if isfile(csvBefore), delete(csvBefore); end

try
    run_topopt_from_json(data);
catch ME
    fprintf('    [Warning] localSolidEigenpairs probe failed: %s\n', ME.message);
    return;
end

csvAfter = fullfile(pwd, 'topopt_config_correlation.csv');
if isfile(csvAfter)
    try
        [~, rowLabels, ~] = readCorrelationCSV(csvAfter);
        for i = 1:min(nModes, numel(rowLabels))
            omegas(i) = localParseOmega(rowLabels{i});
        end
    catch ME2
        fprintf('    [Warning] CSV parse failed: %s\n', ME2.message);
    end
    delete(csvAfter);
end
end

function s=ternary(c,a,b), if c, s=a; else, s=b; end, end

function localEnsurePaths(scriptDir)
repoRoot=fileparts(fileparts(scriptDir));
toolsDir=fullfile(repoRoot,'tools','Matlab');
if exist(toolsDir,'dir')==7, addpath(toolsDir); end
addpath(genpath(fullfile(repoRoot,'analysis')));
end
