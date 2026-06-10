%RUN_ALL_REVISION_EXPERIMENTS  Master script for all peer-review revision experiments.
%
%   Runs every experiment demanded by reviewers for the paper:
%   "Frequency maximization of planar structures using quasi-static
%   approximation in topology optimization"
%
%   MATLAB EXECUTABLE:  /Applications/MATLAB_R2025b.app/bin/matlab
%
%   HOW TO RUN
%   ----------
%   cd examples/Revision_v1
%   run_all_revision_experiments          % full paper-quality mode
%   run_all_revision_experiments('fast')  % quick smoke-test
%
%   Individual experiments:
%     exp1_perf_table()
%     exp2_clamped_beam()
%     exp2b_building()
%     exp3_mesh_convergence()
%     exp4_sensitivity_ablation()
%     exp5_scaling()
%
%   REVIEWER ISSUES ADDRESSED
%   -------------------------
%   Exp 1  V1/MR1  Std devs for Table 1 timing/RAM.
%          V2/MR2  omega_1 for all methods and mesh sizes.
%          R2/MR4  Hardware + MATLAB specification.
%          R1      Init eigensolve cost separated from per-iteration cost.
%
%   Exp 2  V5/m5   omega1^0 and omega2^0 for clamped beam.
%          M5      Topo-mode 1 and 2 shape plots at alpha=1.0.
%          M6      Grayness g for all topologies.
%          CR1/C4  alpha=0.75 re-run with full MAC/freq diagnostic.
%          C2      Comparison: semi_harmonic (paper) vs harmonic Eq.7.
%          V4/MR4  MAC validity threshold enforced.
%
%   Exp 2b V3/MR5  Building example: reproduces Tables 4/5.
%          M7      Spurious low-density mode check.
%
%   Exp 3  V3/MR5  Mesh convergence: 200x25 vs 400x50, both formulations.
%
%   Exp 4  CR2/C1  Sensitivity ablation: Variant A (paper, no load sens)
%                  vs Variant B (full semi-harmonic load sens), plus
%                  Variant C (harmonic frozen) vs Variant D (periodic).
%                  FD gradient check via debug_semi_harmonic confirms full
%                  gradient is accurately computed.
%
%   Exp 5  M4/mn8  Log-log fit corrects O(n_e^1.3) claim.
%
%   EXPECTED RUNTIMES (Apple M-class, R2025b)
%   -----------------------------------------
%   fast  mode (3 samples, small mesh subset): ~15-30 min
%   full  mode (10 samples, all 4 meshes):     ~5-12 hours

function run_all_revision_experiments(mode)

if nargin < 1 || isempty(mode), mode = 'full'; end
mode = lower(strtrim(char(mode)));

scriptDir = fileparts(mfilename('fullpath'));
prevDir   = pwd;
cleanupCd = onCleanup(@() cd(prevDir));
cd(scriptDir);

localEnsurePaths(scriptDir);

fprintf('\n');
fprintf('+===================================================================+\n');
fprintf('|  REVISION EXPERIMENTS -- topOpt4freqMax paper                      |\n');
fprintf('+===================================================================+\n\n');

exp6_hardware_info();

switch mode
    case 'fast'
        nSamples  = 2;
        meshTable = [160, 20; 400, 50];
        alphaVals = [1.0, 0.75, 0.5, 0.25, 0.0];
        fprintf('[MODE] fast -- %d samples, 2 meshes.\n\n', nSamples);
    case 'full'
        nSamples  = 10;
        meshTable = [160, 20; 240, 30; 320, 40; 400, 50];
        alphaVals = [1.0, 0.75, 0.5, 0.25, 0.0];
        fprintf('[MODE] full -- %d samples, 4 meshes.\n\n', nSamples);
    otherwise
        error('Unknown mode "%s". Use "fast" or "full".', mode);
end

allResults = struct();
timings    = struct();

[allResults.exp1, timings.exp1] = localRunExp('EXP1', 'Performance table + omega_1 + init timing', ...
    @() exp1_perf_table(nSamples, meshTable, scriptDir));

[allResults.exp2, timings.exp2] = localRunExp('EXP2', 'Clamped beam: initial freqs, mode shapes, semi_harmonic vs Eq.7', ...
    @() exp2_clamped_beam(alphaVals, scriptDir));

[allResults.exp2b, timings.exp2b] = localRunExp('EXP2b', 'Building example (Tables 4/5), spurious mode check', ...
    @() exp2b_building(alphaVals, scriptDir));

[allResults.exp3, timings.exp3] = localRunExp('EXP3', 'Mesh convergence 200x25 vs 400x50, both formulations', ...
    @() exp3_mesh_convergence(alphaVals, scriptDir));

[allResults.exp4, timings.exp4] = localRunExp('EXP4', 'Sensitivity ablation: Variant A/B + C/D + FD gradient check', ...
    @() exp4_sensitivity_ablation(scriptDir));

[allResults.exp5, timings.exp5] = localRunExp('EXP5', 'Scaling log-log (corrects O(n_e^1.3) claim)', ...
    @() exp5_scaling(localGetField(allResults, 'exp1'), scriptDir));

fprintf('\n====================================================================\n');
fprintf(' REVISION EXPERIMENTS COMPLETE\n');
fprintf('====================================================================\n');
localPrintFinalSummary(timings, scriptDir);

save(fullfile(scriptDir,'all_revision_results.mat'),'allResults','timings');
fprintf('All results saved to: all_revision_results.mat\n\n');
end

% =========================================================================
function [res, elapsed] = localRunExp(tag, desc, fn)
res = []; elapsed = NaN;
fprintf('\n????????????????????????????????????????????????????????????????????\n');
fprintf(' %s: %s\n', tag, desc);
fprintf('????????????????????????????????????????????????????????????????????\n');
t0 = tic;
try
    res     = fn();
    elapsed = toc(t0);
    fprintf('[%s] Completed in %.1fs.\n', tag, elapsed);
catch ME
    elapsed = NaN;
    fprintf('[%s] FAILED: %s\n', tag, ME.message);
    if ~isempty(ME.stack)
        fprintf('   Stack: %s line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
end
end

% =========================================================================
function v = localGetField(s, fn)
v = [];
if isstruct(s) && isfield(s, fn), v = s.(fn); end
end

% =========================================================================
function exp6_hardware_info()
fprintf('??? Hardware and Software Specification ????????????????????????????\n');
fprintf('  (Addresses R2, MR4)\n\n');
fprintf('  MATLAB version : %s\n', version);
try, fprintf('  CPU cores      : %d logical\n', feature('numcores')); catch, end
try, fprintf('  MATLAB threads : %d\n', maxNumCompThreads); catch, end
try
    [~,cpuStr]=system('sysctl -n machdep.cpu.brand_string 2>/dev/null');
    cpuStr=strtrim(cpuStr);
    if ~isempty(cpuStr), fprintf('  CPU            : %s\n',cpuStr); end
catch, end
try
    [~,hw]=memory;
    fprintf('  RAM            : %.1f GB physical\n',hw.PhysicalMemory.Total/1e9);
catch
    try
        [~,ms]=system('sysctl -n hw.memsize 2>/dev/null');
        mb=str2double(strtrim(ms));
        if ~isnan(mb), fprintf('  RAM            : %.1f GB\n',mb/1e9); end
    catch, end
end
try
    [~,os]=system('sw_vers -productVersion 2>/dev/null || uname -r');
    fprintf('  OS             : %s\n',strtrim(os));
catch, end
fprintf('  BLAS/LAPACK    : Apple Accelerate (MATLAB macOS arm64)\n\n');
end

% =========================================================================
function localPrintFinalSummary(timings, outDir)
exps={'exp1','exp2','exp2b','exp3','exp4','exp5'};
descs={
    'Exp1: Performance table (omega_1, std devs, init timing)'; ...
    'Exp2: Clamped beam (freqs, modes, semi_harm vs Eq.7, MAC)'; ...
    'Exp2b: Building (Tables 4/5, spurious modes)'; ...
    'Exp3: Mesh convergence (200x25 vs 400x50)'; ...
    'Exp4: Sensitivity ablation + FD gradient check'; ...
    'Exp5: Scaling analysis (log-log)' ...
};
fprintf('\n  Experiment summary:\n');
for k=1:numel(exps)
    fn=exps{k};
    if isfield(timings,fn) && ~isnan(timings.(fn))
        st=sprintf('OK (%.1fs)', timings.(fn));
    else
        st='FAILED or SKIPPED';
    end
    fprintf('    %-52s %s\n', descs{k}, st);
end
fprintf('\n  Output directory: %s\n', outDir);
dlist=dir(fullfile(outDir,'*.mat'));
for k=1:numel(dlist), fprintf('    %s\n',dlist(k).name); end
dlist=dir(fullfile(outDir,'*.png'));
for k=1:numel(dlist), fprintf('    %s\n',dlist(k).name); end
fprintf('\n');
end

% =========================================================================
function localEnsurePaths(scriptDir)
repoRoot=fileparts(fileparts(scriptDir));
toolsDir=fullfile(repoRoot,'tools','Matlab');
if exist(toolsDir,'dir')==7, addpath(toolsDir); end
addpath(genpath(fullfile(repoRoot,'analysis')));
addpath(scriptDir);
end
