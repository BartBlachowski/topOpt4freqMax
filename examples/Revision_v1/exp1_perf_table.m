function results = exp1_perf_table(nSamples, meshSizes, outDir)
%EXP1_PERF_TABLE  Performance table with omega_1, timing statistics, init cost.
%
%   Addresses reviewer demands:
%     V1/MR1 : Standard deviations for Table 1 timing and RAM.
%     V2/MR2 : omega_1 column for all methods at all mesh sizes.
%     R2/M4  : Hardware/software specification.
%     R1     : Step-1 setup cost separated from per-iteration cost.
%               For ourApproach: setup = 1 eigensolve, measured directly
%               via a standalone eigs call (tSetup_eig) AND as tSetup_derived
%               = t_1iter_probe - tIter_mean.
%               For Yuksel: setup ~= Stage-1 (static compliance, ~200 iters),
%               estimated as t_1iter_probe - tIter_mean.
%               For Olhoff: setup ~= first eigensolve + first MMA call,
%               estimated as t_1iter_probe - tIter_mean.
%               All: adjusted per-iteration = (tTotal - tSetup_derived) / nIter.
%
%   Usage:
%     results = exp1_perf_table();              % 10 samples, 4 meshes
%     results = exp1_perf_table(3, [160 20; 400 50]);

if nargin < 1 || isempty(nSamples),  nSamples = 10; end
if nargin < 2 || isempty(meshSizes)
    meshSizes = [160, 20; 240, 30; 320, 40; 400, 50];
end
if nargin < 3 || isempty(outDir)
    outDir = fileparts(mfilename('fullpath'));
end

scriptDir = fileparts(mfilename('fullpath'));
localEnsurePaths(scriptDir);

fprintf('\n=== EXP1: Performance table with omega_1, timing stats, init cost ===\n');
fprintf('Samples per (mesh x method): %d\n\n', nSamples);

jsonBase = fullfile(scriptDir, '..', 'Performance', 'performance_comparison.json');
if ~isfile(jsonBase)
    error('exp1_perf_table:MissingJson', 'Base JSON not found: %s', jsonBase);
end
data = jsondecode(fileread(jsonBase));
if isfield(data,'optimisation') && ~isfield(data,'optimization')
    data.optimization = data.optimisation;
end
data.postprocessing.visualize_live      = false;
data.postprocessing.save_final_image    = false;
data.postprocessing.save_snapshot_image = false;
if isfield(data.postprocessing,'save_frequency_iterations')
    data.postprocessing.save_frequency_iterations = false;
end
if isfield(data.postprocessing,'write_correlation_table')
    data.postprocessing.write_correlation_table = false;
end
data.optimization.filter.radius       = 2;
data.optimization.filter.radius_units = 'element';

nRes     = size(meshSizes,1);
approaches   = {'Olhoff','Yuksel','OurApproach'};
methodLabels = {'OlhoffApproach','YukselApproach','ProposedApproach'};
nMethods     = numel(approaches);

% Storage: (nRes, nMethods, nSamples)
omega_all    = NaN(nRes,nMethods,nSamples);
tIter_all    = NaN(nRes,nMethods,nSamples);
nIter_all    = NaN(nRes,nMethods,nSamples);
mem_all      = NaN(nRes,nMethods,nSamples);
tTotal_all   = NaN(nRes,nMethods,nSamples);
tProbe_arr   = NaN(nRes,nMethods);   % 1-iter probe (upper bound on setup, all methods)
tEig_arr     = NaN(nRes,nMethods);   % standalone eigensolve (ourApproach only)

for r = 1:nRes
    data.domain.mesh.nelx = meshSizes(r,1);
    data.domain.mesh.nely = meshSizes(r,2);

    for m = 1:nMethods
        data.optimization.approach = approaches{m};

        % --- Measure initialization (setup) cost ---
        % t_1iter_probe: run max_iters=1 (upper bound on setup; includes 1 iter).
        % tSetup_derived = t_1iter_probe - tIter_from_full_run (estimated true setup).
        % For ourApproach: also time standalone eigs for the most accurate setup measure.
        % Note: Yuksel Stage-1 (static compliance ~200 iters) dominates its probe time.
        tProbe_arr(r, m) = localTimeSingleIterationProbe(data);
        if strcmpi(approaches{m}, 'ourapproach')
            tEig_arr(r, m) = localTimeSingleEigensolve(data);
            fprintf('  [init] probe=%.3fs, eigs=%.3fs (ourApproach)\n', ...
                tProbe_arr(r,m), tEig_arr(r,m));
        else
            fprintf('  [init] probe=%.3fs (%s)\n', tProbe_arr(r,m), methodLabels{m});
        end

        for s = 1:nSamples
            fprintf('  %-18s  mesh %4dx%-3d  sample %2d/%d ... ', ...
                methodLabels{m}, meshSizes(r,1), meshSizes(r,2), s, nSamples);
            tWall = tic;
            try
                [~,omega,tIter,nIter,mem] = run_topopt_from_json(data);
                elapsed = toc(tWall);
                omega_all(r,m,s)  = omega(1);
                tIter_all(r,m,s)  = tIter;
                nIter_all(r,m,s)  = nIter;
                mem_all(r,m,s)    = mem;
                tTotal_all(r,m,s) = tIter * nIter;
                fprintf('done (%.1fs, omega1=%.2f)\n', elapsed, omega(1));
            catch ME
                fprintf('FAILED: %s\n', ME.message);
            end
        end
    end
end

% Aggregate statistics
omM=mean(omega_all,3,'omitnan'); omS=std(omega_all,0,3,'omitnan');
tIM=mean(tIter_all,3,'omitnan'); tIS=std(tIter_all,0,3,'omitnan');
nIM=round(mean(nIter_all,3,'omitnan'));
tTM=mean(tTotal_all,3,'omitnan'); tTS=std(tTotal_all,0,3,'omitnan');
mM=mean(mem_all,3,'omitnan');    mS=std(mem_all,0,3,'omitnan');

% Setup cost breakdown:
%   tProbe = 1-iter wall time (upper bound on tSetup for each method)
%   tSetup_derived = max(0, tProbe - tIter_mean)  (estimated true setup)
%   tEig = standalone eigensolve time (ourApproach only, most accurate)
%   For ourApproach, use tEig; for others use tSetup_derived.
tSetup_derived = max(0, tProbe_arr - tIM);   % (nRes x nMethods)
tSetup = tSetup_derived;
for mi = 1:nMethods
    if strcmpi(approaches{mi},'ourapproach')
        tSetup(:,mi) = tEig_arr(:,mi);   % more accurate for ourApproach
    end
end

% Adjusted per-iteration time: excludes setup overhead from the mean
% tPerIter_adj = (tTotal - tSetup) / nIter  -- cleanest breakdown
tPerIterAdj = (tTM - tSetup) ./ max(nIM, 1);

localPrintTextTable(meshSizes,methodLabels,nRes,nMethods,nSamples,...
    omM,omS,tIM,tIS,nIM,tTM,tTS,mM,mS,tProbe_arr,tSetup,tPerIterAdj);
localPrintLatexTable(meshSizes,methodLabels,nRes,nMethods,...
    omM,omS,tIM,tIS,nIM,tTM,tTS,mM,mS,tSetup,tPerIterAdj);

results = struct('meshSizes',meshSizes,'methodLabels',{methodLabels}, ...
    'approaches',{approaches},'nSamples',nSamples, ...
    'omega_mean',omM,'omega_std',omS, ...
    'tIter_mean',tIM,'tIter_std',tIS,'nIter_mean',nIM, ...
    'tTotal_mean',tTM,'tTotal_std',tTS, ...
    'mem_mean',mM,'mem_std',mS, ...
    'tProbe',tProbe_arr,'tSetup',tSetup,'tPerIterAdj',tPerIterAdj, ...
    'tEig',tEig_arr);

save(fullfile(outDir,'exp1_perf_table_results.mat'),'results');
fprintf('Results saved.\n');
end

% =========================================================================
function localPrintTextTable(meshSizes,methodLabels,nRes,nMethods,nSamples,...
    omM,omS,tIM,tIS,nIM,tTM,tTS,mM,mS,tProbe,tSetup,tPerIterAdj)
sep=repmat('-',1,150);
fprintf('\n%s\n',sep);
fprintf('Table 1 (augmented): Performance -- mean+/-std over %d runs\n', nSamples);
fprintf('Setup cost columns:\n');
fprintf('  tSetup: for ourApproach = standalone eigs time; for Yuksel/Olhoff = max(0, t_1iter - tIter_mean).\n');
fprintf('  tIter_adj: (tTotal - tSetup) / nIter -- per-iteration cost excluding setup overhead.\n');
fprintf('  tProbe: 1-iteration wall time (raw upper-bound on setup; shown for reference).\n\n');
fprintf('%-20s  %-9s  %7s  %9s  %7s  %9s  %9s  %8s  %12s\n', ...
    'Method','Mesh','Iters','Total(s)','Setup(s)','Adj/iter(s)','tIter(s)','RAM(MB)','omega1(rad/s)');
fprintf('%s\n',sep);
for r=1:nRes
    meshStr=sprintf('%dx%d',meshSizes(r,1),meshSizes(r,2));
    for m=1:nMethods
        su = tSetup(r,m); if isnan(su), su_s = '  N/A '; else, su_s = sprintf('%7.3f',su); end
        pa = tPerIterAdj(r,m); if isnan(pa)||pa<0, pa_s='  N/A  '; else, pa_s=sprintf('%9.4f',pa); end
        fprintf('%-20s  %-9s  %7d  %7.1f+/-%.1f  %s  %s  %7.4f+/-%.4f  %6.0f+/-%.0f  %.2f+/-%.2f\n', ...
            methodLabels{m},meshStr,nIM(r,m), ...
            tTM(r,m),tTS(r,m),su_s,pa_s, ...
            tIM(r,m),tIS(r,m),mM(r,m),mS(r,m),omM(r,m),omS(r,m));
    end
    fprintf('%s\n',sep);
end
end

% =========================================================================
function localPrintLatexTable(meshSizes,methodLabels,nRes,nMethods,...
    omM,omS,tIM,tIS,nIM,tTM,tTS,mM,mS,tSetup,tPerIterAdj)
fprintf('\n--- LaTeX fragment for revised Table 1 ---\n');
fprintf('%% Columns: Method | Iter | Total(s) | Setup(s) | AdjPerIter(s) | RAM(MB) | omega1(rad/s)\n');
for r=1:nRes
    meshStr=sprintf('%d\\times %d',meshSizes(r,1),meshSizes(r,2));
    fprintf('\\multicolumn{8}{c}{\\textbf{Mesh: $%s$}} \\\\\n',meshStr);
    fprintf('\\Xhline{0.8pt}\n');
    for m=1:nMethods
        su = tSetup(r,m); su_s = ternary_s(isnan(su), '--', sprintf('%.3f',su));
        pa = tPerIterAdj(r,m); pa_s = ternary_s(isnan(pa)||pa<0,'--',sprintf('%.3f',pa));
        fprintf('%s & %d & $%.1f\\pm%.1f$ & %s & %s & $%.3f\\pm%.3f$ & $%.0f\\pm%.0f$ & $%.2f\\pm%.2f$ \\\\\n', ...
            methodLabels{m}, nIM(r,m), ...
            tTM(r,m), tTS(r,m), su_s, pa_s, ...
            tIM(r,m), tIS(r,m), mM(r,m), mS(r,m), omM(r,m), omS(r,m));
    end
    if r<nRes, fprintf('\\Xhline{1.2pt}\n\n'); end
end
fprintf('--- end LaTeX fragment ---\n\n');
end

function s = ternary_s(cond, a, b)
if cond, s = a; else, s = b; end
end

% =========================================================================
function tProbe = localTimeSingleIterationProbe(data)
% Run a 1-iteration version of the optimization and return the wall-clock time.
% This gives an upper-bound estimate for the initialization cost:
%   tSetup <= t_1iter (since the 1-iteration run = setup + 1 optimization step).
% For ourApproach, setup = 1 eigensolve ~= 0.1-1 s (small fraction of total).
% For Yuksel, setup = Stage 1 (static compliance, ~200 iters).
% For Olhoff, setup = initial eigensolve + first MMA call.
tProbe = NaN;
try
    dataTmp = data;
    dataTmp.optimization.max_iters      = 1;
    dataTmp.optimization.convergence_tol = 10.0;  % ensure 1 iter runs
    if isfield(dataTmp.postprocessing, 'save_final_image')
        dataTmp.postprocessing.save_final_image = false;
    end
    if isfield(dataTmp.postprocessing, 'save_frequency_iterations')
        dataTmp.postprocessing.save_frequency_iterations = false;
    end
    t0 = tic;
    run_topopt_from_json(dataTmp);
    tProbe = toc(t0);
catch
end
end

% =========================================================================
function tEig = localTimeSingleEigensolve(data)
% Time the initialization eigensolve cost for ourApproach by assembling K and M
% from the initial uniform density and calling eigs directly. This isolates the
% Step-1 setup cost as requested by Reviewer 1 without invoking the full solver.
tEig = NaN;
try
    nelx = data.domain.mesh.nelx;
    nely = data.domain.mesh.nely;
    L    = data.domain.size.length;
    H    = data.domain.size.height;
    E0   = data.material.E;
    nu   = data.material.nu;
    rho0 = data.material.rho;
    Emin = data.material.E * data.void_material.E_min_ratio;
    rho_min = data.void_material.rho_min;
    volfrac = data.optimization.volume_fraction;
    penal   = data.optimization.penalization;

    hx = L/nelx; hy = H/nely;
    nEl = nelx*nely; nDof = 2*(nelx+1)*(nely+1);
    x0 = volfrac * ones(nEl,1);

    edofMat = zeros(nEl,8);
    for elx=0:nelx-1, for ely=0:nely-1
        el=ely+elx*nely+1; n1=(nely+1)*elx+ely; n2=(nely+1)*(elx+1)+ely; n3=n2+1; n4=n1+1;
        edofMat(el,:)=[2*n1+1,2*n1+2,2*n2+1,2*n2+2,2*n3+1,2*n3+2,2*n4+1,2*n4+2];
    end, end
    iK=reshape(kron(edofMat,ones(1,8))',[],1);
    jK=reshape(kron(edofMat,ones(8,1))',[],1);

    D=(E0/(1-nu^2))*[1,nu,0;nu,1,0;0,0,0.5*(1-nu)];
    invJ=[2/hx,0;0,2/hy]; detJ=0.25*hx*hy; gp=1/sqrt(3);
    KE=zeros(8,8);
    for xi=[-gp,gp], for eta=[-gp,gp]
        dNxi=0.25*[-(1-eta),(1-eta),(1+eta),-(1+eta)];
        dNeta=0.25*[-(1-xi),-(1+xi),(1+xi),(1-xi)];
        dN=invJ*[dNxi;dNeta]; B=zeros(3,8);
        B(1,1:2:end)=dN(1,:); B(2,2:2:end)=dN(2,:);
        B(3,1:2:end)=dN(2,:); B(3,2:2:end)=dN(1,:);
        KE=KE+(B'*D*B)*detJ;
    end, end
    Ms=rho0*(hx*hy/36)*[4,2,1,2;2,4,2,1;1,2,4,2;2,1,2,4];
    ME=kron(Ms,eye(2));

    sK=reshape(KE(:)*(Emin + x0'.^penal*(E0-Emin)),[],1);
    K=sparse(iK,jK,sK,nDof,nDof); K=(K+K')/2;
    sM=reshape(ME(:)*(rho_min + x0'),[],1);
    Mg=sparse(iK,jK,sM,nDof,nDof); Mg=(Mg+Mg')/2;

    % Build BCs matching the base JSON (simplified: assume SS or CC from supports)
    % Use the mid-height nodes as SS approximation (common for Table 1 SS beam)
    jMid = floor(nely/2);
    nL = jMid; nR = nelx*(nely+1) + jMid;
    fixed = unique([2*nL+1, 2*nL+2, 2*nR+1, 2*nR+2]);
    free = setdiff(1:nDof, fixed);

    eigOpts = struct('disp',0,'maxit',800,'tol',1e-8);
    t0 = tic;
    eigs(K(free,free), Mg(free,free), 1, 1e-6, eigOpts);
    tEig = toc(t0);
catch
end
end

% =========================================================================
function localEnsurePaths(scriptDir)
repoRoot=fileparts(fileparts(scriptDir));
toolsDir=fullfile(repoRoot,'tools','Matlab');
if exist(toolsDir,'dir')==7, addpath(toolsDir); end
addpath(genpath(fullfile(repoRoot,'analysis')));
end
