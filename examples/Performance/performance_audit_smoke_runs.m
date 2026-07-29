%PERFORMANCE_AUDIT_SMOKE_RUNS Exercise the three real wrapper branches on
% a cheap 32-by-4 fixture.  This is a control-flow/telemetry diagnostic,
% not performance evidence and not a substitute for the unavailable final
% benchmark topologies.

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));
configPath = fullfile(scriptDir, 'performance_comparison.json');
resultPath = fullfile(scriptDir, 'performance_audit_smoke_results.json');
addpath(fullfile(repoRoot, 'tools', 'Matlab'));

cfgBase = jsondecode(fileread(configPath));
cfgBase.domain.mesh.nelx = 32;
cfgBase.domain.mesh.nely = 4;
cfgBase.optimization.max_iters = 12;
cfgBase.optimization.convergence_tol = 1e-12;
cfgBase.postprocessing.save_final_image = false;
cfgBase.postprocessing.save_snapshot_image = false;
cfgBase.postprocessing.save_frequency_iterations = false;
cfgBase.postprocessing.compute_modes = 3;
cfgBase.optimization.yuksel = struct( ...
    'stage1_max_iters',12, ...
    'stage1_tol',1e-12, ...
    'stage2_tol',1e-12, ...
    'final_mode_count',3);

approaches = {'Olhoff','Yuksel','ourApproach'};
records = cell(numel(approaches),1);
for k = 1:numel(approaches)
    cfg = cfgBase;
    cfg.optimization.approach = approaches{k};
    wallTimer = tic;
    [x,omega,tIter,nIter,mem,diag] = run_topopt_from_json(cfg);
    wallTime = toc(wallTimer);
    independentOmega = localIndependentLinearModes(x,cfg,3);

    record = struct();
    record.approach = approaches{k};
    record.iterations_returned = nIter;
    record.average_iteration_time_seconds = tIter;
    record.reconstructed_loop_time_seconds = tIter*nIter;
    record.wrapper_wall_time_seconds = wallTime;
    record.reported_incremental_rss_mb = mem;
    record.returned_omega_rad_per_s = omega(:);
    record.independent_common_linear_mass_omega_rad_per_s = independentOmega(:);
    record.omega1_relative_difference_from_common_recompute = ...
        abs(omega(1)-independentOmega(1))/max(abs(independentOmega(1)),1e-14);
    record.final_density = struct( ...
        'minimum',min(x), ...
        'maximum',max(x), ...
        'mean',mean(x), ...
        'gray_fraction_0p1_to_0p9',mean(x>0.1 & x<0.9));
    record.diagnostics_top_level_fields = fieldnames(diag);
    record.has_explicit_converged_field = isfield(diag,'converged');
    record.has_explicit_termination_reason = ...
        isfield(diag,'termination_reason') || isfield(diag,'stop_reason');

    switch lower(approaches{k})
        case 'olhoff'
            record.iteration_cap_semantics = ...
                'single loop: maxiter=12; returned iterations=12';
            if isfield(diag,'history') && isfield(diag.history,'change_x')
                record.last_change = diag.history.change_x(end);
            elseif isfield(diag,'change_x')
                record.last_change = diag.change_x(end);
            else
                record.last_change = NaN;
            end
        case 'yuksel'
            record.iteration_cap_semantics = ...
                ['two loops: stage1_max_iters=12 plus stage2 maxiter=12; ' ...
                 'optimization.max_iters is not a total budget'];
            if isfield(diag,'timing')
                record.stage1_iterations = diag.timing.stage1_iterations;
                record.stage2_iterations = diag.timing.stage2_iterations;
            end
            if isfield(diag,'stage1') && isfield(diag.stage1,'ch') && ~isempty(diag.stage1.ch)
                record.stage1_last_change = diag.stage1.ch(end);
            end
            if isfield(diag,'stage2') && isfield(diag.stage2,'ch') && ~isempty(diag.stage2.ch)
                record.stage2_last_change = diag.stage2.ch(end);
            end
        case 'ourapproach'
            record.iteration_cap_semantics = ...
                'single loop: max_iters=12; returned iterations=12';
            if isfield(diag,'last_change')
                record.last_change = diag.last_change;
            else
                record.last_change = NaN;
            end
    end
    records{k} = record;
end

result = struct();
result.audit = 'performance_comparison wrapper smoke runs';
result.status = 'completed';
result.timestamp = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
result.matlab_version = version;
result.fixture = struct( ...
    'source_config','examples/Performance/performance_comparison.json', ...
    'mesh','32x4', ...
    'max_iters',12, ...
    'convergence_tol',1e-12, ...
    'purpose',['Control-flow, timing-boundary, termination-schema, and final-eigenvalue ' ...
        'checks only. No values are reviewer-facing performance evidence.']);
result.methods = records;
result.interpretation = { ...
    'wrapper_wall_time includes setup and postprocessing that tIter*nIter excludes', ...
    'Yuksel can return twice optimization.max_iters because the wrapper applies the budget separately to two stages', ...
    'the common independent recomputation uses stiffness SIMP and linear mass for every returned topology', ...
    'a Yuksel frequency mismatch is expected because that branch returns frequencies from its discontinuous modified mass law'};

fid = fopen(resultPath,'w');
if fid < 0
    error('PerformanceAuditSmoke:Write','Unable to create %s',resultPath);
end
cleanupFid = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(result,PrettyPrint=true));
fprintf('\nPerformance audit wrapper smoke runs completed.\n');
fprintf('Saved: %s\n',resultPath);
for k = 1:numel(records)
    record = records{k};
    fprintf('  %-11s iterations=%d loop=%.3fs wall=%.3fs omega1/common diff=%.3e\n', ...
        record.approach,record.iterations_returned, ...
        record.reconstructed_loop_time_seconds, ...
        record.wrapper_wall_time_seconds, ...
        record.omega1_relative_difference_from_common_recompute);
end

% =========================================================================
function omega = localIndependentLinearModes(x,cfg,nModes)
nelx = double(cfg.domain.mesh.nelx);
nely = double(cfg.domain.mesh.nely);
L = double(cfg.domain.size.length);
H = double(cfg.domain.size.height);
E0 = double(cfg.material.E);
Emin = E0*double(cfg.void_material.E_min_ratio);
rho0 = double(cfg.material.rho);
rhoMin = double(cfg.void_material.rho_min);
penal = double(cfg.optimization.penalization);
nu = double(cfg.material.nu);
thickness = double(cfg.domain.thickness);
hx = L/nelx;
hy = H/nely;
[KE,ME] = localElementMatrices(hx,hy,nu,thickness);
[~,iK,jK] = localEdofData(nelx,nely);
nDof = 2*(nelx+1)*(nely+1);
sK = reshape(KE(:)*(Emin+(E0-Emin)*x(:)'.^penal),[],1);
sM = reshape(ME(:)*(rhoMin+(rho0-rhoMin)*x(:)'),[],1);
K = sparse(iK,jK,sK,nDof,nDof);
M = sparse(iK,jK,sM,nDof,nDof);
K = (K+K')/2;
M = (M+M')/2;

[X,Y] = meshgrid(linspace(0,L,nelx+1),linspace(0,H,nely+1));
nodeX = X(:);
nodeY = Y(:);
fixed = [];
supports = cfg.bc.supports;
for s = 1:numel(supports)
    support = supports(s);
    location = double(support.location(:));
    [~,node] = min((nodeX-location(1)).^2+(nodeY-location(2)).^2);
    dofs = cellstr(string(support.dofs));
    if any(strcmp(dofs,'ux')), fixed(end+1,1) = 2*node-1; end %#ok<AGROW>
    if any(strcmp(dofs,'uy')), fixed(end+1,1) = 2*node; end %#ok<AGROW>
end
free = setdiff((1:nDof)',unique(fixed));
[~,D] = eig(full(K(free,free)),full(M(free,free)),'vector');
D = sort(real(D(isfinite(D) & isreal(D) & D>0)),'ascend');
omega = sqrt(D(1:min(nModes,numel(D))));
omega(end+1:nModes,1) = NaN;
end

function [edofMat,iK,jK] = localEdofData(nelx,nely)
edofMat = zeros(nelx*nely,8);
for elx = 0:nelx-1
    for ely = 0:nely-1
        el = ely+elx*nely+1;
        n1 = (nely+1)*elx+ely;
        n2 = (nely+1)*(elx+1)+ely;
        edofMat(el,:) = [2*n1+1,2*n1+2,2*n2+1,2*n2+2, ...
            2*(n2+1)+1,2*(n2+1)+2,2*(n1+1)+1,2*(n1+1)+2];
    end
end
iK = reshape(kron(edofMat,ones(1,8))',[],1);
jK = reshape(kron(edofMat,ones(8,1))',[],1);
end

function [KE,ME] = localElementMatrices(hx,hy,nu,thickness)
D = (1/(1-nu^2))*[1,nu,0;nu,1,0;0,0,(1-nu)/2];
invJ = [2/hx,0;0,2/hy];
detJ = hx*hy/4;
gp = 1/sqrt(3);
KE = zeros(8,8);
for xi = [-gp,gp]
    for eta = [-gp,gp]
        dNdxi = 0.25*[-(1-eta),(1-eta),(1+eta),-(1+eta)];
        dNdeta = 0.25*[-(1-xi),-(1+xi),(1+xi),(1-xi)];
        dNxy = invJ*[dNdxi;dNdeta];
        B = zeros(3,8);
        B(1,1:2:end)=dNxy(1,:);
        B(2,2:2:end)=dNxy(2,:);
        B(3,1:2:end)=dNxy(2,:);
        B(3,2:2:end)=dNxy(1,:);
        KE = KE+B'*D*B*detJ*thickness;
    end
end
Ms = (hx*hy*thickness/36)*[4,2,1,2;2,4,2,1;1,2,4,2;2,1,2,4];
ME = kron(Ms,eye(2));
end
