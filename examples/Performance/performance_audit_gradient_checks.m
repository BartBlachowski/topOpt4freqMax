%PERFORMANCE_AUDIT_GRADIENT_CHECKS Controlled derivative checks for the
% legacy performance-comparison audit.
%
% This diagnostic does not alter any production solver.  It independently
% reconstructs the element-level mathematics used by the three branches on
% a small 32-by-4 beam and compares analytical derivatives with central
% finite differences at:
%   (1) a uniform design,
%   (2) a deterministic perturbed design, and
%   (3) a clipped/downsampled proxy from the saved CR2 late-cycle topology.
%
% The third design is a proxy, not the unavailable final 240-by-30 design
% from performance_comparison.m.  The distinction is recorded in the JSON.

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));
resultPath = fullfile(scriptDir, 'performance_audit_gradient_results.json');
proxyPath = fullfile(repoRoot, 'examples', 'Revision_v1', 'cr2', ...
    'mma_diagnostic', 'output', 'cr2_mma_variant_b_topology.csv');

model = localBuildModel(32, 4, 8, 1);
fdStepCompliance = 1e-6;
fdStepEigen = 2e-5;

xUniform = 0.5 * ones(model.nEl, 1);
[gx, gy] = meshgrid(1:model.nelx, 1:model.nely);
xPerturbed = 0.5 + 0.16 * sin(2*pi*gx/model.nelx) .* ...
    cos(pi*(gy-0.5)/model.nely) + 0.04 * cos(5*pi*gx/model.nelx);
xPerturbed = localMeanCorrect(min(0.82, max(0.18, xPerturbed(:))), 0.5);

if ~isfile(proxyPath)
    error('PerformanceAudit:MissingProxy', 'Missing late-cycle proxy: %s', proxyPath);
end
proxyTable = readtable(proxyPath);
if ~ismember('density', proxyTable.Properties.VariableNames) || height(proxyTable) ~= 3200
    error('PerformanceAudit:InvalidProxy', ...
        'Expected a 3200-row topology CSV with a density column: %s', proxyPath);
end
proxyFine = reshape(double(proxyTable.density), 20, 160);
proxyCoarse = zeros(model.nely, model.nelx);
for ex = 1:model.nelx
    for ey = 1:model.nely
        rows = (ey-1)*5 + (1:5);
        cols = (ex-1)*5 + (1:5);
        block = proxyFine(rows, cols);
        proxyCoarse(ey, ex) = mean(block(:));
    end
end
xProxy = min(0.98, max(0.02, proxyCoarse(:)));

designNames = {'uniform', 'perturbed', 'late_cycle_proxy'};
designs = {xUniform, xPerturbed, xProxy};

% Solid reference eigenpair used by the proposed semi-harmonic branch.
[~, phi0, omega0Sq] = localFirstEigenpair(ones(model.nEl, 1), model, 'linear');
phi0 = phi0 / sqrt(phi0' * localAssembleMass(ones(model.nEl, 1), model, 'linear') * phi0);

% A deterministic frozen displacement direction for the Yuksel stage-2 load.
uhat = phi0 / norm(phi0(model.free));

results = struct();
for d = 1:numel(designs)
    x = designs{d};
    idx = localTestIndices(x, 14);
    designResult = struct();
    designResult.minimum_density = min(x);
    designResult.maximum_density = max(x);
    designResult.mean_density = mean(x);
    designResult.tested_element_indices = idx;

    % ------------------------------------------------------------------
    % Proposed branch: evolving linear mass and frozen solid mode.
    % Raw complete derivative is exact; "omitted" holds the current load
    % fixed.  The production sensitivity filter is then checked against
    % the true unfiltered design-space objective.
    [ourObjective, ourComplete, ourOmitted, ourLoad] = ...
        localOurObjectiveGradient(x, model, phi0, omega0Sq);
    ourFullFd = localFd(@(z) localOurObjective(z, model, phi0, omega0Sq), ...
        x, idx, fdStepCompliance);
    ourFrozenFd = localFd(@(z) localFixedLoadObjective(z, model, ourLoad), ...
        x, idx, fdStepCompliance);
    ourFiltered = localOurSensitivityFilter(x, ourComplete, model);
    volumeFd = localFd(@(z) mean(z), x, idx, fdStepCompliance);
    volumeAnalytic = ones(model.nEl, 1) / model.nEl;

    our = struct();
    our.objective = ourObjective;
    our.raw_complete_vs_full_objective_fd = ...
        localCompare(ourComplete(idx), ourFullFd);
    our.raw_omitted_vs_frozen_load_fd = ...
        localCompare(ourOmitted(idx), ourFrozenFd);
    our.production_omitted_vs_full_objective_fd = ...
        localCompare(ourOmitted(idx), ourFullFd);
    our.production_filtered_complete_vs_full_objective_fd = ...
        localCompare(ourFiltered(idx), ourFullFd);
    our.volume_gradient_vs_fd = ...
        localCompare(volumeAnalytic(idx), volumeFd);
    designResult.proposed_branch = our;

    % ------------------------------------------------------------------
    % Yuksel stage 2: the implementation treats M(x)*uhat as a frozen
    % in-iteration load and omits dF/dx.  Check both the frozen-load
    % derivative it implements and the full derivative of the objective
    % that is actually re-evaluated after x changes.
    [yObjective, yComplete, yOmitted, yLoad] = ...
        localYukselObjectiveGradient(x, model, uhat);
    yFullFd = localFd(@(z) localYukselObjective(z, model, uhat), ...
        x, idx, fdStepCompliance);
    yFrozenFd = localFd(@(z) localFixedLoadObjective(z, model, yLoad), ...
        x, idx, fdStepCompliance);
    yFiltered = localYukselSensitivityFilter(x, yOmitted, model);

    yuksel = struct();
    yuksel.objective = yObjective;
    yuksel.independent_complete_vs_full_objective_fd = ...
        localCompare(yComplete(idx), yFullFd);
    yuksel.production_raw_omitted_vs_frozen_load_fd = ...
        localCompare(yOmitted(idx), yFrozenFd);
    yuksel.production_raw_omitted_vs_full_objective_fd = ...
        localCompare(yOmitted(idx), yFullFd);
    yuksel.production_filtered_omitted_vs_full_objective_fd = ...
        localCompare(yFiltered(idx), yFullFd);
    designResult.yuksel_stage2 = yuksel;

    % ------------------------------------------------------------------
    % Olhoff branch: check the lowest-eigenvalue chain rule through its
    % symmetric density filter and Heaviside projection, its volume
    % constraint, grayness objective, and its Eb constraint derivative.
    beta = 8;
    eta = 0.5;
    Eb = 1.4;
    [xPhys, dProj] = localOlhoffPhysical(x, model, beta, eta);
    [lambda1, ~, ~, dLambdaPhys] = ...
        localFirstEigenpair(xPhys, model, 'linear');
    dLambdaDesign = localOlhoffBackward(dLambdaPhys .* dProj, model);
    eigFd = localFd(@(z) localOlhoffLambda(z, model, beta, eta), ...
        x, idx, fdStepEigen);

    volumePhysicalGradient = localOlhoffBackward( ...
        (ones(model.nEl, 1) / model.nEl) .* dProj, model);
    olhoffVolumeFd = localFd(@(z) mean(localOlhoffPhysical(z, model, beta, eta)), ...
        x, idx, fdStepCompliance);

    grayWeight = 0.5 * beta / 64;
    grayPhysGradient = grayWeight * 4 * (1 - 2*xPhys) / model.nEl;
    grayDesignGradient = localOlhoffBackward(grayPhysGradient .* dProj, model);
    grayFd = localFd(@(z) localOlhoffGrayObjective(z, model, beta, eta, ...
        Eb, grayWeight), x, idx, fdStepCompliance);

    lambdaRef = 2e4;
    constraintDesignGradient = -dLambdaDesign / lambdaRef / max(1, Eb);
    constraintFd = localFd(@(z) localOlhoffConstraint(z, model, beta, eta, ...
        Eb, lambdaRef), x, idx, fdStepEigen);
    ebFd = (localOlhoffConstraint(x, model, beta, eta, Eb+fdStepCompliance, lambdaRef) - ...
        localOlhoffConstraint(x, model, beta, eta, Eb-fdStepCompliance, lambdaRef)) / ...
        (2*fdStepCompliance);
    ebProduction = 1 / max(1, Eb);
    ebCorrect = (lambda1/lambdaRef) / Eb^2;

    olhoff = struct();
    olhoff.beta = beta;
    olhoff.lambda_1 = lambda1;
    olhoff.eigenvalue_gradient_through_filter_projection_vs_fd = ...
        localCompare(dLambdaDesign(idx), eigFd);
    olhoff.volume_gradient_through_filter_projection_vs_fd = ...
        localCompare(volumePhysicalGradient(idx), olhoffVolumeFd);
    olhoff.gray_objective_gradient_through_filter_projection_vs_fd = ...
        localCompare(grayDesignGradient(idx), grayFd);
    olhoff.eigen_constraint_design_gradient_vs_fd = ...
        localCompare(constraintDesignGradient(idx), constraintFd);
    olhoff.eigen_constraint_Eb_derivative = struct( ...
        'production_value', ebProduction, ...
        'correct_analytic_value', ebCorrect, ...
        'finite_difference_value', ebFd, ...
        'production_relative_error', localScalarRelErr(ebProduction, ebFd), ...
        'correct_relative_error', localScalarRelErr(ebCorrect, ebFd));
    designResult.olhoff_branch = olhoff;

    results.(designNames{d}) = designResult;
end

switchEps = 1e-8;
massBelow = (0.1-switchEps)^6;
massAbove = 0.1+switchEps;

output = struct();
output.audit = 'performance_comparison controlled gradient checks';
output.status = 'completed';
output.timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
output.matlab_version = version;
output.fixture = struct( ...
    'nelx', model.nelx, ...
    'nely', model.nely, ...
    'length', model.L, ...
    'height', model.H, ...
    'filter_radius_element_widths', 2, ...
    'compliance_fd_step', fdStepCompliance, ...
    'eigenvalue_fd_step', fdStepEigen);
output.late_cycle_proxy = struct( ...
    'source', 'examples/Revision_v1/cr2/mma_diagnostic/output/cr2_mma_variant_b_topology.csv', ...
    'source_mesh', '160x20', ...
    'source_iteration_cap', 400, ...
    'construction', '5x5 block average to 32x4, then clipped to [0.02,0.98]', ...
    'limitation', ['This is only a representative late-cycle design from the same proposed ' ...
        'complete-sensitivity/MMA branch; it is not the unavailable 240x30 table design.']);
output.yuksel_mass_interpolation_jump = struct( ...
    'cutoff', 0.1, ...
    'density_below', 0.1-switchEps, ...
    'relative_mass_below', massBelow, ...
    'density_above', 0.1+switchEps, ...
    'relative_mass_above', massAbove, ...
    'jump_ratio_above_to_below', massAbove/massBelow);
output.results = results;
output.interpretation = { ...
    'Small raw-FD error certifies only the reconstructed local derivative on this fixture.', ...
    'A large filtered-vs-FD error means the sensitivity-filter vector is a heuristic search direction, not the exact gradient of the re-evaluated objective.', ...
    'The Yuksel production derivative is exact only for a load frozen within that iteration; its re-evaluated load changes with M(x).', ...
    'No result here certifies convergence, topology quality, or the unavailable final 240x30 design.'};

fid = fopen(resultPath, 'w');
if fid < 0
    error('PerformanceAudit:ResultWrite', 'Unable to create %s', resultPath);
end
cleanupFid = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(output, PrettyPrint=true));

fprintf('\nPerformance audit gradient checks completed.\n');
fprintf('Saved: %s\n', resultPath);
for d = 1:numel(designNames)
    r = results.(designNames{d});
    fprintf('  %-17s Our raw complete/full FD: %.3e; filtered/full FD: %.3e\n', ...
        designNames{d}, ...
        r.proposed_branch.raw_complete_vs_full_objective_fd.relative_l2_error, ...
        r.proposed_branch.production_filtered_complete_vs_full_objective_fd.relative_l2_error);
    fprintf('  %-17s Yuksel raw omitted/frozen FD: %.3e; omitted/full FD: %.3e\n', ...
        '', ...
        r.yuksel_stage2.production_raw_omitted_vs_frozen_load_fd.relative_l2_error, ...
        r.yuksel_stage2.production_raw_omitted_vs_full_objective_fd.relative_l2_error);
    fprintf('  %-17s Olhoff eigen chain/full FD: %.3e; Eb production/FD: %.3e\n', ...
        '', ...
        r.olhoff_branch.eigenvalue_gradient_through_filter_projection_vs_fd.relative_l2_error, ...
        r.olhoff_branch.eigen_constraint_Eb_derivative.production_relative_error);
end

% =========================================================================
% Model and finite-element helpers
% =========================================================================
function model = localBuildModel(nelx, nely, L, H)
model = struct();
model.nelx = nelx;
model.nely = nely;
model.nEl = nelx*nely;
model.L = L;
model.H = H;
model.hx = L/nelx;
model.hy = H/nely;
model.E0 = 1e7;
model.Emin = 1e-6*model.E0;
model.rho0 = 1;
model.rhoMin = 1e-6;
model.penal = 3;
model.KE = localLk(model.hx, model.hy, 0.3);
model.ME = localLm(model.hx, model.hy);
[model.edof, model.iK, model.jK] = localEdofData(nelx, nely);
model.nDof = 2*(nelx+1)*(nely+1);

leftNode = 1 + round(nely/2);
rightNode = nelx*(nely+1) + 1 + round(nely/2);
model.fixed = [2*leftNode-1; 2*leftNode; 2*rightNode-1; 2*rightNode];
model.free = setdiff((1:model.nDof)', model.fixed);

% Sparse truncated filter used by the proposed branch.
rminPhysical = 2*model.hx;
rows = [];
cols = [];
vals = [];
for ex1 = 1:nelx
    for ey1 = 1:nely
        e1 = (ex1-1)*nely+ey1;
        for ex2 = max(1,ex1-ceil(rminPhysical/model.hx)): ...
                min(nelx,ex1+ceil(rminPhysical/model.hx))
            for ey2 = max(1,ey1-ceil(rminPhysical/model.hy)): ...
                    min(nely,ey1+ceil(rminPhysical/model.hy))
                e2 = (ex2-1)*nely+ey2;
                distance = sqrt(((ex1-ex2)*model.hx)^2 + ((ey1-ey2)*model.hy)^2);
                weight = max(0, rminPhysical-distance);
                if weight > 0
                    rows(end+1,1) = e1; %#ok<AGROW>
                    cols(end+1,1) = e2; %#ok<AGROW>
                    vals(end+1,1) = weight; %#ok<AGROW>
                end
            end
        end
    end
end
model.Hfilter = sparse(rows, cols, vals, model.nEl, model.nEl);
model.HsparseSum = sum(model.Hfilter, 2);

% Symmetric image filter used by the Olhoff and Yuksel branches.
rminElem = 2;
[dy, dx] = meshgrid(-ceil(rminElem)+1:ceil(rminElem)-1);
model.hImage = max(0, rminElem-sqrt(dx.^2+dy.^2));
model.HimageSum = imfilter(ones(nely,nelx), model.hImage, 'symmetric');
end

function [objective, completeGradient, omittedGradient, F] = ...
        localOurObjectiveGradient(x, model, phi0, omega0Sq)
K = localAssembleStiffness(x, model);
M = localAssembleMass(x, model, 'linear');
F = omega0Sq*(M*phi0);
U = zeros(model.nDof,1);
U(model.free) = K(model.free,model.free)\F(model.free);
objective = U'*K*U;
[strainK, crossM] = localElementQuadratics(U, phi0, model);
dK = model.penal*(model.E0-model.Emin)*x.^(model.penal-1);
dM = (model.rho0-model.rhoMin)*ones(model.nEl,1);
omittedGradient = -dK.*strainK;
completeGradient = omittedGradient + 2*omega0Sq*dM.*crossM;
end

function value = localOurObjective(x, model, phi0, omega0Sq)
[value, ~, ~, ~] = localOurObjectiveGradient(x, model, phi0, omega0Sq);
end

function [objective, completeGradient, omittedGradient, F] = ...
        localYukselObjectiveGradient(x, model, uhat)
K = localAssembleStiffness(x, model);
M = localAssembleMass(x, model, 'yuksel');
F = M*uhat;
U = zeros(model.nDof,1);
U(model.free) = K(model.free,model.free)\F(model.free);
objective = U'*K*U;
[strainK, crossM] = localElementQuadratics(U, uhat, model);
dK = model.penal*(model.E0-model.Emin)*x.^(model.penal-1);
dMassLaw = ones(model.nEl,1);
low = x <= 0.1;
dMassLaw(low) = 6*x(low).^5;
dM = (model.rho0-model.rhoMin)*dMassLaw;
omittedGradient = -dK.*strainK;
completeGradient = omittedGradient + 2*dM.*crossM;
end

function value = localYukselObjective(x, model, uhat)
[value, ~, ~, ~] = localYukselObjectiveGradient(x, model, uhat);
end

function value = localFixedLoadObjective(x, model, F)
K = localAssembleStiffness(x, model);
U = zeros(model.nDof,1);
U(model.free) = K(model.free,model.free)\F(model.free);
value = U'*K*U;
end

function filtered = localOurSensitivityFilter(x, gradient, model)
filtered = (model.Hfilter*(x.*gradient))./model.HsparseSum./max(1e-3,x);
end

function filtered = localYukselSensitivityFilter(x, gradient, model)
xMat = reshape(max(1e-3,x),model.nely,model.nelx);
g = imfilter(reshape(x.*gradient,model.nely,model.nelx), ...
    model.hImage,'symmetric')./model.HimageSum./xMat;
filtered = g(:);
end

function [xPhys, dProjection] = localOlhoffPhysical(x, model, beta, eta)
xGrid = reshape(x,model.nely,model.nelx);
xTilde = imfilter(xGrid,model.hImage,'symmetric')./model.HimageSum;
denom = tanh(beta*eta)+tanh(beta*(1-eta));
xPhys = (tanh(beta*eta)+tanh(beta*(xTilde-eta)))/denom;
dProjection = beta*(1-tanh(beta*(xTilde-eta)).^2)/denom;
xPhys = xPhys(:);
dProjection = dProjection(:);
end

function designGradient = localOlhoffBackward(physicalGradientTimesProjection, model)
g = reshape(physicalGradientTimesProjection,model.nely,model.nelx);
designGradient = imfilter(g./model.HimageSum,model.hImage,'symmetric');
designGradient = designGradient(:);
end

function value = localOlhoffLambda(x, model, beta, eta)
xPhys = localOlhoffPhysical(x,model,beta,eta);
value = localFirstEigenpair(xPhys,model,'linear');
end

function value = localOlhoffGrayObjective(x, model, beta, eta, Eb, grayWeight)
xPhys = localOlhoffPhysical(x,model,beta,eta);
value = -Eb + grayWeight*mean(4*xPhys.*(1-xPhys));
end

function value = localOlhoffConstraint(x, model, beta, eta, Eb, lambdaRef)
lambda1 = localOlhoffLambda(x,model,beta,eta);
value = (Eb-lambda1/lambdaRef)/max(1,Eb);
end

function [lambda1, phi, omegaSq, dLambda] = localFirstEigenpair(x, model, massLaw)
K = localAssembleStiffness(x,model);
M = localAssembleMass(x,model,massLaw);
[V,D] = eig(full(K(model.free,model.free)),full(M(model.free,model.free)),'vector');
valid = isfinite(D) & isreal(D) & D > 0;
D = real(D(valid));
V = real(V(:,valid));
[D,order] = sort(D,'ascend');
V = V(:,order);
lambda1 = D(1);
omegaSq = lambda1;
phi = zeros(model.nDof,1);
phi(model.free) = V(:,1);
phi = phi/sqrt(phi'*M*phi);
if nargout >= 4
    phiEl = phi(model.edof);
    strain = sum((phiEl*model.KE).*phiEl,2);
    kinetic = sum((phiEl*model.ME).*phiEl,2);
    dK = model.penal*(model.E0-model.Emin)*x.^(model.penal-1);
    dM = (model.rho0-model.rhoMin)*ones(model.nEl,1);
    dLambda = dK.*strain-lambda1*dM.*kinetic;
end
end

function K = localAssembleStiffness(x,model)
coefficient = model.Emin+(model.E0-model.Emin)*x(:)'.^model.penal;
sK = reshape(model.KE(:)*coefficient,[],1);
K = sparse(model.iK,model.jK,sK,model.nDof,model.nDof);
K = (K+K')/2;
end

function M = localAssembleMass(x,model,massLaw)
switch massLaw
    case 'linear'
        relative = x(:)';
    case 'yuksel'
        relative = x(:)';
        low = relative <= 0.1;
        relative(low) = relative(low).^6;
    otherwise
        error('PerformanceAudit:MassLaw','Unknown mass law: %s',massLaw);
end
coefficient = model.rhoMin+(model.rho0-model.rhoMin)*relative;
sM = reshape(model.ME(:)*coefficient,[],1);
M = sparse(model.iK,model.jK,sM,model.nDof,model.nDof);
M = (M+M')/2;
end

function [strainK,crossM] = localElementQuadratics(U,V,model)
Ue = U(model.edof);
Ve = V(model.edof);
strainK = sum((Ue*model.KE).*Ue,2);
crossM = sum((Ue*model.ME).*Ve,2);
end

function fd = localFd(fun,x,indices,step)
fd = zeros(numel(indices),1);
for k = 1:numel(indices)
    xp = x;
    xm = x;
    xp(indices(k)) = xp(indices(k))+step;
    xm(indices(k)) = xm(indices(k))-step;
    fd(k) = (fun(xp)-fun(xm))/(2*step);
end
end

function comparison = localCompare(analytic,fd)
analytic = analytic(:);
fd = fd(:);
scale = max([norm(analytic),norm(fd),1e-14]);
pointScale = max(max(abs([analytic;fd])),1e-14);
pointError = abs(analytic-fd)./max(max(abs(analytic),abs(fd)),1e-8*pointScale);
comparison = struct( ...
    'relative_l2_error',norm(analytic-fd)/scale, ...
    'maximum_stabilized_pointwise_relative_error',max(pointError), ...
    'median_stabilized_pointwise_relative_error',median(pointError), ...
    'maximum_absolute_error',max(abs(analytic-fd)), ...
    'analytical_values',analytic, ...
    'finite_difference_values',fd);
end

function value = localScalarRelErr(a,b)
value = abs(a-b)/max([abs(a),abs(b),1e-14]);
end

function indices = localTestIndices(x,count)
candidates = find(x > 1e-3 & x < 1-1e-3 & abs(x-0.1) > 1e-3);
if numel(candidates) < count
    error('PerformanceAudit:InsufficientFDPoints', ...
        'Only %d interior finite-difference candidates.',numel(candidates));
end
positions = unique(round(linspace(1,numel(candidates),count)),'stable');
indices = candidates(positions);
end

function x = localMeanCorrect(x,target)
for k = 1:20
    delta = target-mean(x);
    x = min(0.9,max(0.1,x+delta));
    if abs(mean(x)-target) < 1e-14
        break
    end
end
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

function KE = localLk(hx,hy,nu)
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
        B(1,1:2:end) = dNxy(1,:);
        B(2,2:2:end) = dNxy(2,:);
        B(3,1:2:end) = dNxy(2,:);
        B(3,2:2:end) = dNxy(1,:);
        KE = KE+B'*D*B*detJ;
    end
end
end

function ME = localLm(hx,hy)
Ms = (hx*hy/36)*[4,2,1,2;2,4,2,1;1,2,4,2;2,1,2,4];
ME = kron(Ms,eye(2));
end
