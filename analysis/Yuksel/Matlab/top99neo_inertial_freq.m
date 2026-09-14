function [xPhys_stage2,U_stage2,info] = top99neo_inertial_freq(nelx,nely,volfrac,penal,rmin,ft,ftBC,eta,beta,move,maxit,stage1_maxit,bcType,nHistModes,runCfg)
solver_tic = tic;
%TOP99NEO_INERTIAL_FREQ  Fast surrogate frequency maximization via design-dependent inertial loads.
%
% Implements the two-stage method from Yuksel & Yilmaz (2025):
%   Stage 1: standard compliance minimization with a unit point load to obtain a
%            reasonable estimate of the fundamental mode shape.
%   Stage 2: repeat compliance minimization, but replace the external point load with a
%            design-dependent inertial load  F = M(x) * u_hat  that is updated each iteration.
%
% This function is designed to be a minimal patch on top of Ferrari & Sigmund's top99neo.m
% implementation style (fsparse assembly, sensitivity filter, OC update, projection, continuation).
%
% OUTPUTS
%   xPhys_stage2 : final physical density field from stage 2
%   U_stage2     : final displacement/mode-shape estimate from stage 2
%   info         : struct with stage-1 and stage-2 histories (including omega1)
%
% INPUT NOTES
%   - bcType controls boundary conditions and the stage-1 point load position.
%     Supported values (strings):
%       "cantilever"   : left edge fixed, load at mid-height of right edge (down)
%                        + concentrated tip mass at the same node (Figure 9 setup)
%       "simply"       : hinges at mid-height (h/2) on both edges, load at beam center (down)
%       "fixedPinned"  : left edge fixed, right end pinned (approx.), stage-1 load chosen
%                        by an eigenmode of the fully-solid beam to locate max deflection.
%
%   - For stage 2, the inertial load is recomputed each iteration as F = M(x)*u_hat.
%     Sensitivities are computed as if F were fixed within that iteration (i.e. dF/dx ignored).
%   - nHistModes (optional): when > 0, stores per-iteration design histories and computes
%     first nHistModes natural-frequency histories (for Figure-6-style plots). This is expensive.

if nargin < 1  || isempty(nelx),         nelx = 300;   end
if nargin < 2  || isempty(nely),         nely = 100;   end
if nargin < 3  || isempty(volfrac),      volfrac = 0.5; end
if nargin < 4  || isempty(penal),        penal = 3.0;   end
if nargin < 5  || isempty(rmin),         rmin = 8.75;   end
if nargin < 6  || isempty(ft),           ft = 1;        end
if nargin < 7  || isempty(ftBC),         ftBC = 'N';    end
if nargin < 8  || isempty(eta),          eta = 0.5;     end
if nargin < 9  || isempty(beta),         beta = 1.0;    end
if nargin < 10 || isempty(move),         move = 0.2;    end
if nargin < 11 || isempty(maxit),        maxit = 100;   end
if nargin < 12 || isempty(stage1_maxit), stage1_maxit = maxit; end
if nargin < 13 || isempty(bcType),       bcType = "simply"; end
if nargin < 14 || isempty(nHistModes),   nHistModes = 0; end
if nargin < 15 || isempty(runCfg),       runCfg = struct(); end
if ~isstruct(runCfg)
    error('runCfg must be a struct when provided.');
end
localEnsurePlotHelpersOnPath();
bcType = string(bcType);
nHistModes = max(0, floor(double(nHistModes)));
stage1Tol = 1e-2;
stage2Tol = 1e-2;
if strcmpi(char(bcType), 'fixedpinned'), stage2Tol = 1e-2; end
if isfield(runCfg, 'conv_tol') && ~isempty(runCfg.conv_tol)
    stage2Tol = runCfg.conv_tol;
end
if isfield(runCfg, 'stage1_tol') && ~isempty(runCfg.stage1_tol), stage1Tol = runCfg.stage1_tol; end
if isfield(runCfg, 'stage2_tol') && ~isempty(runCfg.stage2_tol), stage2Tol = runCfg.stage2_tol; end
finalModes = max(1, floor(double(localOpt(runCfg, 'final_modes', 3))));
if isfield(runCfg, 'visualize_live') && ~isempty(runCfg.visualize_live)
    doPlot = localParseVisualizeLive(runCfg.visualize_live, true);
else
    doPlot = true;
end
visualizationQuality = localParseVisualizationQuality( ...
    localOpt(runCfg, 'visualization_quality', 'regular'));
saveFrqIterations = localParseVisualizeLive(localOpt(runCfg, 'save_frq_iterations', false), false);
if saveFrqIterations
    fprintf(['Warning: save_frq_iterations=yes forces per-iteration eigenvalue solves for plotting; ', ...
        'runtime will increase and comparisons are not fair.\n']);
end
approachName = localApproachName(runCfg, 'Yuksel');

%% ---------------------------- PRE. 1) MATERIAL AND CONTINUATION PARAMETERS
E0 = localOpt(runCfg, 'E0', 1e7);
Emin = localOpt(runCfg, 'Emin', 1e-9 * E0);
nu = localOpt(runCfg, 'nu', 0.3);

rho0 = localOpt(runCfg, 'rho0', 1.0);
rho_min = localOpt(runCfg, 'rho_min', 1e-9 * rho0);
dMass = localOpt(runCfg, 'dMass', 6.0);
xMassCut = localOpt(runCfg, 'xMassCut', 0.1);

penalCnt = { 1,  1, 25, 0.25 };
betaCnt  = { 1,  1, 25,    2 };
if strcmpi(char(ftBC), 'N'), bcF = 'symmetric'; else, bcF = 0; end

%% ----------------------------------------- PRE. 2) DISCRETIZATION FEATURES
[beamL, beamH, tipMassFrac] = localPhysicalSetup(bcType, runCfg);
nEl = nelx * nely;
nodeNrs = int32( reshape( 1 : (1 + nelx) * (1 + nely), 1+nely, 1+nelx ) );
cVec = reshape( 2 * nodeNrs( 1 : end - 1, 1 : end - 1 ) + 1, nEl, 1 );
cMat = cVec + int32( [ 0, 1, 2 * nely + [ 2, 3, 0, 1 ], -2, -1 ] );
nDof = ( 1 + nely ) * ( 1 + nelx ) * 2;

[ sI, sII ] = deal( [ ] );
for j = 1 : 8
    sI  = cat( 2, sI,  j : 8 );
    sII = cat( 2, sII, repmat( j, 1, 8 - j + 1 ) );
end
[ iK , jK ] = deal( cMat( :,  sI )', cMat( :, sII )' );
Iar = sort( [ iK( : ), jK( : ) ], 2, 'descend' ); clear iK jK

% --- element stiffness (plane stress Q4, as in top99neo)
c1 = [12;3;-6;-3;-6;-3;0;3;12;3;0;-3;-6;-3;-6;12;-3;0;-3;-6;3;12;3;...
    -6;3;-6;12;3;-6;-3;12;3;0;12;-3;12];
c2 = [-4;3;-2;9;2;-3;4;-9;-4;-9;4;-3;2;9;-2;-4;-3;4;9;2;3;-4;-9;-2;...
    3;2;-4;3;-2;9;-4;-9;4;-4;-3;-4];
Ke = 1/(1-nu^2)/24*( c1 + nu .* c2 );
Ke0( tril( ones( 8 ) ) == 1 ) = Ke';
Ke0 = reshape( Ke0, 8, 8 );
Ke0 = Ke0 + Ke0' - diag( diag( Ke0 ) );

% --- element consistent mass (paper geometry scaling)
% Match the physical beam dimensions reported for each benchmark case.
elemArea = (beamL / nelx) * (beamH / nely);
% Scalar (4x4) consistent mass: A/36 * [4 2 1 2; 2 4 2 1; 1 2 4 2; 2 1 2 4]
MeS = (elemArea/36) * [4 2 1 2; 2 4 2 1; 1 2 4 2; 2 1 2 4;];
% Expand to 8x8 for 2 dof/node ordering [u1 v1 u2 v2 u3 v3 u4 v4]
Me0 = kron(MeS, eye(2));

%% ----------------------------- PRE. 3) LOADS, SUPPORTS AND PASSIVE DOMAINS
[pasS, pasV] = deal([],[]);
if isfield(runCfg, 'pasS') && ~isempty(runCfg.pasS), pasS = runCfg.pasS(:); end
if isfield(runCfg, 'pasV') && ~isempty(runCfg.pasV), pasV = runCfg.pasV(:); end

% Boundary conditions + stage-1 point load
[fixed, lcDof, tipMassNode] = localBCAndLoad(nodeNrs,nely,nelx,nDof,bcType);
if isfield(runCfg, 'extraFixedDofs') && ~isempty(runCfg.extraFixedDofs)
    fixed = unique([fixed(:); double(runCfg.extraFixedDofs(:))]);
end
if strcmpi(char(bcType), 'fixedpinned')
    % Figure 8 setup: locate load node from first mode of fully-solid fixed-pinned beam.
    lcDof = localFixedPinnedLoadFromSolidMode( ...
        fixed, nodeNrs, nEl, nDof, Iar, Ke, Me0, E0, Emin, penal, ...
        rho0, rho_min, dMass, xMassCut);
end
F_point = localAssemble( lcDof', 1, -1, [ nDof, 1 ] );

% Concentrated non-design mass (Figure 9 cantilever case).
tipMassDofs = [];
tipMassVal = 0;
if tipMassFrac > 0 && ~isempty(tipMassNode)
    permittedMass = volfrac * beamL * beamH * rho0;
    tipMassVal = tipMassFrac * permittedMass;
    tipMassDofs = [2*tipMassNode-1, 2*tipMassNode];
end

free = setdiff( 1 : nDof, fixed );
act  = setdiff( (1 : nEl )', union( pasS, pasV ) );

%% --------------------------------------- PRE. 4) DEFINE IMPLICIT FUNCTIONS
prj  = @(v,eta,beta) (tanh(beta*eta)+tanh(beta*(v(:)-eta)))./(tanh(beta*eta)+tanh(beta*(1-eta)));
deta = @(v,eta,beta) - beta * csch( beta ) .* sech( beta * ( v( : ) - eta ) ).^2 .* ...
    sinh( v( : ) * beta ) .* sinh( ( 1 - v( : ) ) * beta );
dprj = @(v,eta,beta) beta*(1-tanh(beta*(v-eta)).^2)./(tanh(beta*eta)+tanh(beta*(1-eta)));
cnt  = @(v,vCnt,l) v+(l>=vCnt{1})*(v<vCnt{2})*(mod(l,vCnt{3})==0)*vCnt{4};

%% ------------------------------------------------- PRE. 5) PREPARE FILTER
[dy,dx] = meshgrid(-ceil(rmin)+1:ceil(rmin)-1,-ceil(rmin)+1:ceil(rmin)-1);
h  = max( 0, rmin - sqrt( dx.^2 + dy.^2 ) );
Hs = imfilter( ones( nely, nelx ), h, bcF );

%% ------------------------ PRE. 6) ALLOCATE AND INITIALIZE OTHER PARAMETERS
[x, dsK, dV] = deal( zeros( nEl, 1 ) );
dV( act, 1 ) = 1/nEl/volfrac;
x( act ) = ( volfrac*( nEl - length(pasV) ) - length(pasS) )/length( act );
x( pasS ) = 1;

info = struct();
info.stage1 = struct('c',[],'v',[],'ch',[],'xHist',[],'omegaHist',[],'omegaFinal',[], ...
    'freq_iter_omega', []);
info.stage2 = struct('c',[],'v',[],'ch',[],'xHist',[],'omegaHist',[],'omegaFinal',[], ...
    'freq_iter_omega', []);
% Opt-in scientific-audit logging.  These fields do not participate in the
% optimization update and are disabled by default.
auditCollect = logical(localOpt(runCfg, 'audit_collect', false));
auditSnapshotEvery = max(1, floor(double(localOpt(runCfg, 'audit_snapshot_every', 10))));
info.stage1.audit_collect = auditCollect;
info.stage1.audit_snapshot_every = auditSnapshotEvery;
info.stage2.audit_collect = auditCollect;
info.stage2.audit_snapshot_every = auditSnapshotEvery;
% Diagnostic ablations are opt-in and false by default. They are used only
% by the independent audit runner and are reported separately from baseline.
info.stage2.audit_freeze_mode = logical(localOpt(runCfg, 'audit_freeze_mode', false));
info.stage2.audit_freeze_load = logical(localOpt(runCfg, 'audit_freeze_load', false));
if saveFrqIterations
    info.stage1.freq_iter_omega = NaN(stage1_maxit, 3);
    info.stage2.freq_iter_omega = NaN(maxit, 3);
end
info.stage1.loadDof = lcDof;

% Opt-in iteration history (plan section 5).  Seeded here rather than inside
% the loops so both stages share one schema; each loop records into the
% recorder it finds on its stageInfo.
% Extension mode applies to stage 2 only; stage 1's convergence test is the
% handoff into stage 2 and stays active (plan section 4.3).
info.stage2.extend_beyond_native_stop = ...
    logical(localOpt(runCfg, 'extend_beyond_native_stop', false));

recordHistory = logical(localOpt(runCfg, 'record_history', false));
if recordHistory
    histMeta = struct( ...
        'method', 'Yuksel', ...
        'objective_definition', ['compliance F''U of the current stage; ' ...
            'MINIMIZED. Stage 1 uses the unit point load, stage 2 the ' ...
            'design-dependent inertial load, so the two stages do not share ' ...
            'a common objective scale.'], ...
        'objective_sign', -1, ...
        'volfrac', volfrac);
    histMeta.stage = 1;
    info.stage1.history = topopt_history_init(stage1_maxit, histMeta);
    histMeta.stage = 2;
    info.stage2.history = topopt_history_init(maxit, histMeta);
end

%% ================================ STAGE 1: standard compliance minimization
[xPhys,U] = deal(x, zeros(nDof,1));
initialization_time = toc(solver_tic);
[xPhys,U,eta,penal,beta,info.stage1] = localComplianceLoop( ...
    x, xPhys, U, F_point, fixed, free, act, ...
    nelx, nely, nEl, nDof, cMat, Iar, Ke, Ke0, ...
    E0, Emin, penal, rmin, h, Hs, bcF, ft, eta, beta, move, stage1_maxit, ...
    penalCnt, betaCnt, dsK, dV, info.stage1, doPlot, nHistModes, stage1Tol, ...
    approachName, volfrac, saveFrqIterations, Me0, rho0, rho_min, dMass, ...
    xMassCut, tipMassDofs, tipMassVal, pasS, pasV);
info.stage1.xFinal = xPhys;
info.stage1.UFinal = U;
info.stage1.omega1 = localFirstOmega( ...
    xPhys, free, nEl, nDof, Iar, Ke, Me0, E0, Emin, penal, ...
    rho0, rho_min, dMass, xMassCut, tipMassDofs, tipMassVal);

% Use stage-1 outputs as initial guesses for stage 2
x = xPhys;
U_est = U;

%% ================================ STAGE 2: inertial-load compliance loop
% Reset continuation counters if desired (paper keeps p=3 constant in static parts; here we keep user input)
[xPhys_stage2,U_stage2] = deal(xPhys, U);
if recordHistory
    info.stage2.history_xphys_prev = xPhys;
end

[xPhys_stage2,U_stage2,eta,penal,beta,info.stage2] = localInertialLoop( ...
    x, xPhys_stage2, U_est, fixed, free, act, ...
    nelx, nely, nEl, nDof, cMat, Iar, Ke, Ke0, Me0, ...
    E0, Emin, rho0, rho_min, dMass, xMassCut, ...
    tipMassDofs, tipMassVal, ...
    penal, rmin, h, Hs, bcF, ft, eta, beta, move, maxit, ...
    penalCnt, betaCnt, dsK, dV, info.stage2, doPlot, stage2Tol, nHistModes, ...
    approachName, volfrac, saveFrqIterations, pasS, pasV);
info.stage2.xFinal = xPhys_stage2;
info.stage2.UFinal = U_stage2;
info.stage2.omegaFinal = localFirstNOmegas( ...
    xPhys_stage2, free, nEl, nDof, Iar, Ke, Me0, E0, Emin, penal, ...
    rho0, rho_min, dMass, xMassCut, tipMassDofs, tipMassVal, finalModes);
info.stage2.omega1 = info.stage2.omegaFinal(1);
if recordHistory
    % Join the stages into one globally numbered history and mark the handoff.
    % This method has no active continuation -- penalCnt and betaCnt both gate
    % on v < 1 while penal = 3 and beta = 1 -- so the handoff is its only
    % transition and therefore its k_cont.
    info.history = topopt_history_concat(info.stage1.history, info.stage2.history);
    info.history = topopt_history_mark(info.history, info.stage1.iterations, ...
        'stage', 'stage 1 -> stage 2 (inertial load)');
    info.history.k_cont = info.stage1.iterations;
end
if nHistModes > 0
    info.stage1.omegaHist = localModeHistory( ...
        info.stage1.xHist, free, nEl, nDof, Iar, Ke, Me0, E0, Emin, penal, ...
        rho0, rho_min, dMass, xMassCut, tipMassDofs, tipMassVal, nHistModes);
    info.stage2.omegaHist = localModeHistory( ...
        info.stage2.xHist, free, nEl, nDof, Iar, Ke, Me0, E0, Emin, penal, ...
        rho0, rho_min, dMass, xMassCut, tipMassDofs, tipMassVal, nHistModes);
end
if saveFrqIterations
    h1 = info.stage1.freq_iter_omega;
    h2 = info.stage2.freq_iter_omega;
    info.freq_iter_omega = [h1; h2];
end

if isfinite(info.stage2.omega1)
    fprintf('\nFinal design: omega1 = %.4f rad/s\n', info.stage2.omega1);
end
omega2_stage2 = NaN;
if numel(info.stage2.omegaFinal) >= 2, omega2_stage2 = info.stage2.omegaFinal(2); end
plotTopology( ...
    xPhys_stage2, nelx, nely, ...
    formatTopologyTitle(approachName, volfrac, info.stage2.omega1, omega2_stage2), ...
    doPlot, visualizationQuality, true);

info.timing = struct();
info.timing.stage1_loop_time = localOpt(info.stage1, 'loop_time', NaN);
info.timing.stage2_loop_time = localOpt(info.stage2, 'loop_time', NaN);
info.timing.stage1_iterations = localOpt(info.stage1, 'iterations', NaN);
info.timing.stage2_iterations = localOpt(info.stage2, 'iterations', NaN);
info.timing.total_loop_time = info.timing.stage1_loop_time + info.timing.stage2_loop_time;
info.timing.total_iterations = info.timing.stage1_iterations + info.timing.stage2_iterations;
info.timing.t_iter = info.timing.total_loop_time / max(info.timing.total_iterations, 1);
info.timing.initialization_time = initialization_time;
solver_total_time = toc(solver_tic);
info.timing.postprocessing_time = max(0, solver_total_time - ...
    info.timing.initialization_time - info.timing.total_loop_time);
info.timing.total_time = solver_total_time;
info.stopping = struct( ...
    'stop_reason', localOpt(info.stage2, 'stop_reason', 'N/A'), ...
    'stage1_stop_reason', localOpt(info.stage1, 'stop_reason', 'N/A'), ...
    'stage2_stop_reason', localOpt(info.stage2, 'stop_reason', 'N/A'), ...
    'final_max_density_change', localLast(info.stage2.ch), ...
    'final_rms_density_change', localLast(info.stage2.rms_ch), ...
    'final_relative_objective_change', localRelativeLast(info.stage2.c), ...
    'final_grayness', mean(4*xPhys_stage2.*(1-xPhys_stage2)), ...
    'convergence_tolerance', stage2Tol, ...
    'stage1_tolerance', stage1Tol);

end

%% =======================================================================
function [beamL, beamH, tipMassFrac] = localPhysicalSetup(bcType, runCfg)
% Physical dimensions and concentrated-mass setup used in paper benchmarks.
switch lower(bcType)
    case "cantilever"
        beamL = 15;
        beamH = 10;
        tipMassFrac = 0.20; % 20% of total permitted material mass
    case {"simply","fixedpinned"}
        beamL = 8;
        beamH = 1;
        tipMassFrac = 0;
    case "none"
        % All BCs come from extraFixedDofs; dimensions must be set via runCfg.
        beamL = 0;
        beamH = 0;
        tipMassFrac = 0;
    otherwise
        error('Unsupported bcType: %s', bcType);
end
if isfield(runCfg, 'beamL') && ~isempty(runCfg.beamL), beamL = runCfg.beamL; end
if isfield(runCfg, 'beamH') && ~isempty(runCfg.beamH), beamH = runCfg.beamH; end
if isfield(runCfg, 'tipMassFrac') && ~isempty(runCfg.tipMassFrac), tipMassFrac = runCfg.tipMassFrac; end
end

%% =======================================================================
function [fixed, lcDof, tipMassNode] = localBCAndLoad(nodeNrs,nely,nelx,nDof,bcType)
% Returns fixed dofs + the dof for the stage-1 unit point load.
tipMassNode = [];

switch lower(bcType)
    case "simply"
        % Hinged supports at mid-height (neutral axis) on left/right boundaries.
        % For odd nely this picks the nearest node to h/2.
        midRow = round(nely/2) + 1;
        leftMid = nodeNrs(midRow, 1);
        rightMid = nodeNrs(midRow, end);
        fixed = [2*leftMid-1, 2*leftMid, 2*rightMid-1, 2*rightMid];

        % Stage-1 point load: downward at the middle of the beam (paper Figure 4a).
        midCol = round((nelx+1)/2);
        lcNode = nodeNrs(midRow, midCol);
        lcDof = 2*lcNode; % vertical dof

    case "cantilever"
        % Fix left edge (both u,v)
        leftNodes = nodeNrs(:,1);
        fixed = union(2*leftNodes-1, 2*leftNodes);
        % Load at middle of right edge (vertical)
        midRow = round((nely+1)/2);
        lcNode = nodeNrs(midRow, end);
        lcDof = 2*lcNode;
        tipMassNode = lcNode;

    case "fixedpinned"
        % Fixed at left edge, pinned support at right edge mid-height (h/2).
        leftNodes = nodeNrs(:,1);
        fixed = union(2*leftNodes-1, 2*leftNodes);
        midRow = round(nely/2) + 1;
        rightMid = nodeNrs(midRow, end);
        fixed = union(fixed, [2*rightMid-1, 2*rightMid]);
        % Temporary: choose mid of beam for load; caller may override via eigenmode search.
        midCol = round((nelx+1)/2);
        lcNode = nodeNrs(midRow, midCol);
        lcDof = 2*lcNode;

    case "none"
        % No standard hinge/clamp — all fixed DOFs come from extraFixedDofs.
        fixed = [];
        tipMassNode = [];
        % Stage-1 load: vertical DOF at beam center (sensible fallback).
        midRow = round(nely/2) + 1;
        midCol = round((nelx+1)/2);
        lcNode = nodeNrs(midRow, midCol);
        lcDof = 2*lcNode;

    otherwise
        error('Unsupported bcType: %s', bcType);
end

fixed = unique(fixed(:))';
end

%% =======================================================================
function lcDof = localFixedPinnedLoadFromSolidMode( ...
    fixed, nodeNrs, nEl, nDof, Iar, Ke, Me0, E0, Emin, penal, ...
    rho0, rho_min, dMass, xMassCut)
% Determine stage-1 load location from the first eigenmode of a fully solid beam.
% This matches the fixed-pinned setup reported for Figure 8.
lcDof = [];

try
    xSolid = ones(nEl,1);

    % Fully solid stiffness
    sK = (Emin + xSolid.^penal * (E0 - Emin));
    sK = reshape(Ke(:) * sK', length(Ke) * nEl, 1);
    K = localAssemble(Iar(:,1), Iar(:,2), sK, [nDof, nDof]);
    K = K + K' - spdiags(diag(K), 0, nDof, nDof);

    % Fully solid mass
    rhoe = rho_min + (rho0 - rho_min) * xSolid;
    low = xSolid <= xMassCut;
    rhoe(low) = rho_min + (rho0 - rho_min) * (xSolid(low).^dMass);
    ltMask = tril(true(8));
    meLower = Me0(ltMask);
    sM = reshape(meLower * rhoe', nnz(ltMask) * nEl, 1);
    M = localAssemble(Iar(:,1), Iar(:,2), sM, [nDof, nDof]);
    M = M + M' - spdiags(diag(M), 0, nDof, nDof);

    free = setdiff(1:nDof, fixed);
    eigOpts = struct('disp', 0, 'maxit', 1000, ...
        'v0', localDeterministicEigsStartVector(numel(free)));
    [phi, ~] = eigs(K(free,free), M(free,free), 1, 'sm', eigOpts);
    U = zeros(nDof,1);
    U(free) = real(phi(:,1));

    % Select the node with maximum vertical deflection in mode 1.
    allNodes = double(nodeNrs(:));
    vDofs = 2 * allNodes;
    freeVDofs = vDofs(~ismember(vDofs, fixed));
    [~, idx] = max(abs(U(freeVDofs)));
    lcDof = freeVDofs(idx);
catch
    lcDof = [];
end

if isempty(lcDof)
    % Robust fallback if eigs fails for any reason.
    nely = size(nodeNrs,1) - 1;
    midRow = round(nely/2) + 1;
    midCol = round(size(nodeNrs,2)/2);
    lcDof = 2 * nodeNrs(midRow, midCol);
end

lcDof = double(lcDof);
end

%% =======================================================================
function [xPhys,U,eta,penal,beta,stageInfo] = localComplianceLoop( ...
    x, xPhys, U, F, fixed, free, act, ...
    nelx, nely, nEl, nDof, cMat, Iar, Ke, Ke0, ...
    E0, Emin, penal, rmin, h, Hs, bcF, ft, eta, beta, move, maxit, ...
    penalCnt, betaCnt, dsK, dV, stageInfo, doPlot, nHistModes, tolX, ...
    approachName, volfrac, saveFrqIterations, Me0, rho0, rho_min, dMass, ...
    xMassCut, tipMassDofs, tipMassVal, pasS, pasV)

% ---- implicit functions (redefined here: local functions cannot access parent workspace)
prj  = @(v,eta_,beta_) (tanh(beta_*eta_)+tanh(beta_*(v(:)-eta_)))./(tanh(beta_*eta_)+tanh(beta_*(1-eta_)));
deta = @(v,eta_,beta_) - beta_ * csch( beta_ ) .* sech( beta_ * ( v( : ) - eta_ ) ).^2 .* ...
    sinh( v( : ) * beta_ ) .* sinh( ( 1 - v( : ) ) * beta_ );
dprj = @(v,eta_,beta_) beta_*(1-tanh(beta_*(v-eta_)).^2)./(tanh(beta_*eta_)+tanh(beta_*(1-eta_)));
cnt  = @(v,vCnt,l) v+(l>=vCnt{1})*(v<vCnt{2})*(mod(l,vCnt{3})==0)*vCnt{4};

loop = 0;
loop_tic = tic;
histStage = 1;
recordHistory = isfield(stageInfo, 'history') && ~isempty(stageInfo.history);
xPhysPrevHist = [];
stageInfo.rms_ch = [];
auditCollect = isfield(stageInfo, 'audit_collect') && stageInfo.audit_collect;
auditSnapshotEvery = localOpt(stageInfo, 'audit_snapshot_every', 10);
while loop < maxit
    loop = loop + 1;
    % ---- physical density
    if ft == 1
        % Sensitivity filter (Andreassen et al., 2011): no density filtering
        xPhys( act ) = x( act );
    else
        % Density filter (ft=2,3)
        xTilde = imfilter( reshape( x, nely, nelx ), h, bcF ) ./ Hs;
        xPhys( act ) = xTilde( act );
    end
    dHs = Hs;
    if ft > 1
        if ft == 3
            f = ( mean( prj( xPhys, eta, beta ) ) - mean(xPhys) );
            while abs(f) > 1e-6
                eta = eta - f / mean( deta( xPhys(:), eta, beta ) );
                f = mean( prj( xPhys, eta, beta ) ) - mean(xPhys);
            end
        end
        dHs = Hs ./ reshape( dprj( xTilde, eta, beta ), nely, nelx );
        xPhys = prj( xPhys, eta, beta );
    end
    % ---- Enforce passive element densities
    if ~isempty(pasS), xPhys(pasS) = 1; end
    if ~isempty(pasV), xPhys(pasV) = 0; end
    % ---- FE solve
    sK = ( Emin + xPhys.^penal * ( E0 - Emin ) );
    dsK( act ) = -penal * ( E0 - Emin ) * xPhys( act ) .^ ( penal - 1 );
    sK = reshape( Ke( : ) * sK', length( Ke ) * nEl, 1 );
    K = localAssemble( Iar( :, 1 ), Iar( :, 2 ), sK, [ nDof, nDof ] );
    U(:) = 0;
    U( free ) = decomposition( K( free, free ), 'chol','lower' ) \ F( free );

    % ---- sensitivities (standard compliance)
    dc = dsK .* sum( ( U( cMat ) * Ke0 ) .* U( cMat ), 2 );
    if ft == 1
        % Sensitivity filter (Andreassen et al., 2011)
        xMat = reshape( max( 1e-3, x ), nely, nelx );
        dc = imfilter( reshape( x .* dc, nely, nelx ), h, bcF ) ./ Hs ./ xMat;
        dV0 = imfilter( reshape( x .* dV, nely, nelx ), h, bcF ) ./ Hs ./ xMat;
    else
        % Chain-rule for density filter
        dc = imfilter( reshape( dc, nely, nelx ) ./ dHs, h, bcF );
        dV0 = imfilter( reshape( dV, nely, nelx ) ./ dHs, h, bcF );
    end

    % ---- OC update (robust bisection bracket + finite guards)
    xOldAudit = x;
    [x, ch, lambdaOC] = localOcUpdate(x, act, dc, dV0, move, mean(xPhys));
    rmsCh = sqrt(mean((x - xOldAudit).^2));

    penalLog = penal;
    [penal,beta] = deal(cnt(penal,penalCnt,loop), cnt(beta,betaCnt,loop));

    cVal = full(F' * U);
    stageInfo.c(end+1,1)  = cVal;
    stageInfo.v(end+1,1)  = mean(xPhys);
    stageInfo.ch(end+1,1) = ch;
    stageInfo.rms_ch(end+1,1) = rmsCh;
    if recordHistory
        % omega1 stays NaN: this method eigensolves only at the end, and
        % plan section 5 forbids adding a solve merely to fill a column.
        histRec = struct('iter', loop, 'stage', histStage, 'stage_iter', loop, ...
            'xPhys', xPhys, 'volfrac', volfrac, ...
            'objective', cVal, 'elapsed_s', toc(loop_tic), ...
            'x', x, 'xOld', xOldAudit, 'move_limit', move);
        if ~isempty(xPhysPrevHist)
            histRec.xPhysPrev = xPhysPrevHist;
        end
        stageInfo.history = topopt_history_record(stageInfo.history, histRec);
        xPhysPrevHist = xPhys;
    end
    if auditCollect
        [stageInfo, ~] = localRecordAuditStep( ...
            stageInfo, loop, xOldAudit, x, xPhys, act, dc, dV0, ...
            lambdaOC, move, NaN, NaN, NaN, auditSnapshotEvery);
    end
    if saveFrqIterations
        stageInfo.freq_iter_omega(loop,:) = localFirstNOmegas( ...
            xPhys, free, nEl, nDof, Iar, Ke, Me0, E0, Emin, penalLog, ...
            rho0, rho_min, dMass, xMassCut, tipMassDofs, tipMassVal, 3);
    end
    if nHistModes > 0
        stageInfo.xHist(:,end+1) = xPhys;
    end

    if doPlot
        fprintf('S1 It.:%5i C:%10.4e V:%7.3f ch:%0.2e penal:%5.2f beta:%5.1f eta:%6.3f\n', ...
            loop, cVal, mean(xPhys), ch, penal, beta, eta);
        plotTopology( ...
            xPhys, nelx, nely, ...
            formatTopologyTitle(approachName, volfrac, NaN), ...
            true, 'regular', false);
    end
    if loop > 1 && ch < tolX, break; end
end
stageInfo.iterations = loop;
stageInfo.loop_time = toc(loop_tic);
stageInfo.t_iter = stageInfo.loop_time / max(loop, 1);
if loop > 1 && ch < tolX
    stageInfo.stop_reason = 'density_change_tolerance';
else
    stageInfo.stop_reason = 'max_iterations';
end
if recordHistory
    stageInfo.history = topopt_history_finish(stageInfo.history);
end
if saveFrqIterations && isfield(stageInfo, 'freq_iter_omega') && ~isempty(stageInfo.freq_iter_omega)
    stageInfo.freq_iter_omega = stageInfo.freq_iter_omega(1:loop,:);
end

end

%% =======================================================================
function [xPhys,U,eta,penal,beta,stageInfo] = localInertialLoop( ...
    x, xPhys, U_est, fixed, free, act, ...
    nelx, nely, nEl, nDof, cMat, Iar, Ke, Ke0, Me0, ...
    E0, Emin, rho0, rho_min, dMass, xMassCut, ...
    tipMassDofs, tipMassVal, ...
    penal, rmin, h, Hs, bcF, ft, eta, beta, move, maxit, ...
    penalCnt, betaCnt, dsK, dV, stageInfo, doPlot, stage2Tol, nHistModes, ...
    approachName, volfrac, saveFrqIterations, pasS, pasV)

% ---- implicit functions (redefined here: local functions cannot access parent workspace)
prj  = @(v,eta_,beta_) (tanh(beta_*eta_)+tanh(beta_*(v(:)-eta_)))./(tanh(beta_*eta_)+tanh(beta_*(1-eta_)));
deta = @(v,eta_,beta_) - beta_ * csch( beta_ ) .* sech( beta_ * ( v( : ) - eta_ ) ).^2 .* ...
    sinh( v( : ) * beta_ ) .* sinh( ( 1 - v( : ) ) * beta_ );
dprj = @(v,eta_,beta_) beta_*(1-tanh(beta_*(v-eta_)).^2)./(tanh(beta_*eta_)+tanh(beta_*(1-eta_)));
cnt  = @(v,vCnt,l) v+(l>=vCnt{1})*(v<vCnt{2})*(mod(l,vCnt{3})==0)*vCnt{4};

tolX = stage2Tol;
loop = 0; U = U_est;
loop_tic = tic;
histStage = 2;
extendBeyondNativeStop = isfield(stageInfo, 'extend_beyond_native_stop') ...
    && logical(stageInfo.extend_beyond_native_stop);
stageInfo.native_stop_iter = NaN;
stageInfo.xphys_at_native_stop = [];
recordHistory = isfield(stageInfo, 'history') && ~isempty(stageInfo.history);
% Seed with stage 1's final field so the first stage-2 increment is defined.
% Without it the concatenated history carries a NaN d_inf at the handoff, which
% a persistence-window acceptance test would have to special-case.
xPhysPrevHist = [];
if isfield(stageInfo, 'history_xphys_prev')
    xPhysPrevHist = stageInfo.history_xphys_prev;
end
stageInfo.rms_ch = [];
auditCollect = isfield(stageInfo, 'audit_collect') && stageInfo.audit_collect;
auditSnapshotEvery = localOpt(stageInfo, 'audit_snapshot_every', 10);
auditFreezeMode = isfield(stageInfo, 'audit_freeze_mode') && stageInfo.audit_freeze_mode;
auditFreezeLoad = isfield(stageInfo, 'audit_freeze_load') && stageInfo.audit_freeze_load;
frozenModeEstimate = U_est;
frozenLoad = [];

while loop < maxit
    loop = loop + 1;

    % ---- physical density
    if ft == 1
        % Sensitivity filter (Andreassen et al., 2011): no density filtering
        xPhys( act ) = x( act );
    else
        % Density filter (ft=2,3)
        xTilde = imfilter( reshape( x, nely, nelx ), h, bcF ) ./ Hs;
        xPhys( act ) = xTilde( act );
    end
    dHs = Hs;
    if ft > 1
        if ft == 3
            f = ( mean( prj( xPhys, eta, beta ) ) - mean(xPhys) );
            while abs(f) > 1e-6
                eta = eta - f / mean( deta( xPhys(:), eta, beta ) );
                f = mean( prj( xPhys, eta, beta ) ) - mean(xPhys);
            end
        end
        dHs = Hs ./ reshape( dprj( xTilde, eta, beta ), nely, nelx );
        xPhys = prj( xPhys, eta, beta );
    end
    % ---- Enforce passive element densities
    if ~isempty(pasS), xPhys(pasS) = 1; end
    if ~isempty(pasV), xPhys(pasV) = 0; end

    % ---- assemble stiffness
    sK = ( Emin + xPhys.^penal * ( E0 - Emin ) );
    dsK( act ) = -penal * ( E0 - Emin ) * xPhys( act ) .^ ( penal - 1 );
    sK = reshape( Ke( : ) * sK', length( Ke ) * nEl, 1 );
    K = localAssemble( Iar( :, 1 ), Iar( :, 2 ), sK, [ nDof, nDof ] );

    % ---- assemble mass matrix (design dependent)
    % Modified SIMP-like rho(x): linear above xMassCut, x^d below
    rhoe = rho_min + (rho0-rho_min) * xPhys;
    low = xPhys <= xMassCut;
    rhoe(low) = rho_min + (rho0-rho_min) * (xPhys(low).^dMass);
    ltMask = tril( true( 8 ) );
    meLower = Me0( ltMask );
    sM = reshape( meLower * rhoe', nnz( ltMask ) * nEl, 1 );
    M = localAssemble( Iar( :, 1 ), Iar( :, 2 ), sM, [ nDof, nDof ] );
    M = M + M' - spdiags(diag(M),0,nDof,nDof);
    if tipMassVal > 0 && ~isempty(tipMassDofs)
        M = M + sparse(tipMassDofs, tipMassDofs, tipMassVal * ones(numel(tipMassDofs),1), nDof, nDof);
    end

    % ---- inertial load from current mode-shape estimate
    if auditFreezeMode
        uhat = frozenModeEstimate;
    else
        uhat = U;
    end
    nrm = norm( uhat( free ) );
    if nrm == 0, nrm = 1; end
    uhat = uhat / nrm;
    F = M * uhat;
    F(fixed) = 0;
    if auditFreezeLoad
        if isempty(frozenLoad)
            frozenLoad = F;
        else
            F = frozenLoad;
        end
    end

    % ---- solve
    U(:) = 0;
    U( free ) = decomposition( K( free, free ), 'chol','lower' ) \ F( free );
    uhatNew = U;
    nrmNew = norm( uhatNew( free ) );
    if nrmNew == 0, nrmNew = 1; end
    uhatNew = uhatNew / nrmNew;
    sgn = sign( uhat( free )' * uhatNew( free ) );
    if sgn == 0, sgn = 1; end
    modeCos = abs(uhat( free )' * uhatNew( free ));
    uhatNew = sgn * uhatNew;
    du = norm( uhatNew( free ) - uhat( free ) ) / max( 1, norm( uhat( free ) ) );

    % ---- compliance sensitivity (treating F fixed in-iteration)
    dc = dsK .* sum( ( U( cMat ) * Ke0 ) .* U( cMat ), 2 );
    if ft == 1
        % Sensitivity filter (Andreassen et al., 2011)
        xMat = reshape( max( 1e-3, x ), nely, nelx );
        dc = imfilter( reshape( x .* dc, nely, nelx ), h, bcF ) ./ Hs ./ xMat;
        dV0 = imfilter( reshape( x .* dV, nely, nelx ), h, bcF ) ./ Hs ./ xMat;
    else
        % Chain-rule for density filter
        dc = imfilter( reshape( dc, nely, nelx ) ./ dHs, h, bcF );
        dV0 = imfilter( reshape( dV, nely, nelx ) ./ dHs, h, bcF );
    end

    % ---- OC update (robust bisection bracket + finite guards)
    xOldAudit = x;
    [x, ch, lambdaOC] = localOcUpdate(x, act, dc, dV0, move, mean(xPhys));
    rmsCh = sqrt(mean((x - xOldAudit).^2));

    penalLog = penal;
    [penal,beta] = deal(cnt(penal,penalCnt,loop), cnt(beta,betaCnt,loop));

    cVal = full(F' * U);
    stageInfo.c(end+1,1)  = cVal;
    stageInfo.v(end+1,1)  = mean(xPhys);
    stageInfo.ch(end+1,1) = ch;
    stageInfo.rms_ch(end+1,1) = rmsCh;
    if recordHistory
        % omega1 stays NaN: this method eigensolves only at the end, and
        % plan section 5 forbids adding a solve merely to fill a column.
        histRec = struct('iter', loop, 'stage', histStage, 'stage_iter', loop, ...
            'xPhys', xPhys, 'volfrac', volfrac, ...
            'objective', cVal, 'elapsed_s', toc(loop_tic), ...
            'x', x, 'xOld', xOldAudit, 'move_limit', move);
        if ~isempty(xPhysPrevHist)
            histRec.xPhysPrev = xPhysPrevHist;
        end
        stageInfo.history = topopt_history_record(stageInfo.history, histRec);
        xPhysPrevHist = xPhys;
    end
    if auditCollect
        [stageInfo, ~] = localRecordAuditStep( ...
            stageInfo, loop, xOldAudit, x, xPhys, act, dc, dV0, ...
            lambdaOC, move, modeCos, du, norm(F(free)), auditSnapshotEvery);
    end
    if saveFrqIterations
        stageInfo.freq_iter_omega(loop,:) = localFirstNOmegas( ...
            xPhys, free, nEl, nDof, Iar, Ke, Me0, E0, Emin, penalLog, ...
            rho0, rho_min, dMass, xMassCut, tipMassDofs, tipMassVal, 3);
    end
    if nHistModes > 0
        stageInfo.xHist(:,end+1) = xPhys;
    end

    if doPlot
        fprintf('S2 It.:%5i C:%10.4e V:%7.3f ch:%0.2e du:%0.2e |F|:%9.2e penal:%5.2f beta:%5.1f eta:%6.3f\n', ...
            loop, cVal, mean(xPhys), ch, du, norm(F(free)), penal, beta, eta);
        plotTopology( ...
            xPhys, nelx, nely, ...
            formatTopologyTitle(approachName, volfrac, NaN), ...
            true, 'regular', false);
    end
    if loop > 1 && ch < tolX
        % Extension mode disables ONLY this final native termination.  Stage 1's
        % identical test is left alone: it controls the handoff into this loop,
        % which plan section 4.3 requires to stay active.
        if isnan(stageInfo.native_stop_iter)
            stageInfo.native_stop_iter = loop;
            stageInfo.xphys_at_native_stop = xPhys;
        end
        if ~extendBeyondNativeStop
            break;
        end
    end
end
stageInfo.iterations = loop;
stageInfo.loop_time = toc(loop_tic);
stageInfo.t_iter = stageInfo.loop_time / max(loop, 1);
if loop > 1 && ch < tolX
    stageInfo.stop_reason = 'density_change_tolerance';
else
    stageInfo.stop_reason = 'max_iterations';
end
if recordHistory
    stageInfo.history = topopt_history_finish(stageInfo.history);
end
if saveFrqIterations && isfield(stageInfo, 'freq_iter_omega') && ~isempty(stageInfo.freq_iter_omega)
    stageInfo.freq_iter_omega = stageInfo.freq_iter_omega(1:loop,:);
end

end

%% =======================================================================
function A = localAssemble(i,j,s,sz)
% Use fsparse when available; otherwise fall back to MATLAB sparse.
if exist('fsparse','file') == 2 || exist('fsparse','builtin') == 5
    A = fsparse(i,j,s,sz);
else
    A = sparse(double(i), double(j), s, sz(1), sz(2));
end
end

%% =======================================================================
function [x, ch, lambdaOC] = localOcUpdate(x, act, dc, dV0, move, targetMean)
if isempty(act)
    ch = 0;
    lambdaOC = NaN;
    return;
end

xT = x(act);
xU = min(1, xT + move);
xL = max(0, xT - move);

denom = max(dV0(act), 1e-30);
ocArg = -dc(act) ./ denom;
ocArg(~isfinite(ocArg)) = 1e-30;
ocArg = max(ocArg, 1e-30);
ocP = xT .* sqrt(ocArg);

l1 = 0;
l2 = max(mean(ocP) / max(mean(x), eps), 1);

% Expand upper bracket until the volume target is met.
for k = 1:60
    x(act) = max(max(min(min(ocP / l2, xU), 1), xL), 0);
    if mean(x) <= targetMean + 1e-12
        break;
    end
    l2 = 2 * l2;
    if ~isfinite(l2)
        l2 = realmax('double');
        break;
    end
end

% Bisection with an explicit iteration cap to avoid non-terminating loops.
for k = 1:120
    if (l2 - l1) / max(l2 + l1, eps) <= 1e-4
        break;
    end
    lmid = 0.5 * (l1 + l2);
    if ~isfinite(lmid) || lmid <= 0
        break;
    end
    x(act) = max(max(min(min(ocP / lmid, xU), 1), xL), 0);
    if mean(x) > targetMean
        l1 = lmid;
    else
        l2 = lmid;
    end
end

ch = max(abs(x(act) - xT));
lambdaOC = 0.5 * (l1 + l2);
end

%% =======================================================================
function [stageInfo, stepAbs] = localRecordAuditStep( ...
    stageInfo, loop, xOld, xNew, xPhys, act, dc, dV0, ...
    lambdaOC, move, modeCos, du, forceNorm, snapshotEvery)
% Scalar diagnostics only; no values produced here feed back into the solver.
if ~isfield(stageInfo, 'audit') || ~isfield(stageInfo.audit, 'iter')
    stageInfo.audit = struct( ...
        'iter', [], 'lambdaOC', [], 'stepMean', [], 'stepRms', [], ...
        'stepP95', [], 'stepActiveFrac', [], 'moveFrac', [], 'grayFrac', [], ...
        'dcMean', [], 'dcStd', [], 'dcMaxAbs', [], 'dcP95Abs', [], ...
        'ocArgMean', [], 'ocArgCV', [], 'modeCos', [], 'modeAngleDeg', [], ...
        'du', [], 'forceNorm', [], 'snapshotIter', [], 'xPhysSnapshots', []);
end
stepAbs = abs(xNew(act) - xOld(act));
sens = dc(act);
denom = max(dV0(act), 1e-30);
ocArg = max(-sens ./ denom, 1e-30);

stageInfo.audit.iter(end+1,1) = loop;
stageInfo.audit.lambdaOC(end+1,1) = lambdaOC;
stageInfo.audit.stepMean(end+1,1) = mean(stepAbs);
stageInfo.audit.stepRms(end+1,1) = sqrt(mean(stepAbs.^2));
stageInfo.audit.stepP95(end+1,1) = localPercentile(abs(stepAbs), 95);
stageInfo.audit.stepActiveFrac(end+1,1) = mean(stepAbs > 1e-12);
stageInfo.audit.moveFrac(end+1,1) = mean(stepAbs >= 0.999 * move);
stageInfo.audit.grayFrac(end+1,1) = mean(xPhys(act) > 0.1 & xPhys(act) < 0.9);
stageInfo.audit.dcMean(end+1,1) = mean(sens);
stageInfo.audit.dcStd(end+1,1) = std(sens);
stageInfo.audit.dcMaxAbs(end+1,1) = max(abs(sens));
stageInfo.audit.dcP95Abs(end+1,1) = localPercentile(abs(sens), 95);
stageInfo.audit.ocArgMean(end+1,1) = mean(ocArg);
stageInfo.audit.ocArgCV(end+1,1) = std(ocArg) / max(mean(ocArg), eps);
stageInfo.audit.modeCos(end+1,1) = modeCos;
stageInfo.audit.modeAngleDeg(end+1,1) = acosd(min(1, max(-1, modeCos)));
stageInfo.audit.du(end+1,1) = du;
stageInfo.audit.forceNorm(end+1,1) = forceNorm;

if mod(loop - 1, snapshotEvery) == 0
    stageInfo.audit.snapshotIter(end+1,1) = loop;
    stageInfo.audit.xPhysSnapshots(:,end+1) = xPhys(:);
end
end

%% =======================================================================
function q = localPercentile(values, percentile)
values = sort(values(:));
if isempty(values)
    q = NaN;
    return;
end
pos = 1 + (numel(values) - 1) * percentile / 100;
lo = floor(pos);
hi = ceil(pos);
if lo == hi
    q = values(lo);
else
    q = values(lo) + (pos - lo) * (values(hi) - values(lo));
end
end

%% =======================================================================
function omega1 = localFirstOmega(xPhys, free, nEl, nDof, Iar, Ke, Me0, E0, Emin, penal, ...
    rho0, rho_min, dMass, xMassCut, tipMassDofs, tipMassVal)
% Compute first natural circular frequency (rad/s) from current topology.
omega = localFirstNOmegas( ...
    xPhys, free, nEl, nDof, Iar, Ke, Me0, E0, Emin, penal, ...
    rho0, rho_min, dMass, xMassCut, tipMassDofs, tipMassVal, 1);
omega1 = omega(1);
end

%% =======================================================================
function omegaHist = localModeHistory(xHist, free, nEl, nDof, Iar, Ke, Me0, E0, Emin, penal, ...
    rho0, rho_min, dMass, xMassCut, tipMassDofs, tipMassVal, nModes)
% Compute first nModes frequencies for each stored topology iterate.
if isempty(xHist)
    omegaHist = NaN(0, nModes);
    return;
end
nIter = size(xHist,2);
omegaHist = NaN(nIter, nModes);
for k = 1:nIter
    omegaHist(k,:) = localFirstNOmegas( ...
        xHist(:,k), free, nEl, nDof, Iar, Ke, Me0, E0, Emin, penal, ...
        rho0, rho_min, dMass, xMassCut, tipMassDofs, tipMassVal, nModes);
end
end

%% =======================================================================
function omegas = localFirstNOmegas(xPhys, free, nEl, nDof, Iar, Ke, Me0, E0, Emin, penal, ...
    rho0, rho_min, dMass, xMassCut, tipMassDofs, tipMassVal, nModes)
% Compute first nModes natural circular frequencies (rad/s).
omegas = NaN(1, nModes);
if nModes < 1
    return;
end

try
    % Stiffness matrix
    sK = (Emin + xPhys.^penal * (E0 - Emin));
    sK = reshape(Ke(:) * sK', length(Ke) * nEl, 1);
    K = localAssemble(Iar(:,1), Iar(:,2), sK, [nDof, nDof]);
    K = K + K' - spdiags(diag(K), 0, nDof, nDof);

    % Mass matrix (same interpolation as stage 2)
    rhoe = rho_min + (rho0 - rho_min) * xPhys;
    low = xPhys <= xMassCut;
    rhoe(low) = rho_min + (rho0 - rho_min) * (xPhys(low).^dMass);
    ltMask = tril(true(8));
    meLower = Me0(ltMask);
    sM = reshape(meLower * rhoe', nnz(ltMask) * nEl, 1);
    M = localAssemble(Iar(:,1), Iar(:,2), sM, [nDof, nDof]);
    M = M + M' - spdiags(diag(M), 0, nDof, nDof);
    if tipMassVal > 0 && ~isempty(tipMassDofs)
        M = M + sparse(tipMassDofs, tipMassDofs, tipMassVal * ones(numel(tipMassDofs),1), nDof, nDof);
    end

    Kff = K(free, free);
    Mff = M(free, free);
    nReq = min(nModes, max(1, size(Kff,1)-1));

    eigOpts = struct('disp', 0, 'maxit', 1000, ...
        'v0', localDeterministicEigsStartVector(size(Kff,1)));
    lam = eigs(Kff, Mff, nReq, 'sm', eigOpts);
    lam = sort(real(diag(lam)), 'ascend');
    lam = lam(lam > 0);
    nOk = min(nReq, numel(lam));
    if nOk > 0
        omegas(1:nOk) = sqrt(lam(1:nOk))';
    end
catch
    omegas(:) = NaN;
end
end

function v = localOpt(s, name, defaultVal)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
else
    v = defaultVal;
end
end

function v = localLast(values)
if isempty(values)
    v = NaN;
else
    v = values(end);
end
end

function v = localRelativeLast(values)
if numel(values) < 2
    v = NaN;
else
    v = abs(values(end) - values(end-1)) / max(abs(values(end-1)), eps);
end
end

function v0 = localDeterministicEigsStartVector(n)
% Fixed start vector for EIGS.  Without one, EIGS draws its start vector from
% the global random stream, so the eigenpairs -- and therefore the entire
% design trajectory -- depend on stream state and on the order in which runs
% execute within a session.  The performance benchmark requires bit-identical
% repetition (examples/Performance/PLAN_two_table_redesign.md, section 6.1),
% so draw the vector from a private stream and leave the global stream alone.
s  = RandStream('twister', 'Seed', 42);
v0 = randn(s, n, 1);
v0 = v0 / norm(v0);
end

function localEnsurePlotHelpersOnPath()
if exist('plotTopology', 'file') == 2 && exist('formatTopologyTitle', 'file') == 2
    return;
end
thisDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(fileparts(thisDir)));
toolsDir = fullfile(repoRoot, 'tools');
if exist(toolsDir, 'dir') == 7
    addpath(toolsDir);
end
end

function tf = localParseVisualizeLive(value, defaultValue)
if nargin < 2
    defaultValue = true;
end
if isempty(value)
    tf = defaultValue;
    return;
end
if islogical(value) && isscalar(value)
    tf = value;
    return;
end
if isnumeric(value) && isscalar(value)
    tf = value ~= 0;
    return;
end
if isstring(value) && isscalar(value)
    value = char(value);
end
if ischar(value)
    key = lower(strtrim(value));
    if any(strcmp(key, {'yes','y','true','1','on'}))
        tf = true;
        return;
    end
    if any(strcmp(key, {'no','n','false','0','off'}))
        tf = false;
        return;
    end
end
error('top99neo_inertial_freq:InvalidVisualizeLive', ...
    'visualize_live must be yes/no (case-insensitive) or boolean-like.');
end

function quality = localParseVisualizationQuality(value)
if isstring(value) && isscalar(value)
    value = char(value);
end
if ischar(value)
    key = lower(strtrim(value));
    if isempty(key)
        quality = 'regular';
        return;
    end
    if any(strcmp(key, {'regular', 'smooth'}))
        quality = key;
        return;
    end
end
error('top99neo_inertial_freq:InvalidVisualizationQuality', ...
    'visualization_quality must be "regular" or "smooth".');
end

function name = localApproachName(runCfg, defaultName)
if isstruct(runCfg) && isfield(runCfg, 'approach_name') && ~isempty(runCfg.approach_name)
    name = char(string(runCfg.approach_name));
else
    name = defaultName;
end
end
