function [xPhys_stage2,U_stage2,info] = top99neo_inertial_freq(nelx,nely,volfrac,penal,rmin,ft,ftBC,eta,beta,move,maxit,stage1_maxit,bcType,nHistModes,runCfg)
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
if isfield(runCfg, 'visualise_live') && ~isempty(runCfg.visualise_live)
    doPlot = logical(runCfg.visualise_live);
else
    doPlot = true;
end

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

% Boundary conditions + stage-1 point load
[fixed, lcDof, tipMassNode] = localBCAndLoad(nodeNrs,nely,nelx,nDof,bcType);
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
info.stage1 = struct('c',[],'v',[],'ch',[],'xHist',[],'omegaHist',[],'omegaFinal',[]);
info.stage2 = struct('c',[],'v',[],'ch',[],'xHist',[],'omegaHist',[],'omegaFinal',[]);
info.stage1.loadDof = lcDof;

%% ================================ STAGE 1: standard compliance minimization
[xPhys,U] = deal(x, zeros(nDof,1));
[xPhys,U,eta,penal,beta,info.stage1] = localComplianceLoop( ...
    x, xPhys, U, F_point, fixed, free, act, ...
    nelx, nely, nEl, nDof, cMat, Iar, Ke, Ke0, ...
    E0, Emin, penal, rmin, h, Hs, bcF, ft, eta, beta, move, stage1_maxit, ...
    penalCnt, betaCnt, dsK, dV, info.stage1, doPlot, nHistModes, stage1Tol);
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

[xPhys_stage2,U_stage2,eta,penal,beta,info.stage2] = localInertialLoop( ...
    x, xPhys_stage2, U_est, fixed, free, act, ...
    nelx, nely, nEl, nDof, cMat, Iar, Ke, Ke0, Me0, ...
    E0, Emin, rho0, rho_min, dMass, xMassCut, ...
    tipMassDofs, tipMassVal, ...
    penal, rmin, h, Hs, bcF, ft, eta, beta, move, maxit, ...
    penalCnt, betaCnt, dsK, dV, info.stage2, doPlot, stage2Tol, nHistModes);
info.stage2.xFinal = xPhys_stage2;
info.stage2.UFinal = U_stage2;
info.stage2.omegaFinal = localFirstNOmegas( ...
    xPhys_stage2, free, nEl, nDof, Iar, Ke, Me0, E0, Emin, penal, ...
    rho0, rho_min, dMass, xMassCut, tipMassDofs, tipMassVal, finalModes);
info.stage2.omega1 = info.stage2.omegaFinal(1);
if nHistModes > 0
    info.stage1.omegaHist = localModeHistory( ...
        info.stage1.xHist, free, nEl, nDof, Iar, Ke, Me0, E0, Emin, penal, ...
        rho0, rho_min, dMass, xMassCut, tipMassDofs, tipMassVal, nHistModes);
    info.stage2.omegaHist = localModeHistory( ...
        info.stage2.xHist, free, nEl, nDof, Iar, Ke, Me0, E0, Emin, penal, ...
        rho0, rho_min, dMass, xMassCut, tipMassDofs, tipMassVal, nHistModes);
end

if isfinite(info.stage2.omega1)
    fprintf('\nFinal design: omega1 = %.4f rad/s\n', info.stage2.omega1);
    if doPlot
        title(sprintf('\\omega_1 = %.1f rad/s', info.stage2.omega1), 'Interpreter', 'tex');
        drawnow;
    end
end

info.timing = struct();
info.timing.stage1_loop_time = localOpt(info.stage1, 'loop_time', NaN);
info.timing.stage2_loop_time = localOpt(info.stage2, 'loop_time', NaN);
info.timing.stage1_iterations = localOpt(info.stage1, 'iterations', NaN);
info.timing.stage2_iterations = localOpt(info.stage2, 'iterations', NaN);
info.timing.total_loop_time = info.timing.stage1_loop_time + info.timing.stage2_loop_time;
info.timing.total_iterations = info.timing.stage1_iterations + info.timing.stage2_iterations;
info.timing.t_iter = info.timing.total_loop_time / max(info.timing.total_iterations, 1);

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
    eigOpts = struct('disp', 0, 'maxit', 1000);
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
    penalCnt, betaCnt, dsK, dV, stageInfo, doPlot, nHistModes, tolX)

% ---- implicit functions (redefined here: local functions cannot access parent workspace)
prj  = @(v,eta_,beta_) (tanh(beta_*eta_)+tanh(beta_*(v(:)-eta_)))./(tanh(beta_*eta_)+tanh(beta_*(1-eta_)));
deta = @(v,eta_,beta_) - beta_ * csch( beta_ ) .* sech( beta_ * ( v( : ) - eta_ ) ).^2 .* ...
    sinh( v( : ) * beta_ ) .* sinh( ( 1 - v( : ) ) * beta_ );
dprj = @(v,eta_,beta_) beta_*(1-tanh(beta_*(v-eta_)).^2)./(tanh(beta_*eta_)+tanh(beta_*(1-eta_)));
cnt  = @(v,vCnt,l) v+(l>=vCnt{1})*(v<vCnt{2})*(mod(l,vCnt{3})==0)*vCnt{4};

loop = 0;
loop_tic = tic;
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
    [x, ch] = localOcUpdate(x, act, dc, dV0, move, mean(xPhys));

    [penal,beta] = deal(cnt(penal,penalCnt,loop), cnt(beta,betaCnt,loop));

    cVal = full(F' * U);
    stageInfo.c(end+1,1)  = cVal;
    stageInfo.v(end+1,1)  = mean(xPhys);
    stageInfo.ch(end+1,1) = ch;
    if nHistModes > 0
        stageInfo.xHist(:,end+1) = xPhys;
    end

    if doPlot
        fprintf('S1 It.:%5i C:%10.4e V:%7.3f ch:%0.2e penal:%5.2f beta:%5.1f eta:%6.3f\n', ...
            loop, cVal, mean(xPhys), ch, penal, beta, eta);
        colormap(gray); imagesc(1-reshape(xPhys,nely,nelx)); caxis([0 1]); axis equal off; drawnow;
    end
    if loop > 1 && ch < tolX, break; end
end
stageInfo.iterations = loop;
stageInfo.loop_time = toc(loop_tic);
stageInfo.t_iter = stageInfo.loop_time / max(loop, 1);

end

%% =======================================================================
function [xPhys,U,eta,penal,beta,stageInfo] = localInertialLoop( ...
    x, xPhys, U_est, fixed, free, act, ...
    nelx, nely, nEl, nDof, cMat, Iar, Ke, Ke0, Me0, ...
    E0, Emin, rho0, rho_min, dMass, xMassCut, ...
    tipMassDofs, tipMassVal, ...
    penal, rmin, h, Hs, bcF, ft, eta, beta, move, maxit, ...
    penalCnt, betaCnt, dsK, dV, stageInfo, doPlot, stage2Tol, nHistModes)

% ---- implicit functions (redefined here: local functions cannot access parent workspace)
prj  = @(v,eta_,beta_) (tanh(beta_*eta_)+tanh(beta_*(v(:)-eta_)))./(tanh(beta_*eta_)+tanh(beta_*(1-eta_)));
deta = @(v,eta_,beta_) - beta_ * csch( beta_ ) .* sech( beta_ * ( v( : ) - eta_ ) ).^2 .* ...
    sinh( v( : ) * beta_ ) .* sinh( ( 1 - v( : ) ) * beta_ );
dprj = @(v,eta_,beta_) beta_*(1-tanh(beta_*(v-eta_)).^2)./(tanh(beta_*eta_)+tanh(beta_*(1-eta_)));
cnt  = @(v,vCnt,l) v+(l>=vCnt{1})*(v<vCnt{2})*(mod(l,vCnt{3})==0)*vCnt{4};

tolX = stage2Tol;
loop = 0; U = U_est;
loop_tic = tic;

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
    uhat = U;
    nrm = norm( uhat( free ) );
    if nrm == 0, nrm = 1; end
    uhat = uhat / nrm;
    F = M * uhat;
    F(fixed) = 0;

    % ---- solve
    U(:) = 0;
    U( free ) = decomposition( K( free, free ), 'chol','lower' ) \ F( free );
    uhatNew = U;
    nrmNew = norm( uhatNew( free ) );
    if nrmNew == 0, nrmNew = 1; end
    uhatNew = uhatNew / nrmNew;
    sgn = sign( uhat( free )' * uhatNew( free ) );
    if sgn == 0, sgn = 1; end
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
    [x, ch] = localOcUpdate(x, act, dc, dV0, move, mean(xPhys));

    [penal,beta] = deal(cnt(penal,penalCnt,loop), cnt(beta,betaCnt,loop));

    cVal = full(F' * U);
    stageInfo.c(end+1,1)  = cVal;
    stageInfo.v(end+1,1)  = mean(xPhys);
    stageInfo.ch(end+1,1) = ch;
    if nHistModes > 0
        stageInfo.xHist(:,end+1) = xPhys;
    end

    if doPlot
        fprintf('S2 It.:%5i C:%10.4e V:%7.3f ch:%0.2e du:%0.2e |F|:%9.2e penal:%5.2f beta:%5.1f eta:%6.3f\n', ...
            loop, cVal, mean(xPhys), ch, du, norm(F(free)), penal, beta, eta);
        colormap(gray); imagesc(1-reshape(xPhys,nely,nelx)); caxis([0 1]); axis equal off; drawnow;
    end
    if loop > 1 && ch < tolX, break; end
end
stageInfo.iterations = loop;
stageInfo.loop_time = toc(loop_tic);
stageInfo.t_iter = stageInfo.loop_time / max(loop, 1);

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
function [x, ch] = localOcUpdate(x, act, dc, dV0, move, targetMean)
if isempty(act)
    ch = 0;
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

    eigOpts = struct('disp', 0, 'maxit', 1000);
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
