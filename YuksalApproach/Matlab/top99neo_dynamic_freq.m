function [xPhys, info] = top99neo_dynamic_freq(nelx,nely,volfrac,penal,rmin,ft,ftBC,move,maxit,bcType)
%TOP99NEO_DYNAMIC_FREQ Dynamic-code frequency maximization history (Section 3 style).
%
% This solver follows the paper's dynamic setup used for Figure 6:
% - direct eigenvalue-based optimization at each iteration
% - sensitivity filtering (ft = 1)
% - mass interpolation with low-density penalization (Eq. 10, d = 6, x_cut = 0.1)
% - repeated-mode handling via a 4% proximity threshold between omega1 and omega2
%
% Outputs:
%   xPhys : final physical design field
%   info  : struct with omega histories and final omega1

if nargin < 1  || isempty(nelx),    nelx = 320; end
if nargin < 2  || isempty(nely),    nely = 40;  end
if nargin < 3  || isempty(volfrac), volfrac = 0.5; end
if nargin < 4  || isempty(penal),   penal = 3.0; end
if nargin < 5  || isempty(rmin),    rmin = 2.5; end
if nargin < 6  || isempty(ft),      ft = 1; end
if nargin < 7  || isempty(ftBC),    ftBC = 'N'; end
if nargin < 8  || isempty(move),    move = 0.01; end
if nargin < 9  || isempty(maxit),   maxit = 200; end
if nargin < 10 || isempty(bcType),  bcType = "simply"; end

if ft ~= 1
    error('top99neo_dynamic_freq currently supports ft = 1 (sensitivity filtering) only.');
end

% Material/constants from paper.
E0 = 1e7;
Emin = 1e-9 * E0;
nu = 0.3;
rho0 = 1;
rho_min = 1e-9 * rho0;
dMass = 6;
xMassCut = 0.1;
repTol = 0.04; % repeated-mode threshold from the paper's dynamic setup.

% Mesh / indexing.
nEl = nelx * nely;
nodeNrs = int32(reshape(1:(1+nelx)*(1+nely), 1+nely, 1+nelx));
cVec = reshape(2 * nodeNrs(1:end-1,1:end-1) + 1, nEl, 1);
cMat = cVec + int32([0,1,2*nely+[2,3,0,1],-2,-1]);
nDof = (1+nely) * (1+nelx) * 2;

[sI,sII] = deal([]);
for j = 1:8
    sI = cat(2, sI, j:8);
    sII = cat(2, sII, repmat(j,1,8-j+1));
end
[iK,jK] = deal(cMat(:,sI)', cMat(:,sII)');
Iar = sort([iK(:), jK(:)], 2, 'descend'); clear iK jK

% Q4 plane-stress stiffness (same as top99neo-like implementation).
c1 = [12;3;-6;-3;-6;-3;0;3;12;3;0;-3;-6;-3;-6;12;-3;0;-3;-6;3;12;3;...
    -6;3;-6;12;3;-6;-3;12;3;0;12;-3;12];
c2 = [-4;3;-2;9;2;-3;4;-9;-4;-9;4;-3;2;9;-2;-4;-3;4;9;2;3;-4;-9;-2;...
    3;2;-4;3;-2;9;-4;-9;4;-4;-3;-4];
Ke = 1/(1-nu^2)/24*(c1 + nu .* c2);
Ke0(tril(ones(8))==1) = Ke';
Ke0 = reshape(Ke0,8,8);
Ke0 = Ke0 + Ke0' - diag(diag(Ke0));

% Element mass matrix using physical geometry from the selected benchmark case.
[beamL, beamH, tipMassFrac] = localPhysicalSetup(bcType);
elemArea = (beamL / nelx) * (beamH / nely);
MeS = (elemArea/36) * [4 2 1 2; 2 4 2 1; 1 2 4 2; 2 1 2 4];
Me0 = kron(MeS, eye(2));

% Boundary conditions (same setup used in Yuksel scripts).
[fixed, tipMassDofs] = localBCAndTipMass(nodeNrs,nely,bcType);
free = setdiff(1:nDof, fixed);
act = (1:nEl)';

tipMassVal = 0;
if tipMassFrac > 0 && ~isempty(tipMassDofs)
    permittedMass = volfrac * beamL * beamH * rho0;
    tipMassVal = tipMassFrac * permittedMass;
end

% Filter.
if strcmpi(char(ftBC), 'N'), bcF = 'symmetric'; else, bcF = 0; end
[dy,dx] = meshgrid(-ceil(rmin)+1:ceil(rmin)-1, -ceil(rmin)+1:ceil(rmin)-1);
h = max(0, rmin - sqrt(dx.^2 + dy.^2));
Hs = imfilter(ones(nely,nelx), h, bcF);

% Initialization.
x = volfrac * ones(nEl,1);
xPhys = x;
dV = ones(nEl,1) / (nEl * volfrac);
info = struct();
info.omegaHist = NaN(maxit,3);
info.chHist = NaN(maxit,1);
info.repActive = false(maxit,1);

for it = 1:maxit
    xPhys(act) = x(act);

    % Assemble K.
    Ee = Emin + xPhys.^penal * (E0 - Emin);
    sK = reshape(Ke(:) * Ee', length(Ke) * nEl, 1);
    K = localAssemble(Iar(:,1), Iar(:,2), sK, [nDof, nDof]);
    K = K + K' - spdiags(diag(K),0,nDof,nDof);

    % Assemble M with Eq. (10)-style low-density mass penalization.
    rhoe = rho_min + (rho0-rho_min) * xPhys;
    low = xPhys <= xMassCut;
    rhoe(low) = rho_min + (rho0-rho_min) * (xPhys(low).^dMass);
    ltMask = tril(true(8));
    meLower = Me0(ltMask);
    sM = reshape(meLower * rhoe', nnz(ltMask) * nEl, 1);
    M = localAssemble(Iar(:,1), Iar(:,2), sM, [nDof, nDof]);
    M = M + M' - spdiags(diag(M),0,nDof,nDof);
    if tipMassVal > 0 && ~isempty(tipMassDofs)
        M = M + sparse(tipMassDofs, tipMassDofs, tipMassVal * ones(numel(tipMassDofs),1), nDof, nDof);
    end

    Kff = K(free,free);
    Mff = M(free,free);

    eigOpts = struct('disp',0,'maxit',1200,'tol',1e-10);
    [V,D] = eigs(Kff,Mff,3,'sm',eigOpts);
    [lam,ord] = sort(real(diag(D)), 'ascend');
    V = real(V(:,ord));
    lam = max(lam, eps);
    omega = sqrt(lam);
    info.omegaHist(it,1:numel(omega)) = omega';

    % Element sensitivities for first two modes.
    drho = (rho0-rho_min) * ones(nEl,1);
    drho(low) = dMass * (rho0-rho_min) * (xPhys(low).^(dMass-1));
    dlam = zeros(nEl,2);
    for j = 1:2
        vj = V(:,j);
        vj = vj / sqrt(max(eps, real(vj' * (Mff * vj))));
        phi = zeros(nDof,1);
        phi(free) = vj;
        pe = phi(cMat);
        dlam(:,j) = penal*(E0-Emin)*(xPhys.^(penal-1)).*sum((pe*Ke0).*pe,2) ...
                  - lam(j) * drho .* sum((pe*Me0).*pe,2);
    end

    % Repeated eigenvalue handling.
    if (omega(2)-omega(1))/max(omega(1),eps) < repTol
        dlamObj = max(dlam(:,1), dlam(:,2));
        info.repActive(it) = true;
    else
        dlamObj = dlam(:,1);
    end
    % OC update requires positive sensitivities; use a global shift.
    dlamObj = dlamObj - min(dlamObj) + 1e-12;

    % Maximize lambda1 by minimizing -lambda1 with OC.
    dc = -dlamObj;
    xMat = reshape(max(1e-3, x), nely, nelx);
    dcF = imfilter(reshape(x .* dc, nely, nelx), h, bcF) ./ Hs ./ xMat;
    dV0 = imfilter(reshape(x .* dV, nely, nelx), h, bcF) ./ Hs ./ xMat;

    xOld = x;
    xT = x(act);
    xU = xT + move;
    xL = xT - move;
    ocArg = -dcF(act) ./ dV0(act);
    ocArg = max(ocArg, 1e-30);
    ocP = xT .* sqrt(ocArg);
    l = [0, mean(ocP)/mean(x)];
    while (l(2)-l(1))/(l(2)+l(1)) > 1e-4
        lmid = 0.5 * (l(1) + l(2));
        x(act) = max(max(min(min(ocP/lmid, xU), 1), xL), 0);
        if mean(x) > mean(xPhys), l(1) = lmid; else, l(2) = lmid; end
    end

    ch = max(abs(x - xOld));
    info.chHist(it) = ch;
    fprintf('Dyn It.:%4d w1:%8.3f w2:%8.3f w3:%8.3f ch:%0.2e rep:%d\n', ...
        it, omega(1), omega(2), omega(3), ch, info.repActive(it));
end

info.xFinal = xPhys;
info.omega1 = info.omegaHist(maxit,1);
info.tipMassVal = tipMassVal;
fprintf('\nDynamic code final: omega1 = %.4f rad/s\n', info.omega1);

end

function [beamL, beamH, tipMassFrac] = localPhysicalSetup(bcType)
switch lower(string(bcType))
    case "cantilever"
        beamL = 15;
        beamH = 10;
        tipMassFrac = 0.20;
    case {"simply","fixedpinned"}
        beamL = 8;
        beamH = 1;
        tipMassFrac = 0;
    otherwise
        error('Unsupported bcType: %s', string(bcType));
end
end

function [fixed, tipMassDofs] = localBCAndTipMass(nodeNrs,nely,bcType)
tipMassDofs = [];
switch lower(string(bcType))
    case "simply"
        midRow = round(nely/2) + 1;
        leftMid = nodeNrs(midRow, 1);
        rightMid = nodeNrs(midRow, end);
        fixed = [2*leftMid-1, 2*leftMid, 2*rightMid-1, 2*rightMid];
    case "cantilever"
        leftNodes = nodeNrs(:,1);
        fixed = union(2*leftNodes-1, 2*leftNodes);
        midRow = round((nely+1)/2);
        tipNode = nodeNrs(midRow, end);
        tipMassDofs = [2*tipNode-1, 2*tipNode];
    case "fixedpinned"
        leftNodes = nodeNrs(:,1);
        fixed = union(2*leftNodes-1, 2*leftNodes);
        midRow = round(nely/2) + 1;
        rightMid = nodeNrs(midRow, end);
        fixed = union(fixed, [2*rightMid-1, 2*rightMid]);
    otherwise
        error('Unsupported bcType: %s', string(bcType));
end
fixed = unique(fixed(:))';
end

function A = localAssemble(i,j,s,sz)
if exist('fsparse','file') == 2 || exist('fsparse','builtin') == 5
    A = fsparse(i,j,s,sz);
else
    A = sparse(double(i), double(j), s, sz(1), sz(2));
end
end
