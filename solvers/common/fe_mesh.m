function mdl = fe_mesh(prob)
%FE_MESH  Q4 plane-stress mesh of a rectangular domain in the COMMON element
%   numbering used by the proposed method, the evaluator and all scripts:
%   element e = ely + elx*nely + 1 with ely counted from the BOTTOM edge,
%   nodes column by column from the bottom-left corner, dofs [ux uy] per node.
%
%   mdl = fe_mesh(prob)   prob from configs/problem_*.m
%
%   Returns the unit-modulus element matrices KE (E = 1) and ME (rho = 1), the
%   assembly index vectors, the free dofs of prob.support and the passive /
%   active element sets.
nelx = prob.nelx;  nely = prob.nely;  L = prob.L;  H = prob.H;
hx = L/nelx;  hy = H/nely;
mdl = struct('nelx', nelx, 'nely', nely, 'L', L, 'H', H, 'hx', hx, 'hy', hy, ...
    'nEl', nelx*nely, 'ndof', 2*(nelx+1)*(nely+1));
mdl.KE = elementStiffness(hx, hy, prob.nu);
mdl.ME = kron((hx*hy/36) * [4 2 1 2; 2 4 2 1; 1 2 4 2; 2 1 2 4], eye(2));
edofMat = zeros(mdl.nEl, 8);
for elx = 0:nelx-1
    for ely = 0:nely-1
        el = ely + elx*nely + 1;
        n1 = (nely+1)*elx + ely;  n2 = (nely+1)*(elx+1) + ely;  n3 = n2 + 1;  n4 = n1 + 1;
        edofMat(el, :) = [2*n1+1, 2*n1+2, 2*n2+1, 2*n2+2, 2*n3+1, 2*n3+2, 2*n4+1, 2*n4+2];
    end
end
mdl.edofMat = edofMat;
mdl.iK = reshape(kron(edofMat, ones(1,8))', [], 1);
mdl.jK = reshape(kron(edofMat, ones(8,1))', [], 1);
nodeIds = (1:(nelx+1)*(nely+1))';
nx = floor((nodeIds-1)/(nely+1))*hx;  ny = mod(nodeIds-1, nely+1)*hy;
switch prob.support
    case 'simplySupported'    % pins (ux, uy) at mid-height of both end faces
        [~, n1] = min(hypot(nx - 0, ny - H/2));  [~, n2] = min(hypot(nx - L, ny - H/2));
        fixedNodes = [n1; n2];
    case 'clamped'            % both end faces clamped
        fixedNodes = nodeIds(abs(nx) < 1e-9 | abs(nx - L) < 1e-9);
    case 'clampedBottom'      % building: bottom edge clamped
        fixedNodes = nodeIds(abs(ny) < 1e-9);
end
mdl.free = setdiff((1:mdl.ndof)', sort([2*fixedNodes-1; 2*fixedNodes]));
mdl.pas = prob.passive(:);
mdl.act = setdiff((1:mdl.nEl)', mdl.pas);
end

function KE = elementStiffness(hx, hy, nu)
% Unit-modulus bilinear Q4, plane stress, 2x2 Gauss quadrature.
D = (1/(1-nu^2)) * [1, nu, 0; nu, 1, 0; 0, 0, 0.5*(1-nu)];
invJ = [2/hx, 0; 0, 2/hy];  detJ = 0.25*hx*hy;  gp = 1/sqrt(3);
KE = zeros(8, 8);
for xi = [-gp, gp]
    for eta = [-gp, gp]
        dN = invJ * (0.25 * [-(1-eta), (1-eta), (1+eta), -(1+eta); -(1-xi), -(1+xi), (1+xi), (1-xi)]);
        B = zeros(3, 8);
        B(1, 1:2:end) = dN(1, :);  B(2, 2:2:end) = dN(2, :);
        B(3, 1:2:end) = dN(2, :);  B(3, 2:2:end) = dN(1, :);
        KE = KE + (B'*D*B)*detJ;
    end
end
KE = (KE + KE')/2;
end
