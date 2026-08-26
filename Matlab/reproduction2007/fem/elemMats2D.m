function [K0, M0] = elemMats2D(dx, dy, E, nu, rhom, t, elemType, massType)
%ELEMMATS2D  Element stiffness and mass for a rectangular plane-stress element.
%
%   [K0,M0] = ELEMMATS2D(dx,dy,E,nu,rhom,t,elemType,massType)
%
%   Node ordering is CCW from the lower-left corner, matching top88's edofMat:
%       1 = (-1,-1)   2 = (+1,-1)   3 = (+1,+1)   4 = (-1,+1)
%   dof ordering per node is [ux uy].
%
%   elemType : 'Q4'   plain 4-node bilinear, 2x2 Gauss (shear-locking prone)
%              'Q6'   Q4 + Wilson incompatible displacement modes, statically
%                     condensed.  Rectangles only (no Taylor correction needed).
%   massType : 'consistent' or 'lumped'   (the paper does not state which)
%
%   Returns the FULLY SOLID element matrices.  SIMP scaling is applied by the
%   assembler, not here.

if nargin < 7 || isempty(elemType), elemType = 'Q4'; end
if nargin < 8 || isempty(massType), massType = 'consistent'; end

% ---- plane stress constitutive matrix -----------------------------------
D = E/(1-nu^2) * [1 nu 0; nu 1 0; 0 0 (1-nu)/2];

% ---- 2x2 Gauss rule ------------------------------------------------------
g = 1/sqrt(3);
gp = [-g -g; g -g; g g; -g g];
w  = [1 1 1 1];

detJ = dx*dy/4;          % constant for a rectangle
dxi_dx  = 2/dx;          % d(xi)/dx
deta_dy = 2/dy;          % d(eta)/dy

xn = [-1  1  1 -1];      % natural coords of the 4 nodes
yn = [-1 -1  1  1];

Kaa = zeros(8);          % compatible-compatible
Kab = zeros(8,4);        % compatible-incompatible
Kbb = zeros(4);          % incompatible-incompatible
Mc  = zeros(8);

for q = 1:4
    xi = gp(q,1); eta = gp(q,2); wq = w(q);

    % --- bilinear shape functions and their cartesian derivatives -------
    N     = 0.25*(1+xn*xi).*(1+yn*eta);
    dNdxi = 0.25*xn.*(1+yn*eta);
    dNdet = 0.25*(1+xn*xi).*yn;
    dNdx  = dNdxi*dxi_dx;
    dNdy  = dNdet*deta_dy;

    B = zeros(3,8);
    B(1,1:2:7) = dNdx;
    B(2,2:2:8) = dNdy;
    B(3,1:2:7) = dNdy;
    B(3,2:2:8) = dNdx;

    Kaa = Kaa + B'*D*B*detJ*wq*t;

    % --- consistent mass (compatible modes only) ------------------------
    Nm = zeros(2,8);
    Nm(1,1:2:7) = N;
    Nm(2,2:2:8) = N;
    Mc = Mc + Nm'*Nm*rhom*detJ*wq*t;

    % --- Wilson incompatible modes  P1 = 1-xi^2 , P2 = 1-eta^2 ----------
    if strcmpi(elemType,'Q6')
        dP1dx = -2*xi *dxi_dx;   dP1dy = 0;
        dP2dx = 0;               dP2dy = -2*eta*deta_dy;
        % internal dofs  a = [a1 a2 a3 a4] :
        %   u_i = P1*a1 + P2*a2 ,  v_i = P1*a3 + P2*a4
        Bi = zeros(3,4);
        Bi(1,1) = dP1dx;  Bi(1,2) = dP2dx;                    % exx
        Bi(2,3) = dP1dy;  Bi(2,4) = dP2dy;                    % eyy
        Bi(3,1) = dP1dy;  Bi(3,2) = dP2dy;                    % gxy (from u_i,y)
        Bi(3,3) = dP1dx;  Bi(3,4) = dP2dx;                    % gxy (from v_i,x)

        Kab = Kab + B' *D*Bi*detJ*wq*t;
        Kbb = Kbb + Bi'*D*Bi*detJ*wq*t;
    end
end

% ---- static condensation of the internal dofs ---------------------------
if strcmpi(elemType,'Q6')
    K0 = Kaa - Kab*(Kbb\Kab');
    K0 = 0.5*(K0+K0');
else
    K0 = Kaa;
end

% ---- mass ---------------------------------------------------------------
switch lower(massType)
    case 'consistent'
        M0 = 0.5*(Mc+Mc');
    case 'lumped'
        M0 = diag(sum(Mc,2));           % row-sum (HRZ reduces to this for Q4)
    otherwise
        error('elemMats2D:massType','unknown massType %s',massType);
end
end
