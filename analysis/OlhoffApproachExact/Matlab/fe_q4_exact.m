function [Ke_star, Me_star] = fe_q4_exact(nu, t, dx, dy)
% FE_Q4_EXACT  Unit-material Q4 bilinear plane-stress element matrices.
%
%   [Ke_star, Me_star] = fe_q4_exact(nu, t, dx, dy)
%
%   Returns the 8x8 stiffness matrix for E = 1 (plane-stress, Poisson nu,
%   thickness t, element width dx, height dy) and the 8x8 consistent mass
%   matrix for rho = 1.
%
%   Paper interpolation applied by the caller (Du & Olhoff 2007, Eq. 1-2):
%       K_e(rho_e) = rho_e^p        * Ke_star
%       M_e(rho_e) = m(rho_e, mode) * Me_star   [see mass_interp]
%
%   DOF ordering (8 per element, same as assembly cMat):
%       [ ux1 uy1 ux2 uy2 ux3 uy3 ux4 uy4 ]
%   Node local numbering (xi, eta):
%       1 = (-1,-1) = lower-left  (LL)
%       2 = (+1,-1) = lower-right (LR)
%       3 = (+1,+1) = upper-right (UR)
%       4 = (-1,+1) = upper-left  (UL)
%   2x2 Gauss quadrature.
%
%   Reference: Du & Olhoff (2007), Struct Multidisc Optim 34:91-110.

C    = 1/(1-nu^2) * [1, nu, 0; nu, 1, 0; 0, 0, (1-nu)/2];
gp   = [-1/sqrt(3),  1/sqrt(3)];
xiN  = [-1  1  1 -1];
etaN = [-1 -1  1  1];
detJ = (dx*dy)/4;
invJ = [2/dx, 0; 0, 2/dy];

Ke_star = zeros(8,8);
Me_star = zeros(8,8);

for gi = 1:2
    for gj = 1:2
        xi  = gp(gi);
        eta = gp(gj);

        N       = 0.25*(1 + xi*xiN).*(1 + eta*etaN);
        dN_dxi  = 0.25*xiN .*(1 + eta*etaN);
        dN_deta = 0.25*etaN.*(1 + xi*xiN);

        grads  = invJ * [dN_dxi; dN_deta];
        dN_dx  = grads(1,:);
        dN_dy  = grads(2,:);

        B    = zeros(3,8);
        Nmat = zeros(2,8);
        for a = 1:4
            B(1, 2*a-1) = dN_dx(a);
            B(2, 2*a)   = dN_dy(a);
            B(3, 2*a-1) = dN_dy(a);
            B(3, 2*a)   = dN_dx(a);
            Nmat(1, 2*a-1) = N(a);
            Nmat(2, 2*a)   = N(a);
        end

        Ke_star = Ke_star + (B'*C*B) * (t*detJ);
        Me_star = Me_star + (Nmat'*Nmat) * (t*detJ);
    end
end
end
