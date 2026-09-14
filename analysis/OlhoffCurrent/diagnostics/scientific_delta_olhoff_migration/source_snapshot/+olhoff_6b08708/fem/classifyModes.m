function T = classifyModes(mdl, M, Phi, omega)
%CLASSIFYMODES  Split modal kinetic energy into axial (x) and transverse (y).
%
%   For the consistent (and lumped) mass matrices built here the x-dofs couple
%   only to x-dofs, so the split is exact:  Ex + Ey = 1 for M-orthonormal modes.
%   Ex ~ 1 marks an extensional/axial mode, Ey ~ 1 a bending mode.
%   nzy counts sign changes of the transverse motion along the beam centreline,
%   which identifies the bending order.

f  = mdl.free;
isx = mod(f,2) == 1;                 % odd global dof index = ux
J = size(Phi,2);
T = zeros(J,4);
for j = 1:J
    v = Phi(:,j);
    vx = v; vx(~isx) = 0;
    vy = v; vy( isx) = 0;
    Ex = vx'*M*vx;  Ey = vy'*M*vy;
    T(j,1) = omega(j);
    T(j,2) = Ex/(Ex+Ey);
    T(j,3) = Ey/(Ex+Ey);
    % sign changes of uy along the mid-height row
    row = mdl.nely/2 + 1;
    if mod(mdl.nely,2)==0
        nodes = mdl.nodenrs(row,:);
        dofy  = 2*nodes(:);
        full_v = zeros(mdl.ndof,1); full_v(f) = v;
        uy = full_v(dofy);
        T(j,4) = sum(abs(diff(sign(uy(abs(uy)>1e-6*max(abs(uy)))))) > 0);
    end
end
end
