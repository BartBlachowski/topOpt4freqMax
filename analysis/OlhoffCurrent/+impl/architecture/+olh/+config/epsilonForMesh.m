function eps = epsilonForMesh(nelx, nely)
%EPSILONFORMESH  The outer convergence tolerance, scaled with the mesh.
%
%       eps = 0.05 * sqrt(NE/3200)
%
%   3200 = 160*20 is the reference mesh, where eps = 0.05.  The scaling keeps
%   eps/sqrt(NE) -- the RMS density change per element -- constant under
%   refinement, so the criterion means the same thing at every resolution.
%
%   CLASS C.  The value 0.05 and this scaling law are both reconstruction: Du &
%   Olhoff (sec. 3.5.1) call for "a small, predefined value" and never give one.
%
%   This function is the ONLY definition of the law.  It was previously retyped
%   in six audit runners; the expression is reproduced here character for
%   character so that every historical tolerance is bitwise reproduced.
eps = 0.05*sqrt(nelx*nely/3200);
end
