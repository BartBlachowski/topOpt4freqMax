function gz = projChain(H, Hs, sChain, grho)
%PROJCHAIN  Chain rule  d/dz = W' * S * d/drho  for the three-field map.
%
%   gz = PROJCHAIN(H,Hs,sChain,grho)
%
%   With  zTilde = W z,  W = diag(1/Hs)*H,  and  S = diag(sChain),
%
%       (dG/dz)_j = sum_e (H_ej/Hs_e) * s_e * (dG/drho)_e
%                 = [ H * ( (sChain .* grho) ./ Hs ) ]_j          (H symmetric)
%
%   grho may hold several density-gradient vectors as columns; each is
%   transformed independently.
%
%   This is NOT the Sigmund sensitivity filter (filter/applyFilter.m); the two
%   coincide for no choice of parameters.  See WP2 sec.3.1.

gz = zeros(size(grho));
for c = 1:size(grho,2)
    gz(:,c) = H * ((sChain .* grho(:,c)) ./ Hs);
end
end
