function [P, dP] = projectDensity(x, betaProj, eta)
%PROJECTDENSITY  Smooth Heaviside (tanh) projection and its derivative.
%
%   [P,dP] = PROJECTDENSITY(x,betaProj,eta)
%
%       P(x) = [tanh(b*eta) + tanh(b*(x-eta))] / [tanh(b*eta) + tanh(b*(1-eta))]
%      dP(x) = b*sech(b*(x-eta))^2            / [tanh(b*eta) + tanh(b*(1-eta))]
%
%   betaProj = 0 returns the EXACT identity P(x)=x, dP=1, which is the b->0
%   limit of the expression above.  Written as a separate branch so that the
%   identity is exact rather than 0/0.
%
%   CLASS D -- this operator appears nowhere in Du & Olhoff (2007), in the
%   erratum, in Olhoff & Du (2014), or in the Krog & Olhoff lineage.  See
%   audit_m4_projection_invariance/WP1_RECONSTRUCTION_BOUNDARY.md.
%
%   NOTE ON NAMING: betaProj is the PROJECTION SHARPNESS.  It is unrelated to
%   the bound-formulation variable beta of Du & Olhoff eq. (25a), which the
%   solver records as hist.beta.

if betaProj <= 0
    P  = x;
    dP = ones(size(x));
    return
end
den = tanh(betaProj*eta) + tanh(betaProj*(1-eta));
P   = (tanh(betaProj*eta) + tanh(betaProj*(x-eta))) / den;
dP  = betaProj * sech(betaProj*(x-eta)).^2 / den;
end
