function [g, dg] = massInterpolation(rho, mass)
%MASSINTERPOLATION  Mass interpolation factor g(rho) and its derivative.
%
%   [g,dg] = OLH.MATERIAL.MASSINTERPOLATION(rho, cfg.material.mass)
%
%   Value and derivative are produced TOGETHER and by the same branch, so they
%   cannot drift apart.  No optimizer code knows the piecewise formula.
%
%   Models, with the printed equation each one implements:
%
%     'eq2'   g = rho^q                                    Du & Olhoff eq. (2)
%             "the power q>=1.  Apart from exceptions ... normally, q=1 is
%             chosen."  With q=1 this is the linear model, with no low-density
%             cut-off at all.
%
%     'eq4'   g = rho            (rho >  cutoff)           Du & Olhoff eq. (4)
%             g = rho^r          (rho <= cutoff)
%             After Tcherniak (2002), "with a slight modification to avoid
%             numerical singularity".  DISCONTINUOUS at rho = cutoff; the paper
%             says so and calls it numerically unproblematic.
%
%     'eq4a'  g = rho            (rho >  cutoff)           Du & Olhoff eq. (4a)
%             g = c0*rho^6       (rho <= cutoff),  c0 = 1e5
%             "the coefficient c0 = 10^5 enforces the C0 continuity at the
%             value rho_e = 0.1".
%
%     'eq4b'  g = rho                    (rho >  cutoff)   Du & Olhoff eq. (4b)
%             g = c1*rho^6 + c2*rho^7    (rho <= cutoff),  c1 = 6e5, c2 = -5e6
%             "the two coefficients c1 = 6x10^5 and c2 = -5x10^6 ensure the C1
%             continuity of the interpolation model".
%
%   Sec. 2.2 reports that all three of (4), (4a) and (4b) were applied and gave
%   "only negligible differences in the final results".  All three are published;
%   choosing between them is a reconstruction decision.
%
%   ON THE PRINTED CONSTANTS.  c0, c1 and c2 are printed for cutoff = 0.1 and
%   r = 6 and are used here verbatim.  They are NOT re-derived from cutoff and r,
%   because re-deriving them would (a) perturb the values in their last bits and
%   (b) silently redefine a published model.  If cutoff or r is changed away from
%   the printed values this function REFUSES rather than produce a model whose
%   continuity claim no longer holds.

model  = mass.model;
q      = mass.q;
r      = mass.lowDensityExponent;
cutoff = mass.cutoff;

rho = rho(:);
g  = zeros(size(rho));
dg = zeros(size(rho));
lo = rho <= cutoff;   hi = ~lo;

switch model
    case 'eq2'
        if q == 1
            g  = rho;             dg = ones(size(rho));
        else
            g  = rho.^q;          dg = q*rho.^(q-1);
        end
        return
    case {'eq4','eq4a','eq4b'}
        if q ~= 1
            error('olh:material:massQ', ...
               ['Mass models (4)/(4a)/(4b) are printed with the high-density ' ...
                'branch g = rho, i.e. q = 1.  Got q = %g.'], q);
        end
    otherwise
        error('olh:material:massModel','unknown mass model ''%s''', model);
end

g(hi)  = rho(hi);
dg(hi) = 1;

switch model
    case 'eq4'                                   % eq. (4)
        g(lo)  = rho(lo).^r;     dg(lo) = r*rho(lo).^(r-1);
    case 'eq4a'                                  % eq. (4a), c0 = 1e5  (C^0)
        local_requirePrinted(cutoff, r, 'eq4a');
        c0 = 1e5;
        g(lo)  = c0*rho(lo).^6;  dg(lo) = 6*c0*rho(lo).^5;
    case 'eq4b'                                  % eq. (4b), c1=6e5 c2=-5e6 (C^1)
        local_requirePrinted(cutoff, r, 'eq4b');
        c1 = 6e5; c2 = -5e6;
        g(lo)  = c1*rho(lo).^6 + c2*rho(lo).^7;
        dg(lo) = 6*c1*rho(lo).^5 + 7*c2*rho(lo).^6;
end
end

function local_requirePrinted(cutoff, r, model)
if cutoff ~= 0.1 || r ~= 6
    error('olh:material:massConstants', ...
       ['Mass model ''%s'' uses the coefficients printed by Du & Olhoff for ' ...
        'cutoff = 0.1 and r = 6.  With cutoff = %g and r = %g those constants no ' ...
        'longer give the continuity the model is named for.  Refusing to produce ' ...
        'a differently-shaped law under a published model name; use ''eq4'', ' ...
        'which carries no continuity claim.'], model, cutoff, r);
end
end
