function D = fp_dualbound(P, x, mu0, nu0)
%FP_DUALBOUND  Rigorous weak-duality lower bound on the SOCP optimum,
%   MAXIMIZED over the dual variables, independent of any solver's duals.
%
%   Primal:  min f'x  s.t. ||Ac x - bc|| <= dc'x - gammac,  Alin x <= blin,
%            xmin <= x <= xmax.
%   For p = mu*w with ||p|| <= mu (mu >= 0) and nu >= 0, Cauchy-Schwarz gives
%   ||Ac x - bc|| >= w'(Ac x - bc), hence for every feasible x
%       f'x >= q'x - p'bc + mu*gammac - nu'blin,   q = f + Ac'p - mu dc + Alin'nu,
%   and minimizing the right side over the box is closed-form:
%       Dval(p,mu,nu) = sum_i min(q_i xmin_i, q_i xmax_i) - p'bc + mu gammac - nu'blin.
%   Dval is concave; it is maximized here with a derivative-free simplex
%   search over (p1,p2,muExtra,nu1,nu2) with mu = ||p|| + max(muExtra,0),
%   nu = max(nu,0), started from the solver's duals and from the cone
%   direction at x.  The result is a valid bound for ANY returned point.
s = P.Ac*x - P.bc; ns = norm(s);
if ns > 0, w = s/ns; else, w = [1;0]; end
Dfun = @(z) local_D(P, z);
z0 = [mu0*w; 0; max(nu0(:),0)];
best = struct('val',-inf,'z',z0);
starts = {z0, [w; 0; 0; 0.7], [1.0*w; 1e-3; 0; 0.7]};
opts = optimset('MaxFunEvals',20000,'MaxIter',20000,'TolX',1e-14,'TolFun',1e-15,'Display','off');
for k = 1:numel(starts)
    z = starts{k};
    for rep = 1:4
        [z, nv] = fminsearch(@(zz) -Dfun(zz), z, opts);
        if -nv > best.val, best.val = -nv; best.z = z; end
    end
end
% polish: fix the cone direction to the EXACT subgradient direction at x
% (w = s/||s||) so that the cone complementarity term vanishes identically,
% and optimize only (muExtra, nu).  Keep whichever bound is larger.
zb = best.z; pb = zb(1:2); mub = norm(pb) + max(zb(3),0);
Dfix = @(y) local_D(P, [mub_scale(y(1))*w; 0; y(2:3)]);
y = [mub; max(zb(4:5),0)];
for rep = 1:4, [y, nv] = fminsearch(@(yy) -Dfix(yy), y, opts); end
% prefer the exactly-aligned candidate whenever it ties the best within 1e-12
if -nv >= best.val - 1e-12, best.val = -nv; best.z = [mub_scale(y(1))*w; 0; y(2:3)]; end
zA = [mub_scale(y(1))*w; 0; y(2:3)]; valA = Dfix(y);
z = best.z; p = z(1:2); mu = norm(p) + max(z(3),0); nu = max(z(4:5),0);
pA = zA(1:2); muA = norm(pA); nuA = max(zA(4:5),0);
D = struct('dual_bound',best.val,'p',p.','mu',mu,'nu',nu.','w',(p/max(norm(p),eps)).', ...
    'primal',P.f.'*x,'gap',P.f.'*x - best.val,'gap_relative_to_gain',(P.f.'*x - best.val)/max(abs(x(end)-1),eps));
D.aligned = struct('p',pA.','mu',muA,'nu',nuA.','w',w.','dual_bound',valA,'gap',P.f.'*x - valA, ...
    'note','cone direction fixed to the exact subgradient s/||s|| at x; (mu, nu) maximized');
end
function m = mub_scale(y)
m = max(y, 0);
end
function v = local_D(P, z)
p = z(1:2); mu = norm(p) + max(z(3),0); nu = max(z(4:5),0);
q = P.f + P.Ac.'*p - mu*P.dc + P.Alin.'*nu;
v = sum(min(q.*P.xmin, q.*P.xmax)) - p.'*P.bc + mu*P.gammac - nu.'*P.blin;
end
