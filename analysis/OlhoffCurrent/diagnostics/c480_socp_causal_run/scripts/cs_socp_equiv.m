function [ok, Q] = cs_socp_equiv(P, x, apex)
%CS_SOCP_EQUIV  Preregistered equivalence checks E9-E12 at a point x = [drho; bs].
%   Production rows come from P.evalProd, a character-for-character mirror of
%   innerLoop's constraint block (deltaLambda included).
[fval, dfdx] = P.evalProd(x);
lin = P.Alin*x - P.blin;
Q = struct();
Q.row_nextmode_err = abs(fval(3) - lin(1));
Q.row_volume_err   = abs(fval(4) - lin(2));
Q.E9  = Q.row_nextmode_err <= 1e-12 && Q.row_volume_err <= 1e-12;
Q.cone_err = abs(fval(1) - P.coneResid(x));
Q.E10 = Q.cone_err <= 1e-10;
Q.redundancy = fval(2) - fval(1);
Q.E11 = Q.redundancy <= 1e-12;
if apex
    Q.grad_relerr = NaN; Q.E12 = true; Q.E12_skipped_apex = true;
else
    g = P.coneGrad(x);
    Q.grad_relerr = norm(dfdx(1,:).' - g, inf)/max(norm(dfdx(1,:), inf), eps);
    Q.E12 = Q.grad_relerr <= 1e-9; Q.E12_skipped_apex = false;
end
ok = Q.E9 && Q.E10 && Q.E11 && Q.E12;
Q.ok = ok;
end
