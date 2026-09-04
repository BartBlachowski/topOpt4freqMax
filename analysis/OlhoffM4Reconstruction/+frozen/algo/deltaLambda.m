function [dlam, ddlam, V, degen] = deltaLambda(F, drho, dOff)
%DELTALAMBDA  Solve the subeigenvalue problem (25d) in its ERRATUM form
%
%       det | f_sk' * drho  -  delta_sk * DELTA(omega^2) | = 0
%
%   for the increments DELTA(lambda_j) and their gradients w.r.t. drho.
%
%   F     : NE x N x N array of generalized gradients (genGrad)
%   drho  : NE x 1 design increment
%   dOff  : OPTIONAL N x 1 vector of eigenvalue OFFSETS lambda_j - lambda_n.
%           Omitted or empty  ->  (25d) exactly as printed, which assumes the
%           N eigenvalues are EXACTLY equal (Du & Olhoff p.96, eq. 17: "let us
%           assume ... a N-fold multiple eigenvalue").
%           Supplied ->  the same determinant with the actual separation
%           retained on the diagonal,
%               det | diag(dOff) + f_sk'*drho - delta_sk*DELTA(omega^2) | = 0,
%           i.e. the eigenvalues of diag(lambda_j) + A(drho) measured from
%           lambda_n.  This reduces to the printed (25d) when dOff = 0 and to
%           the simple result (20)/(23) when the off-diagonals are negligible
%           against the separation.  RECONSTRUCTION (class C) -- it is not
%           written in any source; see WP1/WP3 of
%           audit_multiplicity_reconstruction/.
%
%   dlam  : N x 1 ascending eigenvalue increments
%   ddlam : NE x N gradient matrix,  ddlam(e,j) = d(dlam_j)/d(drho_e)
%           = sum_{s,k} v_js v_jk (f_sk)_e                [RECONSTRUCTION --
%           not stated in the paper, see CLAUDE.md sec.5]
%   V     : N x N eigenvectors of A
%   degen : true if A is (near) zero, where the eigenvectors -- and hence the
%           gradients -- are NOT uniquely defined.  At drho = 0 this is always
%           the case; the identity basis is then used, which reduces the first
%           inner sub-iterate to the diagonal terms.  Logged, not patched.

[NE, N, ~] = size(F);
A = zeros(N);
for s = 1:N
    for k = s:N
        a = F(:,s,k).'*drho;
        A(s,k) = a;  A(k,s) = a;
    end
end
A = (A+A')/2;

scaleA = max(abs(A(:)));
degen = scaleA <= 0;

haveOff = nargin >= 3 && ~isempty(dOff);
if haveOff
    A = A + diag(dOff(:));
end

[V, D] = eig(A);
[dlam, ord] = sort(real(diag(D)),'ascend');
V = V(:,ord);
if haveOff
    % return increments measured from the CURRENT lambda_j, so that the
    % caller's constraint  lambda_j + dlam_j  is the predicted eigenvalue.
    dlam = dlam - dOff(:);
end
for j = 1:N                      % eig returns orthonormal V for symmetric A
    V(:,j) = V(:,j)/norm(V(:,j));
end

if nargout < 2, return; end

ddlam = zeros(NE, N);
for j = 1:N
    vj = V(:,j);
    g = zeros(NE,1);
    for s = 1:N
        for k = 1:N
            w = vj(s)*vj(k);
            if w ~= 0
                g = g + w*F(:,s,k);
            end
        end
    end
    ddlam(:,j) = g;
end
end
