# Algebraic generalization of the frozen reduction

This is code analysis only. No new mesh, FE solve, topology trajectory or density
update is used. Let z denote the increment, G(z) the symmetric matrix assembled
by `deltaLambda` from the fixed generalized gradients, e_j its ordered
sub-eigenvalues after any offsets, and b_j=lam_j-dOff_j when offsets are present.
Production cluster rows are beta <= b_j+e_j. Positive lamref scaling does not
change these inequalities. All following claims include the existing fixed
increment box and beta box, and the affine next-mode and volume rows.

## N=1

The scalar sub-eigenvalue minus its offset is F11' z. Thus the cluster row is
beta <= lam1+F11' z. With the simple next-mode row and affine volume, this is
an LP. No numerical SDP is needed. The fixed-coordinate box can be kept exactly.

## N=2 with production-consistent offsets

If dOff_j=lam_j-lam1, all b_j=lam1. Both cluster inequalities reduce to
lambda_min(G(z)+diag(dOff)) >= beta-lam1, equivalently

    G(z)+diag(dOff) - (beta-lam1) I is positive semidefinite.

For a symmetric 2x2 matrix [[a,b],[b,c]], positive semidefiniteness is exactly

    norm([(a-c)/2; b]) <= (a+c)/2.

Substituting the threshold in its diagonal gives the reference SOC, including
its affine centre and constant caused by dOff. Off-diagonal gradients contribute
to b; they do not prevent convexity. Retaining them is essential for equivalence.
The higher ordered cluster eigenvalue row is redundant; its smooth local
expression can be concave even though removing it leaves the identical convex
feasible set. This distinction matters to approximation methods.

## dOff absent

The code returns ascending eigenvalues e_j of G. The physical lam_j are also
ascending. Therefore lam1+e1 <= lam_j+e_j for every j. The first cluster row
again dominates, with the LMI G(z)-(beta-lam1)I >= 0. This is a property of the
implemented sorted-branch reconstruction. It does not establish physical
validity of using the exactly-multiple formula for separated physical modes.

## N>2

For consistent offsets, the same affine PSD inequality is an exact spectral
representation. It is an SDP constraint of order N. A general PSD cone of order
N>2 is not generally representable by one second-order cone. Special diagonal
or block-2x2 structure can reduce to LP/SOC pieces, but no such universal
structure follows from the production gradients. No production-ready SDP
implementation is claimed or introduced in this study.

For arbitrary offsets, first-row dominance is sufficient when b1<=b_j for all
j, even if b_j are not monotone. It need not hold otherwise. Demanding a lower
bound on a higher ordered eigenvalue alone is generally nonconvex, so arbitrary
inconsistent offsets must not be silently folded into the simple LMI. Validate
lam-dOff before conic dispatch.

## Off-diagonal equality route and next modes

With offDiag=false, innerLoop uses individual affine diagonal predictions and
adds the two signed inequalities for each Fsk' z=0. This entire reconstruction
is an LP for every N, even if those equality rows are rank deficient. Removing
those rows would change the problem. A simple next-mode constraint is affine;
several separately affine next-mode constraints stay affine. A genuinely
multiple next cluster would require its own consistent spectral treatment;
the code's simple-mode approximation is not replaced here.

## Conditions and limits

G must be affine in increments with fixed coefficients and genuinely symmetric.
The primal code mirrors upper-triangular F; its derivative loops use both
triangles, so their equality must be verified for gradient consistency. Filtering
all frozen coefficients keeps G affine even if the filter is non-integrable as
an outer density gradient field. Convexity of this frozen subproblem says
nothing about conservativeness of the filter as rho changes.

An optional nonlinear volFun (projection path), nonlinear density-dependent
coefficients, or inconsistent offsets invalidates the present generic LP/SOC
assembly unless each extra part has an independently proved conic formulation.
The audited state has none of these complications.

The prospective C480 control uses multiplicity.method='subspace', size 2.
`olh.multi.detect` fixes N=min(subspaceSize,Jcalc-n), independently of the
current spectrum. The same configuration therefore stays N=2. A future guarded
SOCP adapter could explicitly reject unsupported states (no accepted outer
update) and retain the exact existing method as a declared compatibility
fallback outside this treatment; a mathematical SDP expression is not itself
an implemented or validated fallback. The promotion gate states what is and is
not established by this frozen evidence.

At a repeated predicted sub-eigenvalue, lambda_min is nonsmooth. The affine
PSD/SOC formulation remains exact, but KKT must use an admissible conic dual
subgradient (possibly a mixture of eigendirections), not an arbitrary single
eigenvector gradient from eig. The present oracle is separated, so its smooth
certificate is valid. A future adapter must either validate that generalized
certificate or reject the unsupported certificate case before any outer update.
