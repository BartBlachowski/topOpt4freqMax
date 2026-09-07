function [Mc, node] = lumped_mass(nodeNrs, nDof, where, mass)
% LUMPED_MASS  Design-independent concentrated non-structural mass.
%
%   [Mc, node] = lumped_mass(nodeNrs, nDof, where, mass)
%
%   Builds a diagonal nDof x nDof sparse matrix holding a concentrated mass
%   m_c attached at a single node, acting on both translational DOFs.  Used by
%   the gap-maximization example of Olhoff & Du (2014) section 3.3 / Fig. 7(a):
%   a clamped-clamped beam with "a concentrated mass m_c attached at the
%   mid-point of the lower edge of the beam-like structure ... which has the
%   value m_c = 1/2 m_b (here m_b is the total mass of the initial design)".
%
%   The mass is NON-STRUCTURAL: it does not depend on rho, so
%   dM_c/drho_e = 0 and it contributes to M in the eigenproblem (1b) but NOT to
%   the generalized gradients f_sk of Eq. (13).  The caller must add Mc to M
%   after assembly and must NOT pass it to generalized_gradients.
%
%   Inputs
%     nodeNrs  (nely+1) x (nelx+1)  1-based node numbers, row 1 = y = 0
%     nDof     scalar               total DOF count
%     where    char                 'bottom_mid' (Fig. 7a) | 'top_mid' | 'centre'
%     mass     scalar               m_c
%
%   Outputs
%     Mc       nDof x nDof sparse   diagonal, entries m_c at ux and uy of `node`
%     node     scalar               1-based node index the mass sits on
%
%   Requires an even nelx so that a node exists exactly at mid-span.
%
%   Reference: Olhoff & Du (2014), section 3.3, Fig. 7(a).

[nrow, ncol] = size(nodeNrs);
nely = nrow - 1;
nelx = ncol - 1;

if mod(nelx, 2) ~= 0
    error('lumped_mass:OddNelx', ...
        'A node at mid-span requires an even nelx (got %d).', nelx);
end
mid_col = nelx/2 + 1;

switch lower(strtrim(char(where)))
    case 'bottom_mid'
        node = nodeNrs(1, mid_col);
    case 'top_mid'
        node = nodeNrs(end, mid_col);
    case 'centre'
        if mod(nely, 2) ~= 0
            error('lumped_mass:OddNely', ...
                'where = ''centre'' requires an even nely (got %d).', nely);
        end
        node = nodeNrs(nely/2 + 1, mid_col);
    otherwise
        error('lumped_mass:UnknownLocation', ...
            'Unknown location ''%s''. Use bottom_mid, top_mid or centre.', where);
end

dofs = [2*node-1; 2*node];
Mc   = sparse(dofs, dofs, mass*ones(2,1), nDof, nDof);
end
