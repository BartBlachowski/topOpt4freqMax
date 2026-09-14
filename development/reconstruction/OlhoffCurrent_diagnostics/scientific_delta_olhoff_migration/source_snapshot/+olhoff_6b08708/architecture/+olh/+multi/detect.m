function [N, st] = detect(cfg, w, n, Jcalc, st)
%DETECT  Step 1 of Fig. 1: decide the multiplicity N of omega_n.
%
%   [N, st] = OLH.MULTI.DETECT(cfg, w, n, Jcalc, st)
%
%   `st` carries whatever state the method needs between outer iterations.
%
%   EVIDENCE.  Du & Olhoff sec. 3.5.1 is the whole of what is specified:
%       "the term 'multiplicity' is used if the numerical value of the relative
%        difference between eigenfrequencies in question is within a
%        predefined, very small tolerance."
%   The MEASURE (relative frequency difference) and the fact that it is
%   recomputed in step 1 of every outer iteration are class A.  The VALUE is
%   never given in 2007 or 2014; Krog & Olhoff sec. 5.3 report 1e-4 for their
%   own examples (class B, and used there in an algorithm whose detector does
%   NOT gate the ascent mechanism).  Persistence and hysteresis appear in NO
%   source.
%
%   cfg.multiplicity.method
%     'binary'      the memoryless test above.  CLASS C only in its threshold.
%     'latch'       'binary' entry, then a one-way monotone latch: N never
%                   decreases again.  Explicitly declared persistence. CLASS C.
%     'hysteresis'  enter the cluster at enterTolerance, leave it only above
%                   exitTolerance > enterTolerance.  CLASS C.
%     'subspace'    NO classifier.  N is fixed at subspaceSize for the whole
%                   run.  CLASS C as an interpolation, but assembled only from
%                   the paper's own (19), (24), (25c) and (25d).
%
%   NOTE.  Whether the subeigenvalue problem retains the diagonal offsets
%   diag(lambda_j - lambda_n) is a SEPARATE choice, cfg.multiplicity.
%   diagonalOffsets, and is not decided here.  The legacy code derived it from
%   the method name; that coupling is resolved once, in olh.config.fromLegacy.

method = cfg.multiplicity.method;
if nargin < 5 || isempty(st), st = struct(); end

switch method
    case 'binary'
        N = local_greedy(w, n, Jcalc, cfg.multiplicity.tolerance);

    case 'latch'
        N = local_greedy(w, n, Jcalc, cfg.multiplicity.tolerance);
        if ~isfield(st,'latched'), st.latched = 1; end
        N = max(N, st.latched);
        st.latched = N;

    case 'hysteresis'
        if ~isfield(st,'held'), st.held = 1; end
        if st.held <= 1
            N = local_greedy(w, n, Jcalc, cfg.multiplicity.enterTolerance);   % entry test
        else
            N = local_greedy(w, n, Jcalc, cfg.multiplicity.exitTolerance);    % looser exit test
        end
        st.held = N;

    case 'subspace'
        N = cfg.multiplicity.subspaceSize;
        N = min(N, Jcalc - n);      % J = n+N must not exceed Jcalc

    otherwise
        error('olh:multi:method','unknown multiplicity method ''%s''', method);
end
end

function N = local_greedy(w, n, Jcalc, tol)
%LOCAL_GREEDY  A run of consecutive modes whose relative frequency difference
%   FROM THE BASE omega_n is below tol.  (A diameter test, not a chain test --
%   the paper does not say which.)
N = 1;
while n+N <= Jcalc-1 && abs(w(n+N)-w(n))/w(n) < tol
    N = N + 1;
end
end
