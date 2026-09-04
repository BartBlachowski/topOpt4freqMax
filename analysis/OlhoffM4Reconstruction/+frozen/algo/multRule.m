function [N, st] = multRule(cfg, w, n, Jcalc, st)
%MULTRULE  Step 1 of Fig. 1: decide the multiplicity N of omega_n.
%
%   [N, st] = MULTRULE(cfg, w, n, Jcalc, st)
%
%   `st` carries whatever state the rule needs between outer iterations; it is
%   returned so the caller can pass it back.  The default rule is stateless and
%   ignores it entirely.
%
%   EVIDENCE (audit_multiplicity_reconstruction/WP1_source_evidence.md)
%   ------------------------------------------------------------------
%   Du & Olhoff p.98 is the whole of what is specified:
%       "the term 'multiplicity' is used if the numerical value of the relative
%        difference between eigenfrequencies in question is within a
%        predefined, very small tolerance."
%   The MEASURE (relative frequency difference) and the fact that it is
%   recomputed in step 1 of every outer iteration are class A.  The VALUE is
%   never given in 2007 or 2014; Krog & Olhoff sec.5.3 p.306 report 1e-4 for
%   their own examples (class B, and used there in an algorithm whose detector
%   does NOT gate the ascent mechanism -- see WP1 sec.6.2).  Persistence and
%   hysteresis are mentioned in NO source.
%
%   cfg.multRule
%     'binary'   (default) the memoryless test above.  Reproduces the frozen
%                reconstruction exactly.                            [M0/M1]
%     'latch'    'binary' entry, then a one-way monotone latch: N never
%                decreases again for the rest of the run.  Explicitly declared
%                persistence rule.  CLASS C -- supported by no source.   [M2]
%     'hyst'     enter the cluster at cfg.tolEnter, leave it only above
%                cfg.tolExit > cfg.tolEnter.  CLASS C -- supported by no
%                source.                                                [M3]
%     'subspace' NO classifier.  N is fixed at cfg.subN for the whole run and
%                the subeigenvalue problem carries the actual eigenvalue
%                separation on its diagonal (see deltaLambda's dOff argument),
%                so that it reduces to (25d) at exact degeneracy and to
%                (20)/(24) when the modes are separated.  CLASS C as an
%                interpolation, but assembled only from the paper's own (19),
%                (24), (25c) and (25d).                                 [M4]

if ~isfield(cfg,'multRule') || isempty(cfg.multRule)
    rule = 'binary';
else
    rule = lower(cfg.multRule);
end
if nargin < 5 || isempty(st), st = struct(); end

switch rule
    case 'binary'
        N = greedy(w, n, Jcalc, cfg.tolMult);

    case 'latch'
        N = greedy(w, n, Jcalc, cfg.tolMult);
        if ~isfield(st,'latched'), st.latched = 1; end
        N = max(N, st.latched);
        st.latched = N;

    case 'hyst'
        if ~isfield(st,'held'), st.held = 1; end
        if st.held <= 1
            N = greedy(w, n, Jcalc, cfg.tolEnter);      % entry test
        else
            N = greedy(w, n, Jcalc, cfg.tolExit);       % looser exit test
        end
        st.held = N;

    case 'subspace'
        N = cfg.subN;
        N = min(N, Jcalc - n);      % J = n+N must not exceed Jcalc

    otherwise
        error('multRule:unknown','unknown cfg.multRule ''%s''', rule);
end
end

function N = greedy(w, n, Jcalc, tol)
%GREEDY  The rule as coded in the frozen reconstruction: a run of consecutive
%   modes whose relative frequency difference FROM THE BASE omega_n is below
%   `tol`.  (A diameter test, not a chain test -- the paper does not say which.)
N = 1;
while n+N <= Jcalc-1 && abs(w(n+N)-w(n))/w(n) < tol
    N = N + 1;
end
end
