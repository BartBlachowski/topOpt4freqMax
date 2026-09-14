function [mv, state] = moveControl(cfg, outer, hist, state)
%MOVECONTROL  Step/move control families for the Du & Olhoff (2007) outer loop.
%
%   EVIDENCE CLASSIFICATION (see audit_stepcontrol/WP1_evidence.md):
%
%   A. SPECIFIED by Du & Olhoff (2007).  The ONLY bound placed on the design
%      increment in the printed formulation is the box (25f),
%          0 < rho_min <= rho_e + drho_e <= 1 .
%      The strings "move limit", "trust region" and "step size" do not occur
%      anywhere in the paper, nor in Olhoff & Du (2014).
%
%   B. NOT specified by Du & Olhoff (2007), but supported by the authors'
%      own methodological lineage:
%        - Krog & Olhoff (CISM, Eq.103): the multiple-eigenvalue sensitivity
%          model is a FIRST-ORDER DIRECTIONAL expansion about the current
%          design, a + Da = a + eps*e, with eps "a small positive scalar which
%          gives the magnitude of the perturbation in this direction".  The
%          model underlying (18)/(25d) is therefore only valid for increments
%          of restricted MAGNITUDE -- a theoretical basis for a move bound,
%          though no numeric value is given anywhere in the lineage.
%        - Krog & Olhoff (abstract): "such problems may be treated like
%          differentiable optimization problems, if some restrictions are
%          imposed on the vector of design changes at each iteration."
%          NOTE: in context this refers to the off-diagonal equalities
%          Eq(134), i.e. a restriction on DIRECTION, not magnitude.  It is
%          recorded here so the distinction is not lost.
%        - Krog & Olhoff (p.305), on eigenfrequency topology optimization:
%          "we see a nice and stable convergence of the eigenfrequencies, and
%          it is noted that final convergence for the two problems is very
%          slow".  Slow tails are normal in this lineage.
%
%   C. PURE RECONSTRUCTION.  Every functional form below -- the contraction
%      rate, the staging levels, the transition criteria, and all numeric
%      values -- is reconstruction.  None of it is stated by Du & Olhoff
%      (2007) or by the lineage.  Do not describe any of it as the authors'.
%
%   FAMILIES
%     'S0'  fixed move (the frozen strict baseline)
%               mv = cfg.move
%     'S1'  monotone geometric contraction, floored
%               mv_k = max(moveMin, move0 * gamma^(k-1))
%           optionally triggered only after coalescence is first detected
%           (cfg.s1AfterCoal), so the early transient is left untouched.
%     'S2'  staged move: a finite declared ladder cfg.s2Levels, descending one
%           rung when progress stalls -- when the relative change in the
%           objective bound beta over the last cfg.s2Window outer iterations
%           falls below cfg.s2Tol.
%     'S3'  lineage-derived: hold the move so that the realized increment
%           magnitude stays inside a trust measure on ||drho||_2 / sqrt(NE),
%           i.e. bound the RMS density change per iteration, which is the
%           direct discrete analogue of Krog & Olhoff's eps in Eq(103).
%           Contracts when the predicted-vs-realized eigenvalue gain ratio
%           degrades (the first-order model losing validity), expands when it
%           is good.  Labelled lineage-DERIVED, not lineage-specified.
%
%   state fields: mv (current bound), stage, lastBeta, stall, ratioHist

if isempty(state)
    state = struct('mv',cfg.move,'stage',1,'lastBeta',NaN,'stall',0, ...
                   'ratioHist',[],'coalSeen',false,'lastStage',0, ...
                   'lastRealized',NaN);
end

fam = upper(cfg.moveFamily);

% has the pair coalesced yet?  (used by S1's optional trigger)
if ~state.coalSeen && outer > 1 && ~isempty(hist.N) && any(hist.N >= 2)
    state.coalSeen = true;
end

switch fam
    case 'S0'
        mv = cfg.move;

    case 'S1'
        k = outer;
        if cfg.s1AfterCoal
            if ~state.coalSeen
                mv = cfg.move;                      % untouched transient
                state.mv = mv; return
            end
            k = outer - find(hist.N >= 2, 1) + 1;   % restart clock at coalescence
        end
        mv = max(cfg.moveMin, cfg.move * cfg.s1Gamma^(max(k,1)-1));

    case 'S2'
        % Stall detector.  A max-minus-min test over one window cannot fire
        % while the signal oscillates (which it does under a fixed move), so
        % compare the MEAN of the last window against the mean of the window
        % before it: that measures net progress and is insensitive to the
        % oscillation riding on top of it.  This windowed form is also the
        % persistence requirement -- one anomalous iteration cannot trigger a
        % descent.
        %
        % WHICH SIGNAL is watched is cfg.s2Signal:
        %
        %   'beta'  (legacy) the OBJECTIVE bound.  "Has the objective stopped
        %           improving?"  Mesh-dependent by construction: beta's
        %           improvement history is a property of the optimization path,
        %           not of the design's resolution state.
        %
        %   'drms'  the DESIGN update d = ||drho||_2/sqrt(NE) -- the same
        %           quantity the outer criterion measures.  "Has the design
        %           update stopped DECREASING at the present move scale?"
        %           While d still falls the current scale is doing work; when
        %           it plateaus the design has settled into the largest step
        %           this scale permits.  The test is DIFFERENTIAL, so the
        %           mesh-dependent baseline of d cancels identically -- no
        %           absolute threshold on d is used here, and this test is
        %           therefore independent of the outer convergence test, which
        %           compares d against a fixed absolute epsilon.
        %
        % Both use the same window cfg.s2Window and tolerance cfg.s2Tol.  No
        % new numerical constant is introduced by 'drms'.
        if ~isfield(cfg,'s2Signal') || isempty(cfg.s2Signal)
            sig = 'beta';
        else
            sig = lower(cfg.s2Signal);
        end
        switch sig
            case 'beta'
                b = hist.beta;  wantDrop = false;   % beta INCREASES when useful
            case 'drms'
                NEloc = cfg.nelx*cfg.nely;
                b = hist.dxNorm2/sqrt(NEloc);       % d_RMS, the frozen measure
                wantDrop = true;                    % d_RMS DECREASES when useful
            otherwise
                error('moveControl:s2Signal','unknown s2Signal %s',sig);
        end
        if numel(b) >= 2*cfg.s2Window && (outer - state.lastStage) > cfg.s2Window
            w2 = mean(b(end-cfg.s2Window+1:end));
            w1 = mean(b(end-2*cfg.s2Window+1:end-cfg.s2Window));
            if wantDrop
                rel = (w1-w2)/max(abs(w1),eps);     % relative DECREASE
            else
                rel = (w2-w1)/max(abs(w1),eps);     % relative INCREASE
            end
            if rel < cfg.s2Tol
                state.stage = min(state.stage+1, numel(cfg.s2Levels));
                state.lastStage = outer;
            end
        end
        mv = cfg.s2Levels(state.stage);

    case 'S3'
        % Predicted-vs-realized gain in lambda_n over the LAST completed outer
        % step.  `predicted` is what the frozen sub-problem promised at that
        % step (beta minus the lambda_n it was built at); `realized` is what
        % the fresh FE analysis actually delivered, supplied by the caller as
        % state.lastRealized.  A low ratio means the first-order expansion
        % (Krog & Olhoff Eq.103) is being pushed beyond its validity -> shrink.
        mv = state.mv;
        if outer > 1 && numel(hist.beta) >= 1 && isfield(state,'lastRealized')
            predicted = hist.beta(end) - hist.omega(1,end)^2;
            realized  = state.lastRealized;
            if ~isnan(realized) && abs(predicted) > 1e-12
                r = realized/predicted;
                state.ratioHist(end+1) = r;
                if r < cfg.s3Lo
                    mv = max(cfg.moveMin, mv*cfg.s3Down);   % model losing validity
                elseif r > cfg.s3Hi
                    mv = min(cfg.move,    mv*cfg.s3Up);     % model trustworthy
                end
            end
        end

    otherwise
        error('moveControl:family','unknown moveFamily %s',fam);
end

state.mv = mv;
end
