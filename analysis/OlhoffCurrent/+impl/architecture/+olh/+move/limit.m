function [mv, state] = limit(cfg, outer, hist, state)
%LIMIT  The move-limit policy in force at this outer iteration.
%
%   EVIDENCE CLASSIFICATION
%
%   A. SPECIFIED by Du & Olhoff (2007).  The ONLY bound placed on the design
%      increment in the printed formulation is the box (25f),
%          0 < rho_min <= rho_e + drho_e <= 1 .
%      The strings "move limit", "trust region" and "step size" do not occur
%      anywhere in the paper, nor in Olhoff & Du (2014).
%
%   B. NOT specified, but supported by the authors' own lineage:
%      Krog & Olhoff (CISM, Eq. 103) make the multiple-eigenvalue sensitivity
%      model a FIRST-ORDER DIRECTIONAL expansion about the current design,
%      a + Da = a + eps*e, valid only for increments of restricted MAGNITUDE.
%      No numeric value is given anywhere in the lineage.
%
%   C. PURE RECONSTRUCTION.  Every functional form below -- the contraction
%      rate, the ladder levels, the transition criteria and all numeric values.
%      Do not describe any of it as the authors'.
%
%   POLICIES  (cfg.move.policy)
%     'fixed'       mv = cfg.move.initial
%     'geometric'   mv_k = max(minimum, initial * ratio^(k-1)), optionally
%                   started only once coalescence is first detected, so the
%                   early transient is left untouched
%     'ladder'      a finite declared ladder cfg.move.levels, descending one
%                   rung when progress stalls
%     'trustRatio'  hold the move so that the realized increment stays inside a
%                   trust measure; contract when the predicted-vs-realized
%                   eigenvalue gain ratio degrades.  Lineage-DERIVED, not
%                   lineage-specified.
%
%   state fields: mv, stage, lastBeta, stall, ratioHist, coalSeen, lastStage,
%                 lastRealized

if isempty(state)
    state = struct('mv',cfg.move.initial,'stage',1,'lastBeta',NaN,'stall',0, ...
                   'ratioHist',[],'coalSeen',false,'lastStage',0, ...
                   'lastRealized',NaN,'ex',[],'stageStarts',1,'descents',zeros(0,4));
end

% has the pair coalesced yet?  (used by the geometric policy's optional trigger)
if ~state.coalSeen && outer > 1 && ~isempty(hist.N) && any(hist.N >= 2)
    state.coalSeen = true;
end

switch cfg.move.policy
    case 'fixed'
        mv = cfg.move.initial;

    case 'geometric'
        k = outer;
        if cfg.move.geometric.afterCoalescence
            if ~state.coalSeen
                mv = cfg.move.initial;                  % untouched transient
                state.mv = mv; return
            end
            k = outer - find(hist.N >= 2, 1) + 1;       % restart clock at coalescence
        end
        mv = max(cfg.move.minimum, cfg.move.initial * cfg.move.geometric.ratio^(max(k,1)-1));

    case 'ladder'
        % Stall detector.  A max-minus-min test over one window cannot fire
        % while the signal oscillates (which it does under a fixed move), so
        % compare the MEAN of the last window against the mean of the window
        % before it: that measures net progress and is insensitive to the
        % oscillation riding on top of it.  This windowed form is also the
        % persistence requirement -- one anomalous iteration cannot trigger a
        % descent.
        %
        % WHICH SIGNAL is watched is cfg.move.continuation.signal:
        %
        %   'boundVariable' the bound variable beta of (25a).  "Has the
        %                   objective stopped improving?"  Mesh-dependent by
        %                   construction: beta's improvement history is a
        %                   property of the optimization path, not of the
        %                   design's resolution state.
        %
        %   'designRms'     the design update ||drho||_2/sqrt(NE) -- the same
        %                   quantity the outer criterion measures.  "Has the
        %                   update stopped DECREASING at the present scale?"
        %                   Differential, so the mesh-dependent baseline of the
        %                   signal cancels identically, and no absolute
        %                   threshold on it is used.
        %
        %   'stageExhaustion'  the FROZEN two-branch rule E = A OR B of
        %                   two_branch_maturity_240/PREREGISTRATION.md, evaluated
        %                   online by olh.move.exhaustion.  "Has this move level
        %                   stopped producing useful topology evolution?"  A
        %                   statement about the DESIGN'S dynamics, not about the
        %                   objective's improvement history.  The detector is
        %                   advanced by the solver after each design update and
        %                   read here, so at the top of iteration `outer` it
        %                   carries information through outer-1 -- the same
        %                   causal structure the beta stall detector has.
        %
        %                   No dwell clock is needed: descending resets the
        %                   detector's whole window to the new stage, which is a
        %                   strictly longer re-arm than cfg.move.continuation.window.
        % 'boundVariable' and 'designRms' use the same window and tolerance.
        % 'designRms' introduces no new numerical constant.  It was TESTED and
        % NOT ADOPTED (audit_s2_design_continuation, verdict
        % S2_LADDER_ITSELF_DEFECTIVE).  'stageExhaustion' uses neither the
        % window nor the tolerance, and is handled in its own branch below.
        %
        if strcmp(cfg.move.continuation.signal, 'stageExhaustion')
            if ~isempty(state.ex) && state.ex.declared && ...
                    state.stage < numel(cfg.move.levels)
                state.descents(end+1,:) = [outer, state.stage, ...
                    state.ex.declIter, state.ex.declBegin];
                state.stage       = state.stage + 1;
                state.lastStage   = outer;
                state.stageStarts(end+1) = outer;
                state.ex.stageStart = outer;
                state.ex.cntA = 0;  state.ex.cntB = 0;
                state.ex.declared = false;
                state.ex.events(end+1,:) = [outer, state.ex.declIter, state.ex.declBegin];
                state.ex.eventBranch{end+1} = state.ex.declBranch;
                state.ex.declIter = NaN; state.ex.declBegin = NaN; state.ex.declBranch = '';
            end
            mv = cfg.move.levels(state.stage);
            state.mv = mv;
            return
        end
        W   = cfg.move.continuation.window;
        tol = cfg.move.continuation.tolerance;
        switch cfg.move.continuation.signal
            case 'boundVariable'
                b = hist.beta;  wantDrop = false;   % beta INCREASES when useful
            case 'designRms'
                NEloc = cfg.domain.mesh.nelx*cfg.domain.mesh.nely;
                b = hist.dxNorm2/sqrt(NEloc);
                wantDrop = true;                    % the update DECREASES when useful
            otherwise
                error('olh:move:signal','unknown move.continuation.signal ''%s''', ...
                    cfg.move.continuation.signal);
        end
        if numel(b) >= 2*W && (outer - state.lastStage) > W
            w2 = mean(b(end-W+1:end));
            w1 = mean(b(end-2*W+1:end-W));
            if wantDrop
                rel = (w1-w2)/max(abs(w1),eps);     % relative DECREASE
            else
                rel = (w2-w1)/max(abs(w1),eps);     % relative INCREASE
            end
            if rel < tol
                state.stage = min(state.stage+1, numel(cfg.move.levels));
                state.lastStage = outer;
            end
        end
        mv = cfg.move.levels(state.stage);

    case 'trustRatio'
        % Predicted-vs-realized gain in lambda_n over the LAST completed outer
        % step.  `predicted` is what the sub-problem promised at that step (beta
        % minus the lambda_n it was built at); `realized` is what the fresh FE
        % analysis actually delivered.  A low ratio means the first-order
        % expansion (Krog & Olhoff Eq. 103) is being pushed beyond its validity.
        mv = state.mv;
        if outer > 1 && numel(hist.beta) >= 1 && isfield(state,'lastRealized')
            predicted = hist.beta(end) - hist.omega(1,end)^2;
            realized  = state.lastRealized;
            if ~isnan(realized) && abs(predicted) > 1e-12
                r = realized/predicted;
                state.ratioHist(end+1) = r;
                if r < cfg.move.trust.loRatio
                    mv = max(cfg.move.minimum, mv*cfg.move.trust.shrink);
                elseif r > cfg.move.trust.hiRatio
                    mv = min(cfg.move.initial,  mv*cfg.move.trust.grow);
                end
            end
        end

    otherwise
        error('olh:move:policy','unknown move policy ''%s''', cfg.move.policy);
end

state.mv = mv;
end
