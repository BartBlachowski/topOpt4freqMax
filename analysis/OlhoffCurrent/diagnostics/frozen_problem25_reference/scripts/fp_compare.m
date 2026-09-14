function out = fp_compare()
%FP_COMPARE  Parts 11-12: the certified reference against the repeated-MMA
%   states P19 / M500 / M5000 and along the frozen replay checkpoints.
%   Also exports the fields the figures need.  No density is updated.
S = fp_setup();                                   %#ok<NASGU>
ev = fullfile(S.study,'evaluations');
L = load(fullfile(ev,'frozen_ctx.mat'),'ctx','xP19','xM500'); P = fp_problem(L.ctx);
R = load(fullfile(ev,'conic_reference.mat'),'xRef','muRef','xsiRef','etaRef','masks','qRef');
M = load(fullfile(ev,'mma_replay.mat'),'CK','H','out','xM5000','checkpoints');
NE = P.NE; nvar = P.nvar; move = P.move;
xRef = R.xRef; dRef = xRef(1:NE); bsRef = xRef(end); gainRef = bsRef - 1;
sRow0 = sqrt(mean((P.F11/P.lamref).^2));
tolB = 1e-6*max(P.xmax(1:NE)-P.xmin(1:NE),eps);
setsOf = @(d) local_sets(P, d, tolB);
refSets = setsOf(dRef);

    function m = metrics(x, lamMMA, xsi, eta, label)
        d = x(1:NE); bs = x(end);
        [fval, dfdx] = P.evalProd(x);
        m = struct('label',label,'bs',bs,'beta',bs*P.lamref,'obj_gap',bsRef-bs,'G',(bsRef-bs)/gainRef, ...
            'dist2',norm(d-dRef),'distInf',max(abs(d-dRef)),'dist2_rel',norm(d-dRef)/norm(dRef), ...
            'cosine',(d.'*dRef)/(norm(d)*norm(dRef)),'max_util',max(abs(d))/move, ...
            'norm2',norm(d),'viol',max(max(fval),0),'fval',fval.', ...
            'image_abc',P.abc(d).','fJJ_drho',P.fJJ.'*d,'sum_drho',sum(d));
        % exact-MMA-dual KKT residual of the true problem (as the prior audit)
        if ~isempty(lamMMA)
            gL = P.f + dfdx.'*lamMMA(:) - xsi(:) + eta(:);
            m.kkt_mma_norm_rms = sqrt(mean(gL(1:NE).^2))/sRow0;
            m.kkt_mma_norm_max = max(abs(gL(1:NE)))/sRow0;
            m.lam_mma = lamMMA(:).';
        else
            m.kkt_mma_norm_rms = NaN; m.kkt_mma_norm_max = NaN; m.lam_mma = nan(1,4);
        end
        % residual under the REFERENCE multipliers (Lagrangian with optimal
        % duals is minimized over the box at the reference; projected residual)
        gR = P.f + dfdx.'*R.muRef(:);
        rp = gR(1:NE); atLo = d <= P.xmin(1:NE)+tolB; atHi = d >= P.xmax(1:NE)-tolB;
        rp(atLo) = min(rp(atLo),0); rp(atHi) = max(rp(atHi),0);
        m.kkt_refdual_proj_norm_rms = sqrt(mean(rp.^2))/sRow0;
        st = setsOf(d);
        m.jaccard = struct('minus_move',local_jac(st.mm,refSets.mm),'plus_move',local_jac(st.pm,refSets.pm), ...
            'floor',local_jac(st.fl,refSets.fl),'ceiling',local_jac(st.ce,refSets.ce),'any_bound',local_jac(st.any,refSets.any), ...
            'same_bound_side',mean((st.side == refSets.side) & refSets.side ~= 0));
        m.counts = struct('minus_move',nnz(st.mm),'plus_move',nnz(st.pm),'floor',nnz(st.fl),'ceiling',nnz(st.ce),'interior',nnz(~st.any));
        m.frac_move_bound = (nnz(st.mm)+nnz(st.pm))/NE;
        m.sign_agreement = mean(sign(d) == sign(dRef));
    end

o = struct();
o.reference = struct('bs',bsRef,'beta',bsRef*P.lamref,'gain',gainRef,'norm2',norm(dRef), ...
    'counts',struct('minus_move',nnz(refSets.mm),'plus_move',nnz(refSets.pm),'floor',nnz(refSets.fl),'ceiling',nnz(refSets.ce),'interior',nnz(~refSets.any)));
% ---- the three named states ---------------------------------------------
ck19 = M.CK([M.CK.iter]==19); ck500 = M.CK([M.CK.iter]==500); ck5000 = M.CK([M.CK.iter]==M.out.nIter);
o.replay = M.out;
o.P19 = metrics(L.xP19, ck19.lam, ck19.xsi, ck19.eta, 'P19 production');
o.M500 = metrics(L.xM500, ck500.lam, ck500.xsi, ck500.eta, 'M500');
o.M5000 = metrics(M.xM5000, ck5000.lam, ck5000.xsi, ck5000.eta, 'M5000');
o.P19_equals_replay19 = isequal(L.xP19, ck19.x);
o.M500_equals_replay500 = isequal(L.xM500, ck500.x);
% ---- checkpoints ----------------------------------------------------------
nck = numel(M.CK); T = struct('iter',{},'bs',{},'obj_gap',{},'G',{},'dist2',{},'dist2_rel',{},'distInf',{},'cosine',{}, ...
    'max_util',{},'viol',{},'kkt_mma',{},'kkt_refdual',{},'relStep',{},'jac_any',{},'same_side',{},'frac_move',{});
for k = 1:nck
    c = M.CK(k); m = metrics(c.x, c.lam, c.xsi, c.eta, sprintf('it%d',c.iter));
    T(end+1) = struct('iter',c.iter,'bs',m.bs,'obj_gap',m.obj_gap,'G',m.G,'dist2',m.dist2,'dist2_rel',m.dist2_rel, ...
        'distInf',m.distInf,'cosine',m.cosine,'max_util',m.max_util,'viol',m.viol,'kkt_mma',m.kkt_mma_norm_rms, ...
        'kkt_refdual',m.kkt_refdual_proj_norm_rms,'relStep',M.H.relStep(c.iter),'jac_any',m.jaccard.any_bound, ...
        'same_side',m.jaccard.same_bound_side,'frac_move',m.frac_move_bound); %#ok<AGROW>
end
o.checkpoints = T;
% per-iteration cheap history from the replay (bs, max util, relStep)
o.hist = struct('iter',1:M.out.nIter,'bs',(M.H.beta(1:M.out.nIter)/P.lamref).','G',((bsRef - M.H.beta(1:M.out.nIter)/P.lamref)/gainRef).', ...
    'max_util',(M.H.maxAbsDrho(1:M.out.nIter)/move).','relStep',M.H.relStep(1:M.out.nIter).','ymax',M.H.ymax(1:M.out.nIter).','zmma',M.H.zmma(1:M.out.nIter).');
% ---- trajectory classification (preregistration sec. 7) -------------------
it = [T.iter]; dist = [T.dist2_rel]; G = [T.G];
d19 = dist(it==19); d5000 = dist(end); G19 = G(it==19); G5000 = G(end);
late = it >= 1000; dl = dist(late); il = it(late);
rho_s = corr(il(:), dl(:), 'type','Spearman');
cls = struct('approaching_objective', G5000 <= 0.1 && G5000 < 0.5*G19, ...
    'approaching_design', d5000 < 0.5*d19 && d5000 <= 0.1, ...
    'moving_away', d5000 > 1.1*d19, ...
    'orbiting', (max(dl)-min(dl)) >= 0.2*mean(dl) && abs(rho_s) < 0.5, ...
    'late_dist_range',[min(dl) max(dl)],'late_dist_mean',mean(dl),'late_spearman',rho_s, ...
    'dist_19',d19,'dist_5000',d5000,'G_19',G19,'G_5000',G5000, ...
    'late_G_range',[min(G(late)) max(G(late))],'G_min_over_all_checkpoints',min(G),'iter_of_G_min',it(G==min(G)));
if cls.approaching_objective && cls.approaching_design, cls.verdict = 'A_approaching';
elseif cls.approaching_objective && ~cls.approaching_design, cls.verdict = 'D_objective_not_primal';
elseif cls.moving_away, cls.verdict = 'B_moving_away';
elseif cls.orbiting, cls.verdict = 'C_orbiting';
else, cls.verdict = 'E_inconclusive'; end
o.trajectory = cls;
% ---- decision logic (preregistration sec. 8) -----------------------------
GP19 = o.P19.G; GM5000 = o.M5000.G; cosM = o.M5000.cosine; fracMove = o.reference.counts;
refMoveFrac = (fracMove.minus_move + fracMove.plus_move)/NE;
if GP19 <= 0.1, rel = 'PRODUCTION_TRUNCATION_NEAR_REFERENCE';
elseif GP19 >= 0.5 && GM5000 <= 0.1 && cosM >= 0.9 && refMoveFrac >= 0.5, rel = 'PRODUCTION_INNER_TRUNCATION_MATERIALLY_PREMATURE';
else, rel = 'REPEATED_MMA_REALIZATION_FAILS_PROBLEM25_REFERENCE'; end
o.decision = struct('G_P19',GP19,'G_M5000',GM5000,'cos_M5000',cosM,'ref_move_bound_frac',refMoveFrac, ...
    'production_truncation_materially_premature', GP19 >= 0.5, 'relationship_verdict', rel, ...
    'case_C_blocked_only_by', {{}});
if GP19 >= 0.5 && GM5000 <= 0.1 && cosM >= 0.9 && refMoveFrac < 0.5
    o.decision.case_C_blocked_only_by = {'ref_move_bound_frac < 0.5'};
end
fid = fopen(fullfile(ev,'mma_comparison.json'),'w');
fprintf(fid,'%s',jsonencode(o,'PrettyPrint',true)); fclose(fid);
% ---- fields for the figures ---------------------------------------------
F = struct('nelx',S.nelx,'nely',S.nely,'move',move,'rho385',P.rho,'drho_ref',dRef,'drho_P19',L.xP19(1:NE), ...
    'drho_M500',L.xM500(1:NE),'drho_M5000',M.xM5000(1:NE),'q_ref',R.qRef(1:NE), ...
    'ref_side',refSets.side,'ref_kind',refSets.kind,'lo',P.xmin(1:NE),'hi',P.xmax(1:NE), ...
    'ck_iter',[T.iter],'ck_G',[T.G],'ck_dist2_rel',[T.dist2_rel],'ck_cosine',[T.cosine],'ck_kkt_mma',[T.kkt_mma], ...
    'ck_kkt_refdual',[T.kkt_refdual],'ck_max_util',[T.max_util],'ck_viol',[T.viol],'ck_bs',[T.bs], ...
    'h_iter',o.hist.iter,'h_G',o.hist.G,'h_max_util',o.hist.max_util,'h_relStep',o.hist.relStep,'h_bs',o.hist.bs, ...
    'bsRef',bsRef,'F11',P.F11,'F22',P.F22,'F12',P.F12,'fJJ',P.fJJ);
save(fullfile(ev,'compare_fields.mat'),'-struct','F','-v7.3');
fprintf('[fp_compare] P19: G=%.4f dist2rel=%.3f cos=%.3f util=%.3f | M500: G=%.4f dist2rel=%.3f cos=%.3f | M5000: G=%.4f dist2rel=%.3f cos=%.3f util=%.3f viol=%.2e\n', ...
    o.P19.G, o.P19.dist2_rel, o.P19.cosine, o.P19.max_util, o.M500.G, o.M500.dist2_rel, o.M500.cosine, o.M5000.G, o.M5000.dist2_rel, o.M5000.cosine, o.M5000.max_util, o.M5000.viol);
fprintf('  trajectory: %s (G19=%.3f G5000=%.3f d19=%.3f d5000=%.3f late range [%.3f %.3f] spearman %.2f)\n', cls.verdict, G19, G5000, d19, d5000, cls.late_dist_range, rho_s);
fprintf('  relationship verdict: %s\n', rel);
out = o;
end
function st = local_sets(P, d, tolB)
NE = P.NE; atLo = d <= P.xmin(1:NE)+tolB; atHi = d >= P.xmax(1:NE)-tolB;
st.mm = atLo & P.loMoveLimited; st.fl = atLo & ~P.loMoveLimited;
st.pm = atHi & P.hiMoveLimited; st.ce = atHi & ~P.hiMoveLimited; st.any = atLo | atHi;
st.side = zeros(NE,1); st.side(atLo) = -1; st.side(atHi) = 1;
st.kind = zeros(NE,1); st.kind(st.mm) = -2; st.kind(st.fl) = -1; st.kind(st.pm) = 2; st.kind(st.ce) = 1;
end
function j = local_jac(a, b)
u = nnz(a | b); if u == 0, j = 1; else, j = nnz(a & b)/u; end
end
