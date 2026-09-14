function out = fp_lp_insight()
%FP_LP_INSIGHT  Interpretation aid (not a verdict input).  The certificate's
%   cone direction w = [-1, -2.8e-5] says the cluster cone acts, at this
%   state, almost exactly like the FIRST-MODE linear functional a = F11'drho.
%   Test: solve the LP  max F11'drho  s.t. volume and the production box
%   (linprog, dual-simplex), evaluate that drho through the production
%   deltaLambda, and compare with the certified reference.  drho discarded.
S = fp_setup();                                   %#ok<NASGU>
ev = fullfile(S.study,'evaluations');
L = load(fullfile(ev,'frozen_ctx.mat'),'ctx'); P = fp_problem(L.ctx);
R = load(fullfile(ev,'conic_reference.mat'),'xRef'); NE = P.NE;
opts = optimoptions('linprog','Display','off','Algorithm','dual-simplex','OptimalityTolerance',1e-10,'ConstraintTolerance',1e-10);
[d, ~, ef] = linprog(-P.F11, ones(1,NE), P.Vtot - sum(P.rho), [], [], P.xmin(1:NE), P.xmax(1:NE), opts);
e1 = P.e1closed(d); bsLP = (P.lam(1) + e1)/P.lamref;
xLP = [d; bsLP]; fval = P.evalProd(xLP);
dRef = R.xRef(1:NE);
tolB = 1e-6*(P.xmax(1:NE)-P.xmin(1:NE));
atB = @(v) (v <= P.xmin(1:NE)+tolB) | (v >= P.xmax(1:NE)-tolB);
side = @(v) (v >= P.xmax(1:NE)-tolB) - (v <= P.xmin(1:NE)+tolB);
out = struct('linprog_exitflag',ef,'bs_LP',bsLP,'bs_ref',R.xRef(end),'bs_LP_minus_ref',bsLP-R.xRef(end), ...
    'G_LP',(R.xRef(end)-bsLP)/(R.xRef(end)-1),'max_fval_LP',max(fval),'fval_LP',fval.', ...
    'dist2',norm(d-dRef),'dist2_rel',norm(d-dRef)/norm(dRef),'distInf_over_move',max(abs(d-dRef))/P.move, ...
    'cosine',(d.'*dRef)/(norm(d)*norm(dRef)),'same_bound_side_frac',mean(side(d)==side(dRef)), ...
    'n_at_bound_LP',nnz(atB(d)),'n_interior_LP',nnz(~atB(d)),'image_abc_LP',P.abc(d).','image_abc_ref',P.abc(dRef).', ...
    'threshold_F11_over_lamref', 0.7291136447250609/P.Vtot, ...
    'note','LP = max F11''drho s.t. volume + box; evaluated through production deltaLambda; interpretation only');
fid = fopen(fullfile(ev,'lp_insight.json'),'w'); fprintf(fid,'%s',jsonencode(out,'PrettyPrint',true)); fclose(fid);
fprintf('[fp_lp_insight] ef=%d bs_LP=%.12f ref=%.12f diff=%.3e G_LP=%.3e dist2rel=%.4f cos=%.6f sameside=%.5f interiorLP=%d\n', ...
    ef, bsLP, R.xRef(end), bsLP-R.xRef(end), out.G_LP, out.dist2_rel, out.cosine, out.same_bound_side_frac, out.n_interior_LP);
end
