function D = cs_preflight_degeneracy()
%CS_PREFLIGHT_DEGENERACY  PRE-LAUNCH diagnostic of the P4 failure.  Frozen
%   C480 problem only (the reference study's hashed ctx); no density update,
%   no treatment state.  Questions:
%   (1) where do the augmented and oracle (schur) solutions differ, and what are
%       the certified reduced costs there (degenerate optimal face?);
%   (2) does schur, single-threaded, reproduce the oracle bitwise, and at what cost;
%   (3) does schur with more BLAS threads reproduce it, and at what cost.
S = cs_setup();
guard = olhoffcurrent_paths(); %#ok<NASGU>
maxNumCompThreads(1);
FC = load(S.frozenCtx, 'ctx'); ctx = FC.ctx;
RF = load(S.conicRef, 'xRef'); xRef = RF.xRef;
P = fp_problem(ctx); NE = P.NE; mv = ctx.move;
soc = secondordercone(P.Ac, P.bc, P.dc, P.gammac);
D = struct();
runs = {'augmented',1; 'schur',1; 'schur',4; 'schur',8; 'augmented',8};
X = cell(size(runs,1),1);
for i = 1:size(runs,1)
    maxNumCompThreads(runs{i,2});
    opts = optimoptions('coneprog','Display','off','OptimalityTolerance',1e-10, ...
        'ConstraintTolerance',1e-10,'MaxIterations',500,'LinearSolver',runs{i,1});
    t = tic; [x,~,ef,op,la] = coneprog(P.f, soc, P.Alin, P.blin, [], [], P.xmin, P.xmax, opts); w = toc(t);
    maxNumCompThreads(1);
    [C, xc] = cs_socp_certify(P, x, la);
    X{i} = xc;
    z = xc(1:NE); zo = xRef(1:NE);
    r = struct('solver',runs{i,1},'threads',runs{i,2},'wall_s',w,'exitflag',ef,'iterations',op.iterations, ...
        'bitwise_oracle',isequal(x,xRef),'abs_dbs',abs(xc(end)-xRef(end)),'d2',norm(z-zo)/norm(zo), ...
        'dinf',norm(z-zo,inf)/mv,'n_diff_gt_0p1move',nnz(abs(z-zo) > 0.1*mv), ...
        'n_diff_gt_1e6move',nnz(abs(z-zo) > 1e-6*mv),'certified',C.certified,'candidate',C.candidate, ...
        'bestGap',C.bestGap,'nInterior',nnz(z > P.xmin(1:NE)+1e-6*(P.xmax(1:NE)-P.xmin(1:NE)) & z < P.xmax(1:NE)-1e-6*(P.xmax(1:NE)-P.xmin(1:NE))));
    if C.certified
        % certified reduced cost at the elements that differ from the oracle
        qK = P.f + P.Ac.'*C.p - C.mu*P.dc + P.Alin.'*C.nu;
        idx = find(abs(z-zo) > 0.1*mv);
        s0 = C.sRow0;
        r.diff_elements_abs_q_over_sRow0 = abs(qK(idx)).'/s0;
        r.diff_elements_rho = ctx.rho(idx).';
        r.diff_elements_z = z(idx).'/mv; r.diff_elements_zo = zo(idx).'/mv;
        r.all_abs_q_over_sRow0_quantiles = quantile(abs(qK(1:NE))/s0, [0 1e-3 1e-2 0.1 0.5]);
        r.n_abs_q_lt_1e_6_sRow0 = nnz(abs(qK(1:NE)) < 1e-6*s0);
        r.n_abs_q_lt_1e_4_sRow0 = nnz(abs(qK(1:NE)) < 1e-4*s0);
    end
    D.runs(i) = r; %#ok<AGROW>
    fprintf('%s x%d: wall %.2f ef %d it %d bitwise %d dbs %.2e d2 %.4f dinf %.3f n>0.1mv %d cert %d gap %.2e\n', ...
        r.solver, r.threads, w, ef, op.iterations, r.bitwise_oracle, r.abs_dbs, r.d2, r.dinf, ...
        r.n_diff_gt_0p1move, r.certified, r.bestGap);
end
% the change in objective along the oracle->augmented segment certifies flatness
za = X{1}; D.segment = struct();
tt = linspace(0,1,11); vals = zeros(size(tt)); feas = zeros(size(tt));
for k = 1:numel(tt)
    xk = (1-tt(k))*xRef + tt(k)*za;
    fv = P.evalProd(xk); vals(k) = xk(end); feas(k) = max(fv);
end
D.segment.t = tt; D.segment.bs = vals; D.segment.maxRow = feas;
cs_json(fullfile(S.study,'evaluations','preflight_degeneracy.json'), D);
end
