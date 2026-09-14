function out = fp_gradcheck()
%FP_GRADCHECK  Central-difference check of the production constraint
%   gradients at S1 along 5 seeded directions, swept over the step so that the
%   preregistered step (1e-6*move = 1e-8) can be judged against roundoff.
S = fp_setup();                                   %#ok<NASGU>
ev = fullfile(S.study,'evaluations');
L = load(fullfile(ev,'frozen_ctx.mat'),'ctx','xP19'); P = fp_problem(L.ctx);
nvar = P.nvar; xg = L.xP19;
steps = [1e-8 1e-7 1e-6 1e-5 1e-4 1e-3];
rng(20260912,'twister'); V = randn(nvar,5); V = V./vecnorm(V);
[~, dfdx] = P.evalProd(xg);
E = zeros(numel(steps), P.m); A = zeros(numel(steps), P.m);
for si = 1:numel(steps)
    h = steps(si); errs = zeros(5,P.m); abse = zeros(5,P.m);
    for k = 1:5
        v = V(:,k);
        fp_ = P.evalProd(xg + h*v); fm_ = P.evalProd(xg - h*v);
        fd = (fp_ - fm_)/(2*h); an = dfdx*v;
        errs(k,:) = (abs(fd - an)./max(abs(an),1e-300)).'; abse(k,:) = abs(fd-an).';
    end
    E(si,:) = max(errs,[],1); A(si,:) = max(abse,[],1);
end
out = struct('steps',steps,'max_relerr_rows',E,'max_abserr_rows',A, ...
    'preregistered_step',1e-8,'pass_at_preregistered_step',all(E(1,:) <= 1e-5), ...
    'best_step_per_row',steps(arrayfun(@(j) find(E(:,j)==min(E(:,j)),1), 1:P.m)), ...
    'min_relerr_per_row',min(E,[],1),'pass_at_best_step',all(min(E,[],1) <= 1e-5), ...
    'note','rows 3-4 are exactly linear; FD error there is pure roundoff/h');
fid = fopen(fullfile(ev,'gradient_check.json'),'w');
fprintf(fid,'%s',jsonencode(out,'PrettyPrint',true)); fclose(fid);
fprintf('[fp_gradcheck] step sweep (rows = steps, cols = constraint rows), max relerr:\n');
disp([steps.' E]);
fprintf('  pass at preregistered 1e-8: %d ; pass at best step: %d (min per row %s)\n', out.pass_at_preregistered_step, out.pass_at_best_step, mat2str(out.min_relerr_per_row,3));
end
