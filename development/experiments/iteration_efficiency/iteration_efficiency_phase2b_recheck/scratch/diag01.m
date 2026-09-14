repo='/Users/piotrek/Programming/topOpt4freqMax';
S=load(fullfile(repo,'analysis','iteration_efficiency_phase2b_precision','qualification_runs','gray_full_24x4_h200_paired_states.mat'));
k=find(S.pairIterations==114);
xd=double(S.x_double(:,k)); xs=double(S.x_single(:,k));
fprintf('n=%d  max|dx|=%.6e\n',numel(xd),max(abs(xd-xs)));
% branch boundary of the E2/E3 mass law: g = x^6 for x<=0.1 else x
bd = xd<=0.1;  bs = xs<=0.1;
fprintf('elements on x<=0.1 branch: double=%d single=%d  FLIPPED=%d\n',nnz(bd),nnz(bs),nnz(bd~=bs));
idx=find(bd~=bs);
for i=1:numel(idx)
  j=idx(i);
  fprintf('  el %d: xd=%.20g  xs=%.20g   xd<=0.1:%d  xs<=0.1:%d\n',j,xd(j),xs(j),xd(j)<=0.1,xs(j)<=0.1);
  fprintf('        g_double=%.6e  g_single=%.6e  ratio=%.4g\n', ...
     (xd(j)<=0.1)*xd(j)^6 + (xd(j)>0.1)*xd(j), (xs(j)<=0.1)*xs(j)^6 + (xs(j)>0.1)*xs(j), ...
     ((xs(j)<=0.1)*xs(j)^6 + (xs(j)>0.1)*xs(j)) / ((xd(j)<=0.1)*xd(j)^6 + (xd(j)>0.1)*xd(j)));
end
fprintf('exact 0.1 in double: %.20g\nsingle(0.1) as double: %.20g\n',0.1,double(single(0.1)));
fprintf('count xd exactly == 0.1 : %d\n',nnz(xd==0.1));
% E3 stiffness floor boundary max(x,1e-3)
fprintf('elements at/below rhomin 1e-3: double=%d single=%d flipped=%d\n', nnz(xd<=1e-3),nnz(xs<=1e-3),nnz((xd<=1e-3)~=(xs<=1e-3)));
