function hist_summary(matfile)
S=load(matfile); res=S.res; h=res.hist;
n=res.nOuter;
fprintf('%s : outer=%d\n',matfile,n);
fprintf('N history: '); u=unique(h.N); for k=u, fprintf('N=%d:%d iters  ',k,sum(h.N==k)); end; fprintf('\n');
first2=find(h.N>=2,1);
if isempty(first2), fprintf('N never reached 2\n'); else
  fprintf('first N>=2 at iter %d; N>=2 in %d of the last 50 iters\n',first2,sum(h.N(max(1,n-49):n)>=2));
end
g=(h.omega(2,:)-h.omega(1,:))./h.omega(1,:);
fprintf('gap (w2-w1)/w1: start %.3f  min %.4f (iter %d)  final %.4f\n',g(1),min(g),find(g==min(g),1),g(end));
fprintf('omega1: start %.2f  max %.2f (iter %d)  final %.2f\n',h.omega(1,1),max(h.omega(1,:)),find(h.omega(1,:)==max(h.omega(1,:)),1),h.omega(1,end));
k=max(1,n-9):n;
fprintf('last 10 iters  w1: '); fprintf('%.1f ',h.omega(1,k)); fprintf('\n');
fprintf('last 10 iters  w2: '); fprintf('%.1f ',h.omega(2,k)); fprintf('\n');
fprintf('last 10 maxdrho: '); fprintf('%.4f ',h.dxOuter(k)); fprintf('\n');
end
