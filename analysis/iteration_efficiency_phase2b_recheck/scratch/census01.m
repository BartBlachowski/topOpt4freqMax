repo='/Users/piotrek/Programming/topOpt4freqMax';
meshes={'160x20','240x30','320x40','400x50','480x60','560x70','640x80','720x90'};
s01=single(0.1); fprintf('single(0.1)=%.20g  double 0.1=%.20g\n',double(s01),0.1);
fprintf('%-9s %7s %9s %12s %12s %10s %10s\n','mesh','states','nel','atrisk_elem','atrisk_states','maxper','fracstates');
for i=1:numel(meshes)
  f=fullfile(repo,'examples','Performance','final_campaign','raw','olhoff',['s1_' meshes{i} '.mat']);
  S=load(f,'res'); X=S.res.rho_snapshots; [ne,ns]=size(X);
  hit = (X==s01);                       % single equals single(0.1) exactly
  perState = sum(hit,1);
  nStatesHit = nnz(perState>0);
  fprintf('%-9s %7d %9d %12d %13d %10d %10.4f\n', meshes{i}, ns, ne, sum(perState), nStatesHit, max(perState), nStatesHit/ns);
  clear S X hit
end
