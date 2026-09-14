function fi_oracle()
[S,P,L,R,E]=fi_setup();
old=jsondecode(fileread(fullfile(S.study,'evaluations','state_identity.json')));
o=struct();
o.rho386=fp_hash(S.rho386); o.rho385=fp_hash(S.rho385);
o.drho386=fp_hash(S.drho386);
assert(strcmp(o.rho386,old.rho386_sha256));
assert(strcmp(o.rho385,old.rho385_sha256));
assert(strcmp(o.drho386,old.drho386_sha256));
o.cfg=olhoffcurrent_config_hash(S.cfg); assert(strcmp(o.cfg,S.cfgHash));
sm=olhoffcurrent_source_manifest('Verify',true);
o.implTree=sm.treeHash; assert(sm.ok && strcmp(sm.treeHash,S.implTree));
assert(S.nelx==480 && S.nely==60 && S.nOuter==386 && S.hist.stage(end)==3);
assert(P.move==.01 && isequal(L.ctx.dOff,L.ctx.lam-L.ctx.lam(1)));
assert(isequal(L.xP19(1:P.NE),S.drho386));
o.state=struct('mesh',[480 60],'outer',386,'stage',3,'move',P.move,'N',P.N);
o.xRef_sha256=fp_hash(R.xRef);
ref=jsondecode(fileread(fullfile(S.study,'evaluations','reference_solution.json')));
D=ref.certificate_aligned; p=D.p(:); mu=D.mu; nu=D.nu(:);
q=P.f+P.Ac.'*p-mu*P.dc+P.Alin.'*nu;
bound=sum(min(q.*P.xmin,q.*P.xmax))-p.'*P.bc+mu*P.gammac-nu.'*P.blin;
o.dual_bound=bound; o.certified_gap=P.f.'*R.xRef-bound;
o.dual_feasible=norm(p)<=mu+1e-14 && all(nu>=0);
K=fp_kkt(P,R.xRef,R.muRef,R.xsiRef,R.etaRef,'authenticated oracle');
o.kkt=rmfield(K,'masks'); o.beta=R.xRef(end)*P.lamref; o.bs=R.xRef(end);
[c,J]=P.evalProd(R.xRef); o.constraints=c.';
assert(strcmp(K.verdict,'REFERENCE_PROBLEM25_KKT_PASS'));
assert(o.dual_feasible && o.certified_gap>=-1e-12 && o.certified_gap<=1e-8);
rng(250386,'twister'); X=[[zeros(P.NE,1);1],R.xRef,L.xP19,L.xM500];
for k=1:20, X(:,end+1)=P.xmin+rand(P.nvar,1).*(P.xmax-P.xmin); end
v=zeros(size(X,2),1); g=v; reduction=v;
for k=1:size(X,2)
 [ck,jk]=P.evalProd(X(:,k));
 v(k)=abs(ck(1)-P.coneResid(X(:,k)));
 g(k)=norm(jk(1,:).'-P.coneGrad(X(:,k)),inf)/max(norm(jk(1,:),inf),eps);
 reduction(k)=max(ck(2)-ck(1),0);
 assert(max(abs(ck(3:4)-(P.Alin*X(:,k)-P.blin)))<1e-12);
end
o.equivalence=struct('points',size(X,2),'max_value_error',max(v),'max_gradient_relerr',max(g),'max_redundancy_violation',max(reduction));
assert(max(v)<1e-12 && max(g)<1e-10 && max(reduction)<1e-12);
o.zero_density_updates=0; o.topology_runs=0;
o.verdict='FROZEN_SOLVER_ORACLE_PASS';
fi_json(fullfile(E,'oracle_identity.json'),o);
fprintf('%s: beta=%.12f gap=%.3e KKT=%.3e\n',o.verdict,o.beta,o.certified_gap,K.complementarity.max_box_comp_normalized);
end
