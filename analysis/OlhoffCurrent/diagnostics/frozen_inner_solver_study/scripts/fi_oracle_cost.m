function fi_oracle_cost()
done=fullfile(fileparts(fileparts(mfilename('fullpath'))),'evaluations','SOCP_COST.json');
if isfile(done),fprintf('SOCP cost replicates already complete; reusing measured audit results.\n');return,end
[S,P,L,R,E]=fi_setup();O=[];CK=[];
for j=1:2
 t=tic;P=fp_problem(L.ctx);soc=secondordercone(P.Ac,P.bc,P.dc,P.gammac);assembly=toc(t);
 opts=optimoptions('coneprog','Display','off','OptimalityTolerance',1e-10,'ConstraintTolerance',1e-10,'MaxIterations',500,'LinearSolver','schur');
 t=tic;[x,fv,ef,op,la]=coneprog(P.f,soc,P.Alin,P.blin,[],[],P.xmin,P.xmax,opts);wall=toc(t);
 t=tic;D=fp_dualbound(P,x,max(la.soc(1),0),la.ineqlin);certWall=toc(t);
 p=D.aligned.p(:);mu=D.aligned.mu;nu=D.aligned.nu(:);
 q=P.f+P.Ac.'*p-mu*P.dc+P.Alin.'*nu;
 m=[mu;0;nu];xi=max(q,0);et=max(-q,0);
 [M,K]=fi_metric(P,R,x,m,xi,et);
 M.iter=op.iterations;M.calls=0;M.nonlinearEvaluations=0;M.gradientEvaluations=0;M.solverWall=wall;
 Ks=fp_kkt(P,x,[max(la.soc(1),0);0;la.ineqlin],la.lower,la.upper,'coneprog returned duals');
 a=whos('P','soc');bytes=sum([a.bytes]);
 o=struct('replicate',j,'assembly_s',assembly,'solve_s',wall,'certification_s',certWall,'iterations',op.iterations,'exitflag',ef,'output',op,'assembly_object_bytes',bytes,'certificate',D,'metric',M,'kkt',rmfield(K,'masks'),'solverDualKkt',rmfield(Ks,'masks'),'bitwise_oracle',isequal(x,R.xRef),'x_sha256',fp_hash(x));
 ck=struct('iter',op.iterations,'x',x,'lam',m,'xsi',xi,'eta',et,'metric',M,'kkt',rmfield(K,'masks'));
 if isempty(O),O=o;CK=ck;else,O(end+1)=o;CK(end+1)=ck;end
 fprintf('SOCP repeat %d: assembly %.3fs solve %.3fs cert %.3fs gap %.3e fidelity %d bitwise %d\n',j,assembly,wall,certWall,D.aligned.gap,M.fidelity,o.bitwise_oracle);
end
fi_json(fullfile(E,'SOCP_COST.json'),O);save(fullfile(E,'SOCP_COST.mat'),'CK','O','-v7.3');
end
