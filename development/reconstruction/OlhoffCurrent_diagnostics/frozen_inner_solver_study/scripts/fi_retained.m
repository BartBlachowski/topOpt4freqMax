function fi_retained()
[S,P,L,R,E]=fi_setup();A=load(fullfile(S.study,'evaluations','mma_replay.mat'));
H=[];CK=[];
for k=1:numel(A.CK)
 c=A.CK(k);[M,K]=fi_metric(P,R,c.x,c.lam,c.xsi,c.eta);
 M.iter=c.iter;M.calls=c.iter;M.nonlinearEvaluations=c.iter;M.gradientEvaluations=c.iter;
 M.relStep=A.H.relStep(c.iter);M.dx=A.H.dx(c.iter);
 M.solverWall=NaN;M.wall=NaN;
 if ~isempty(c.low)
  % Saved asymptotes are around the PRE-step iterate, unavailable at most
  % checkpoints. Report their width, not an invented distance to that iterate.
  M.asyWidthMin=min((c.upp-c.low)./(P.xmax-P.xmin));
  M.asyWidthMedian=median((c.upp-c.low)./(P.xmax-P.xmin));
  M.asyWidthMax=max((c.upp-c.low)./(P.xmax-P.xmin));
 else,M.asyWidthMin=NaN;M.asyWidthMedian=NaN;M.asyWidthMax=NaN;end
 ck=struct('iter',c.iter,'x',c.x,'lam',c.lam,'xsi',c.xsi,'eta',c.eta,'metric',M,'kkt',rmfield(K,'masks'));
 if isempty(H),H=M;CK=ck;else,H(end+1)=M;CK(end+1)=ck;end
end
out=struct('name','B0_RETAINED_5000','source',fullfile(S.study,'evaluations','mma_replay.mat'),'fresh_solver_calls',0,'prior_wall_s',A.out.wall_s,'retained_checkpoints',numel(H),'bitwise19',isequal(CK([CK.iter]==19).x,L.xP19),'bitwise500',isequal(CK([CK.iter]==500).x,L.xM500));
assert(out.bitwise19 && out.bitwise500);
fi_json(fullfile(E,'B0_RETAINED_5000.json'),struct('out',out,'history',H,'all_relStep',A.H.relStep,'all_beta',A.H.beta));
save(fullfile(E,'B0_RETAINED_5000.mat'),'CK','H','out','-v7.3');
end
