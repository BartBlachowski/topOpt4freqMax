function fi_structure()
[S,P,L,R,E]=fi_setup();N=P.NE;z=R.xRef(1:N);q=R.qRef(1:N);
lo=P.xmin(1:N);hi=P.xmax(1:N);w=hi-lo;tol=1e-6*w;
atlo=z<=lo+tol;athi=z>=hi-tol;
floorRaw=P.rhomin-P.rho;ceilRaw=1-P.rho;
lowerDensity=floorRaw>-P.move;upperDensity=ceilRaw<P.move;
coLo=abs(floorRaw+P.move)<=1e-14;coHi=abs(ceilRaw-P.move)<=1e-14;
category=zeros(N,1);category(atlo&lowerDensity)=1;category(atlo&~lowerDensity)=2;
category(athi&~upperDensity)=3;category(athi&upperDensity)=4;category((atlo&coLo)|(athi&coHi))=5;
names={'interior','lower density','lower move','upper move','upper density','coincident'};
o=struct();o.bound=struct('names',{names},'counts',arrayfun(@(v)nnz(category==v),0:5),'any_fraction',mean(category~=0),'move_fraction',mean(category==2|category==3));
[~,J]=P.evalProd(R.xRef);gEff=-J(1,1:N).';sens=P.F11/P.lamref;
mu=R.muRef;threshold=mu(4)/P.Vtot;
o.threshold=struct('mu',mu.','volume_threshold',threshold,'effective_formula','q=-mu1*gEffective-mu2*gSecond-mu3*fJJ/lamref+mu4/Vtot','effective_vs_F11_rms',norm(gEff-sens)/norm(sens),'q_positive',nnz(q>0),'q_negative',nnz(q<0),'q_nearzero_rel1e6',nnz(abs(q)<=1e-6*max(abs(q))),'lower_wrong_cost',nnz(atlo&q<0),'upper_wrong_cost',nnz(athi&q>0),'cone_direction',((P.Ac*R.xRef-P.bc)/norm(P.Ac*R.xRef-P.bc)).');
% Exact greedy solution of the first-mode linear interpretation, not a
% competing solver for problem (25), and never applied to rho.
d=lo;remaining=P.Vtot-sum(P.rho)-sum(lo);[~,order]=sort(sens,'descend');
for k=1:N
 e=order(k);if sens(e)<=0||remaining<=0,break,end
 add=min(w(e),remaining);d(e)=d(e)+add;remaining=remaining-add;
end
xx=[d;1];c=P.evalProd(xx);xx(end)=1-max(c(1:3));
c=P.evalProd(xx);o.first_mode_rule=struct('bs',xx(end),'gainLoss',(R.xRef(end)-xx(end))/(R.xRef(end)-1),'d2',norm(d-z)/norm(z),'dinf',norm(d-z,inf)/P.move,'maxConstraint',max(c),'allSignAgreement',mean(sign(d)==sign(z)));
raw=L.Fraw(:,1,1)/P.lamref;
classes={P.rho<=.1, P.rho>.1&P.rho<.9&~(P.rho>=.4&P.rho<=.6),P.rho>=.4&P.rho<=.6,P.rho>=.9};
cnames={'void','gray_shell','gray_core','solid'};densityClass=zeros(N,1);
prior=load(fullfile(S.study,'evaluations','mma_replay.mat'),'CK');
loc=[];maps=[];
for iter=[19 500 5000]
 ck=prior.CK([prior.CK.iter]==iter);a=ck.x(1:N);bx=zeros(N,1);bx(a<=lo+tol)=-1;bx(a>=hi-tol)=1;bo=double(athi)-double(atlo);
 [cv,jv]=P.evalProd(ck.x);qp=P.f+jv.'*ck.lam;rawK=qp-ck.xsi+ck.eta;
 totaldist=sum((a-z).^2);allErrors=nnz(sign(a)~=sign(z));large=abs(z)>=.9*P.move;
 classRows=[];
 for j=1:4
  id=classes{j};densityClass(id)=j;li=id&large;ib=id&(bo~=0);
  contrib=q(id).*(a(id)-z(id));
  comp=max(qp(1:N),0).*(a-lo)+max(-qp(1:N),0).*(hi-a);
  amp=abs(sens(id))./max(abs(raw(id)),eps);dis=abs(a(id)-z(id));
  rr=struct('name',cnames{j},'n',nnz(id),'sign_error',mean(sign(a(id))~=sign(z(id))),'large_sign_error',mean(sign(a(li))~=sign(z(li))),'share_all_sign_errors',nnz(id&(sign(a)~=sign(z)))/max(allErrors,1),'bound_error',mean(bx(ib)~=bo(ib)),'distance2',norm(a(id)-z(id)),'distance2_normalized',norm(a(id)-z(id))/norm(z(id)),'share_squared_distance',sum((a(id)-z(id)).^2)/totaldist,'reduced_cost_contribution',sum(contrib),'first_mode_linear_loss',sens(id).'*(z(id)-a(id)),'kkt_raw_rms',sqrt(mean(rawK(find(id)).^2))/sqrt(mean(sens.^2)),'box_complementarity_rms',sqrt(mean(comp(id).^2))/(sqrt(mean(sens.^2))*P.move),'sensitivity_rms_amplification',norm(sens(id))/norm(raw(id)),'amp_disagreement_spearman',corr(amp,dis,'Type','Spearman'));
  if isempty(classRows),classRows=rr;else,classRows(end+1)=rr;end
 end
 r=struct('iter',iter,'classes',classRows,'constraint',cv.');
 if isempty(loc),loc=r;maps=struct('iter',iter,'drho',a,'signError',sign(a)~=sign(z),'boundError',bx~=bo);else,loc(end+1)=r;maps(end+1)=struct('iter',iter,'drho',a,'signError',sign(a)~=sign(z),'boundError',bx~=bo);end
end
o.localization=loc;
fi_json(fullfile(E,'structure.json'),o);
rho=P.rho;oracle=z;reducedCost=q;filteredSensitivity=sens;rawSensitivity=raw;effectiveSensitivity=gEff;boundCategory=category;
save(fullfile(E,'structure.mat'),'rho','oracle','reducedCost','filteredSensitivity','rawSensitivity','effectiveSensitivity','threshold','boundCategory','densityClass','maps','xx','-v7.3');
end
