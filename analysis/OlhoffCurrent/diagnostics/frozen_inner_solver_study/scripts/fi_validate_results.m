function fi_validate_results()
[S,P,L,R,E]=fi_setup();
names={'B0_CURRENT_REPEATED_MMA','S1_PERSISTENT','S2_ASYINIT_001','S3_CANONICAL_CLAMP','S4_SUBSOLV_ACCURACY','S5_UNIT_BOX','G0_UNSAFE','G1_GCMMA','G2_GCMMA_ACCURATE','S34_CLAMP_ACCURACY','B0_RETAINED_5000','SOCP_COST'};
o=struct('points',0,'max_scalar_discrepancy',0,'max_conic_discrepancy',0,'max_box_violation',0);checks=[];
for k=1:numel(names)
 A=load(fullfile(E,[names{k} '.mat']),'CK');mx=0;mc=0;
 for j=1:numel(A.CK)
  c=A.CK(j);assert(all(isfinite(c.x)));
  [v,J]=P.evalProd(c.x);[M,~]=fi_metric(P,R,c.x,c.lam,c.xsi,c.eta);
  fields={'bs','d2','dinf','gainRecovery','signAgreement','boundAgreement','kkt','constraintResidual'};
  for f=1:numel(fields),mx=max(mx,abs(M.(fields{f})-c.metric.(fields{f})));end
  mc=max(mc,abs(v(1)-P.coneResid(c.x)));
  assert(M.boxViolation<=1e-10 && mc<1e-12 && mx<1e-10);
  assert(norm(J(1,:).'-P.coneGrad(c.x),inf)/max(norm(J(1,:),inf),eps)<1e-10);
  o.points=o.points+1;o.max_box_violation=max(o.max_box_violation,M.boxViolation);
 end
 r=struct('method',names{k},'checkpoints',numel(A.CK),'max_scalar_discrepancy',mx,'max_conic_discrepancy',mc);
 if isempty(checks),checks=r;else,checks(end+1)=r;end
 o.max_scalar_discrepancy=max(o.max_scalar_discrepancy,mx);o.max_conic_discrepancy=max(o.max_conic_discrepancy,mc);
end
A=load(fullfile(E,'G0_UNSAFE.mat'),'CK');B=load(fullfile(E,'G1_GCMMA.mat'),'CK');
o.g0_g1_bitwise=isequal({A.CK.x},{B.CK.x}) && isequal({A.CK.lam},{B.CK.lam});assert(o.g0_g1_bitwise);
o.rho_hash_after=fp_hash(P.rho);assert(strcmp(o.rho_hash_after,fp_hash(S.rho385)));
sm=olhoffcurrent_source_manifest('Verify',true);assert(sm.ok&&strcmp(sm.treeHash,S.implTree));
o.implTree_after=sm.treeHash;o.checks=checks;o.verdict='FROZEN_VARIANTS_EXACT_PROBLEM_AND_METRICS_PASS';
fi_json(fullfile(E,'results_validation.json'),o);disp(rmfield(o,'checks'));
end
