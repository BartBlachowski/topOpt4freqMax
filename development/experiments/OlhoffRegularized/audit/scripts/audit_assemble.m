function [K,M]=audit_assemble(mdl,rho,p,massInterp)
%AUDIT_ASSEMBLE  Independent assembly incl. the concentrated tip mass.
rho=rho(:);
sK=mdl.K0(:)*(rho.^p)';
gm=massScale(rho,massInterp);
sM=mdl.M0(:)*gm';
K=sparse(mdl.iK,mdl.jK,sK(:),mdl.ndof,mdl.ndof);
M=sparse(mdl.iK,mdl.jK,sM(:),mdl.ndof,mdl.ndof);
K=(K+K')/2;M=(M+M')/2;
f=mdl.free;K=K(f,f);M=M(f,f);
if mdl.tipMassValue>0
    [ok,red]=ismember(mdl.tipMassDofs,mdl.free);
    assert(all(ok),'tip mass DOF constrained');
    nF=numel(mdl.free);
    M=M+sparse(red,red,mdl.tipMassValue*ones(numel(red),1),nF,nF);
    M=(M+M')/2;
end
end
