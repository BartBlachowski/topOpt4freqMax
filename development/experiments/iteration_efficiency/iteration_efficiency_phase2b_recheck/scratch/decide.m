repo='/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(repo,'analysis','iteration_efficiency_phase2b_recheck'));
[p,guard]=ie2br.setup_paths(); %#ok<ASGLU>
S=load(fullfile(p.runs,'probe_96x12_H3200.mat'));
Qlo=S.Qlo; Qhi=S.Qhi; H0=S.H0; levels=[.98 .99 .995];
refLo=ie2a.reference_phase(Qlo,H0); refHi=ie2a.reference_phase(Qhi,H0);
fprintf('reference  SINGLE(Q_lo): status=%s b_ref=%s Q_ref=%s\n',refLo.status,mat2str(refLo.b_ref),mat2str(refLo.Q_ref,8));
fprintf('reference  UPPER (Q_hi): status=%s b_ref=%s Q_ref=%s\n',refHi.status,mat2str(refHi.b_ref),mat2str(refHi.Q_ref,8));
B0=3200; P=100; BRef=3200;
mbLo=ie2a.measurement_budget(B0,refLo.b_ref,P,BRef); mbHi=ie2a.measurement_budget(B0,refHi.b_ref,P,BRef);
fprintf('B_meas SINGLE=%d tail_trunc=%d | UPPER=%d tail_trunc=%d\n',mbLo.B_meas,mbLo.certification_tail_truncated,mbHi.B_meas,mbHi.certification_tail_truncated);
n=size(Qlo,1);
accept=@(Q,qref) deal_(Q,qref,H0,levels);
[passLo,robLo]=deal_(Qlo,refLo.Q_ref,H0,levels);
[passHi,robHi]=deal_(Qhi,refHi.Q_ref,H0,levels);
pLo=ie2a.scan_persistence(passLo,P); pHi=ie2a.scan_persistence(passHi,P);
fprintf('\n q       k_enter_SINGLE k_cert_SINGLE | k_enter_UPPER k_cert_UPPER | identical\n');
for j=1:3
  fprintf('%.3f        %8s     %8s   |   %8s   %8s   |   %d\n',levels(j), ...
    num2str(pLo.k_enter(j)),num2str(pLo.k_cert(j)),num2str(pHi.k_enter(j)),num2str(pHi.k_cert(j)), ...
    isequaln(pLo.k_enter(j),pHi.k_enter(j))&&isequaln(pLo.k_cert(j),pHi.k_cert(j)));
end
fprintf('\nper-state acceptance disagreement counts (SINGLE vs UPPER bracket):\n');
for j=1:3, fprintf('  q=%.3f : %d of %d states differ\n',levels(j),nnz(passLo(:,j)~=passHi(:,j)),n); end
% margin analysis
fprintf('\nrobust ratio: min margin to each threshold and max bracket perturbation\n');
dRob=robHi-robLo;
fprintf('  max robust bracket width = %.4e  (median %.4e)\n',max(dRob),median(dRob));
for j=1:3
  m=abs(robLo-levels(j)); fprintf('  q=%.3f min |robust-q| over states = %.4e ; states within bracket of q = %d\n', ...
    levels(j),min(m),nnz(m<=dRob));
end
save(fullfile(p.runs,'decide_96x12.mat'),'passLo','passHi','robLo','robHi','refLo','refHi','pLo','pHi','mbLo','mbHi','-v7.3');
function [pass,rob]=deal_(Q,qref,H0,levels)
ratio=Q./qref; rob=min(ratio,[],2); pass=false(size(Q,1),numel(levels));
for j=1:numel(levels), pass(:,j)=H0&rob>=levels(j); end
end
