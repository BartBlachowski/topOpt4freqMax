function fi_export_fields()
%FI_EXPORT_FIELDS  Write the element fields the figures need.  Read-only.
S = fi_setup(); D = fi_directions(S);
study = fileparts(fileparts(mfilename('fullpath')));
rho = S.rho386(:);
E = fi_eval(S, rho, 'mode1');

% --- inner-KKT residual field at the production terminal iterate ---------
L = load(fullfile(study,'evaluations','inner_kkt_state.mat'));
ctx = L.ctx; r = L.recP(end); x = L.stP.xFinal;
NE = S.NE; nvar = NE+1; N = numel(ctx.lam); lamref = ctx.lam(1); Vtot = ctx.volfrac*NE;
drho = x(1:NE); bs = x(end);
[~, ddlam] = deltaLambda(ctx.F, drho, ctx.dOff);
dfdx = zeros(N+2,nvar);
for j = 1:N, dfdx(j,1:NE) = -ddlam(:,j).'/lamref; dfdx(j,nvar) = 1; end
dfdx(N+1,1:NE) = -ctx.fJJ.'/lamref; dfdx(N+1,nvar) = 1;
dfdx(N+2,1:NE) = 1/Vtot;
df0dx = zeros(nvar,1); df0dx(nvar) = -1;
gL = df0dx + dfdx.'*r.lam(:);
gLfull = gL - r.xsi(:) + r.eta(:);
sRow = sqrt(mean((ddlam(:,1)/lamref).^2));

F = struct();
F.nelx = S.nelx; F.nely = S.nely; F.NE = NE;
F.rho = rho;
F.gPhys = E.gPhys;
F.gFilt = E.gFilt;
F.gDiff = E.gFilt - E.gPhys;
F.kkt_resid = gLfull(1:NE);
F.kkt_resid_nobox = gL(1:NE);
F.sRow = sRow;
F.xsi = r.xsi(1:NE); F.eta = r.eta(1:NE);
F.drho = drho;
F.gray = double(D.gray_mask);
F.core = double(D.core_mask);
F.depth = D.depth;
F.lam = r.lam(:);
F.rminEl = S.rminEl;
save(fullfile(study,'evaluations','fields.mat'),'-struct','F','-v7.3');
fprintf('[fi_export_fields] wrote fields.mat  (sRow=%.6e)\n', sRow);

% class-resolved residual statistics (figure 11/12)
cls = struct('name',{'void','gray-shell','gray-core','solid'}, ...
             'mask',{rho<0.1, D.gray_mask & ~D.core_mask, D.core_mask, rho>0.9});
T = struct('class',{},'n',{},'frac',{},'kkt_norm_rms',{},'kkt_norm_p90',{}, ...
           'gPhys_rms',{},'gFilt_rms',{},'gFilt_over_gPhys',{});
for k = 1:numel(cls)
    m = cls(k).mask(:);
    q = numel(T)+1; T(q).class = cls(k).name; T(q).n = nnz(m); T(q).frac = mean(m);
    T(q).kkt_norm_rms = sqrt(mean((gLfull(find(m)).^2)))/sRow; %#ok<FNDSB>
    T(q).kkt_norm_p90 = quantile(abs(gLfull(find(m)))/sRow, 0.9); %#ok<FNDSB>
    T(q).gPhys_rms = sqrt(mean(E.gPhys(m).^2));
    T(q).gFilt_rms = sqrt(mean(E.gFilt(m).^2));
    T(q).gFilt_over_gPhys = T(q).gFilt_rms/T(q).gPhys_rms;
end
f = fullfile(study,'evaluations','residual_by_class.json');
fid = fopen(f,'w'); fprintf(fid,'%s',jsonencode(T,'PrettyPrint',true)); fclose(fid);
fprintf('  %-12s %-8s %-12s %-12s %-12s %-10s\n','class','n','kktRMS/s','gPhysRMS','gFiltRMS','ratio');
for q = 1:numel(T)
    fprintf('  %-12s %-8d %-12.4f %-12.3e %-12.3e %-10.3f\n', T(q).class, T(q).n, ...
        T(q).kkt_norm_rms, T(q).gPhys_rms, T(q).gFilt_rms, T(q).gFilt_over_gPhys);
end
end
