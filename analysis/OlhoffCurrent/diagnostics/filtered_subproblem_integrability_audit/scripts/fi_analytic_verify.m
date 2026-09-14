function out = fi_analytic_verify()
%FI_ANALYTIC_VERIFY  Exact test of the Part-8 decomposition on FULL columns.
%
%   Claim:  J_filt = A*D_{g/rho} + A*Hess - diag(g_filt/rho),  A = D_{1/(Hs.rho)} H D_rho
%   Column form, for a single element j:
%       J_filt(:,j) = A*J_phys(:,j) + A(:,j)*g_j/rho_j - e_j*g_filt_j/rho_j
%   J_phys(:,j) = Hess(:,j) is measured by central FD; A, g, g_filt are exact.
%   A match to FD accuracy verifies the derivation with no fitted quantity.

S = fi_setup(); D = fi_directions(S);
study = fileparts(fileparts(mfilename('fullpath')));
rho = S.rho386(:); NE = S.NE; de = 1e-5;
H = S.flt.H; Hs = full(S.flt.Hs(:));
A = spdiags(1./(Hs.*rho),0,NE,NE) * H * spdiags(rho,0,NE,NE);

E0 = fi_eval(S, rho, 'mode1');
g = E0.gPhys; gf = E0.gFilt;

depth = D.depth; core = D.core_mask; gray = D.gray_mask;
[~, oc] = sort(depth.*core,'descend');             c = oc(1:2);
gd = depth; gd(~gray) = inf; [~, og] = sort(gd,'ascend');  sh = og(1:2);
rs = rho; rs(rho<=0.9) = -inf; [~, os] = sort(rs,'descend'); so = os(1);
rv = rho; rv(rho>=0.1) =  inf; [~, ov] = sort(rv,'ascend');  vo = ov(1);
sel = [c(:); sh(:); so; vo];
lbl = {'core1','core2','shell1','shell2','solid','void'};

rows = struct('element',{},'class',{},'rho',{},'relerr',{},'relerr_stencil',{}, ...
              'normJfilt',{},'normPred',{},'normResid',{}, ...
              'term_AHess',{},'term_AD',{},'term_diag',{});
for k = 1:numel(sel)
    j = sel(k);
    rp = rho; rp(j) = rp(j)+de;
    rm = rho; rm(j) = rm(j)-de;
    Ep = fi_eval(S, rp, 'mode1'); Em = fi_eval(S, rm, 'mode1');
    JPcol = (Ep.gPhys - Em.gPhys)/(2*de);        % = Hess(:,j)
    JFcol = (Ep.gFilt - Em.gFilt)/(2*de);        % measured

    t_AHess = A*JPcol;
    t_AD    = full(A(:,j))*(g(j)/rho(j));
    t_diag  = zeros(NE,1); t_diag(j) = gf(j)/rho(j);
    pred    = t_AHess + t_AD - t_diag;
    resid   = JFcol - pred;

    stencil = find(H(:,j) ~= 0);
    q = numel(rows)+1;
    rows(q).element = j; rows(q).class = lbl{k}; rows(q).rho = rho(j);
    rows(q).normJfilt = norm(JFcol); rows(q).normPred = norm(pred);
    rows(q).normResid = norm(resid);
    rows(q).relerr = norm(resid)/max(norm(JFcol),realmin);
    rows(q).relerr_stencil = norm(resid(stencil))/max(norm(JFcol(stencil)),realmin);
    rows(q).term_AHess = norm(t_AHess);
    rows(q).term_AD = norm(t_AD);
    rows(q).term_diag = norm(t_diag);
end

out.rows = rows;
out.delta = de;
out.max_relerr = max([rows.relerr]);
out.median_relerr = median([rows.relerr]);
out.verified = out.max_relerr < 1e-3;
out.claim = ['J_filt(:,j) = A*J_phys(:,j) + A(:,j)*g_j/rho_j - e_j*g_filt_j/rho_j, ' ...
             'with A = diag(1/(Hs.*rho))*H*diag(rho)'];

f = fullfile(study,'evaluations','analytic_verification.json');
fid = fopen(f,'w'); fprintf(fid,'%s',jsonencode(out,'PrettyPrint',true)); fclose(fid);

fprintf('[fi_analytic_verify] column-wise test of the analytic decomposition\n');
fprintf('  %-8s %-10s %-12s %-12s %-12s %-12s %-12s\n', ...
    'class','rho','relerr','||A*Hess||','||A*D_g/r||','||diag||','||J_filt||');
for q = 1:numel(rows)
    fprintf('  %-8s %-10.4g %-12.3e %-12.3e %-12.3e %-12.3e %-12.3e\n', ...
        rows(q).class, rows(q).rho, rows(q).relerr, rows(q).term_AHess, ...
        rows(q).term_AD, rows(q).term_diag, rows(q).normJfilt);
end
fprintf('  max relative error = %.3e -> derivation VERIFIED = %d\n', out.max_relerr, out.verified);
end
