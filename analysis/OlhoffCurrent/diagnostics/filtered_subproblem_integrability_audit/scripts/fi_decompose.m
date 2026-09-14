function out = fi_decompose()
%FI_DECOMPOSE  Split the measured antisymmetry into its two exact mechanisms.
%
%   From the verified decomposition
%       J_filt = A*D_{g/rho} + A*Hess - diag(g_filt/rho),
%   the diagonal term cancels identically in any antisymmetric combination, so
%       u'J_filt v - v'J_filt u  =  S1 + S2
%   with
%       S1 = u'(A D_{g/rho})v - v'(A D_{g/rho})u      EXACT, no finite differences
%       S2 = (A'u)'(Hess v) - (A'v)'(Hess u)          needs only Hess*d = J_phys*d
%   Both are computed here and compared with the directly measured asymmetry.

S = fi_setup(); D = fi_directions(S);
study = fileparts(fileparts(mfilename('fullpath')));
rho = S.rho386(:); NE = S.NE; dl = 1e-3;
H = S.flt.H; Hs = full(S.flt.Hs(:));
A  = spdiags(1./(Hs.*rho),0,NE,NE) * H * spdiags(rho,0,NE,NE);
E0 = fi_eval(S, rho, 'mode1'); g = E0.gPhys;
T1 = spdiags(1./(Hs.*rho),0,NE,NE) * H * spdiags(g,0,NE,NE);   % A*D_{g/rho}

names = unique([D.pairs(:,1); D.pairs(:,2)]);
JPd = containers.Map(); JFd = containers.Map();
for k = 1:numel(names)
    d = D.(names{k});
    Ep = fi_eval(S, rho + dl*d, 'mode1');
    Em = fi_eval(S, rho - dl*d, 'mode1');
    JPd(names{k}) = (Ep.gPhys - Em.gPhys)/(2*dl);   % = Hess*d
    JFd(names{k}) = (Ep.gFilt - Em.gFilt)/(2*dl);   % = J_filt*d
end

rows = struct('pair',{},'measured',{},'S1',{},'S2',{},'S1plusS2',{}, ...
              'closure_relerr',{},'S1_share',{},'S2_share',{},'rel_asym',{});
for i = 1:size(D.pairs,1)
    un = D.pairs{i,1}; vn = D.pairs{i,2};
    u = D.(un); v = D.(vn);
    meas = u.'*JFd(vn) - v.'*JFd(un);
    S1 = u.'*(T1*v) - v.'*(T1*u);
    S2 = (A.'*u).'*JPd(vn) - (A.'*v).'*JPd(un);
    tot = S1 + S2;
    q = numel(rows)+1;
    rows(q).pair = sprintf('%s,%s',un,vn);
    rows(q).measured = meas; rows(q).S1 = S1; rows(q).S2 = S2; rows(q).S1plusS2 = tot;
    rows(q).closure_relerr = abs(meas - tot)/max(abs(meas),realmin);
    rows(q).S1_share = abs(S1)/max(abs(S1)+abs(S2),realmin);
    rows(q).S2_share = abs(S2)/max(abs(S1)+abs(S2),realmin);
    rows(q).rel_asym = abs(meas)/max(abs(u.'*JFd(vn)), abs(v.'*JFd(un)));
end

out.delta = dl;
out.rows = rows;
out.max_closure_relerr = max([rows.closure_relerr]);
out.median_closure_relerr = median([rows.closure_relerr]);
out.median_S1_share = median([rows.S1_share]);
out.median_S2_share = median([rows.S2_share]);
out.closure_verified = out.max_closure_relerr < 1e-3;
out.identity = 'u''J_filt v - v''J_filt u = S1 + S2 exactly (the diagonal term cancels)';

f = fullfile(study,'evaluations','asymmetry_decomposition.json');
fid = fopen(f,'w'); fprintf(fid,'%s',jsonencode(out,'PrettyPrint',true)); fclose(fid);

fprintf('[fi_decompose] delta = %g\n', dl);
fprintf('  %-14s %-12s %-12s %-12s %-10s %-10s %-10s\n', ...
    'pair','measured','S1 (rho-wt)','S2 (A*Hess)','closure','S1 share','rel asym');
for q = 1:numel(rows)
    fprintf('  %-14s %-12.4e %-12.4e %-12.4e %-10.2e %-10.3f %-10.3e\n', ...
        rows(q).pair, rows(q).measured, rows(q).S1, rows(q).S2, ...
        rows(q).closure_relerr, rows(q).S1_share, rows(q).rel_asym);
end
fprintf('  closure verified = %d (max rel err %.3e); median S1 share %.3f, S2 share %.3f\n', ...
    out.closure_verified, out.max_closure_relerr, out.median_S1_share, out.median_S2_share);
end
