function out = fi_counterfactual_operators()
%FI_COUNTERFACTUAL_OPERATORS  ANALYSIS ONLY -- would a different filter operator
%   be conservative?  No production file is changed and no run uses these.
%
%   The decomposition showed the A*Hess term carries ~99% of the antisymmetry.
%   That term survives for ANY operator B that does not commute with the
%   Hessian, so this asks, purely diagnostically, what would happen under:
%       A     the production sensitivity filter  diag(1/(Hs.rho)) H diag(rho)
%       B     the plain weighted average          diag(1/Hs) H        (rho-free)
%       Bsym  its symmetrization                  (B+B')/2
%       W'    the DENSITY-filter chain rule       (diag(1/Hs) H)'
%   For each, the antisymmetry of  M*Hess  is evaluated on the same direction
%   pairs using the measured Hessian-vector products.

S = fi_setup(); D = fi_directions(S);
study = fileparts(fileparts(mfilename('fullpath')));
rho = S.rho386(:); NE = S.NE; dl = 1e-3;
H = S.flt.H; Hs = full(S.flt.Hs(:));

A    = spdiags(1./(Hs.*rho),0,NE,NE) * H * spdiags(rho,0,NE,NE);
B    = spdiags(1./Hs,0,NE,NE) * H;
Bsym = (B + B.')/2;
Wt   = B.';

names = unique([D.pairs(:,1); D.pairs(:,2)]);
Hd = containers.Map();
for k = 1:numel(names)
    d = D.(names{k});
    Ep = fi_eval(S, rho + dl*d, 'mode1');
    Em = fi_eval(S, rho - dl*d, 'mode1');
    Hd(names{k}) = (Ep.gPhys - Em.gPhys)/(2*dl);        % Hess*d
end

ops = struct('name',{'A_production','B_plain_average','Bsym_symmetrized','Wt_density_chain'}, ...
             'M',{A,B,Bsym,Wt});
rows = struct('pair',{},'op',{},'skew',{},'rel',{});
for i = 1:size(D.pairs,1)
    un = D.pairs{i,1}; vn = D.pairs{i,2};
    u = D.(un); v = D.(vn);
    for k = 1:numel(ops)
        M = ops(k).M;
        a = (M.'*u).'*Hd(vn);       % u'(M*Hess)v
        b = (M.'*v).'*Hd(un);       % v'(M*Hess)u
        q = numel(rows)+1;
        rows(q).pair = sprintf('%s,%s',un,vn); rows(q).op = ops(k).name;
        rows(q).skew = a-b;
        rows(q).rel  = abs(a-b)/max(max(abs(a),abs(b)),realmin);
    end
end

out.rows = rows;
out.summary = struct('op',{},'median_rel',{},'max_rel',{},'operator_symmetric',{});
for k = 1:numel(ops)
    sel = strcmp({rows.op}, ops(k).name);
    q = numel(out.summary)+1;
    out.summary(q).op = ops(k).name;
    out.summary(q).median_rel = median([rows(sel).rel]);
    out.summary(q).max_rel = max([rows(sel).rel]);
    out.summary(q).operator_symmetric = isequal(ops(k).M, ops(k).M.');
end
out.note = ['Diagnostic only.  M*Hess is symmetric iff M*Hess = Hess*M''. ' ...
            'Symmetrizing M does NOT make M*Hess symmetric unless M also ' ...
            'commutes with Hess.  The density-filter chain rule W'' is listed ' ...
            'because F(rho)=f(W*rho) has gradient W''*grad f EVALUATED AT W*rho; ' ...
            'the entry here uses grad f at rho, so it is NOT that consistent ' ...
            'formulation and is shown only to isolate the operator effect.'];

f = fullfile(study,'evaluations','counterfactual_operators.json');
fid = fopen(f,'w'); fprintf(fid,'%s',jsonencode(out,'PrettyPrint',true)); fclose(fid);

fprintf('[fi_counterfactual_operators] antisymmetry of M*Hess on the 10 pairs\n');
fprintf('  %-22s %-10s %-14s %-12s\n','operator','symmetric','median rel','max rel');
for k = 1:numel(out.summary)
    fprintf('  %-22s %-10d %-14.4e %-12.4e\n', out.summary(k).op, ...
        out.summary(k).operator_symmetric, out.summary(k).median_rel, out.summary(k).max_rel);
end
end
