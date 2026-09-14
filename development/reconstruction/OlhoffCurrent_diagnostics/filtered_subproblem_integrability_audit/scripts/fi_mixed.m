function out = fi_mixed()
%FI_MIXED  Part 7: element-pair mixed partials for g_phys and g_filt, and the
%   test of the analytic skew decomposition of Part 8.
%
%   Columns J(:,j) are taken by central differences on a SINGLE element j, so
%   one pair of evaluations yields d g_i / d rho_j for every i at once.  No
%   density is updated.

S = fi_setup(); D = fi_directions(S);
study = fileparts(fileparts(mfilename('fullpath')));
rho = S.rho386(:); NE = S.NE;
nelx = S.nelx; nely = S.nely;
de = 1e-5;                                   % preregistered single-element step

depth = D.depth; core = D.core_mask; gray = D.gray_mask;
% --- preregistered deterministic selection ------------------------------
[~, oc] = sort(depth .* core, 'descend');          sCore  = oc(1:8);
gd = depth; gd(~gray) = inf;  [~, og] = sort(gd, 'ascend');  sShell = og(1:8);
rs = rho; rs(rho <= 0.9) = -inf; [~, os] = sort(rs,'descend'); sSolid = os(1:6);
rv = rho; rv(rho >= 0.1) =  inf; [~, ov] = sort(rv,'ascend');  sVoid  = ov(1:6);
c1 = sCore(1); [iy,ix] = ind2sub([nely nelx], c1);
nb = [];
for d = [-1 1]
    if iy+d>=1 && iy+d<=nely, nb(end+1) = sub2ind([nely nelx], iy+d, ix); end %#ok<AGROW>
    if ix+d>=1 && ix+d<=nelx, nb(end+1) = sub2ind([nely nelx], iy, ix+d); end %#ok<AGROW>
end
sel = unique([sCore(:); sShell(:); sSolid(:); sVoid(:); nb(:)], 'stable');
ns = numel(sel);
cls = strings(ns,1);
for k = 1:ns
    e = sel(k);
    if     any(e==sCore),  cls(k) = "core";
    elseif any(e==sShell), cls(k) = "shell";
    elseif any(e==sSolid), cls(k) = "solid";
    elseif any(e==sVoid),  cls(k) = "void";
    else,                  cls(k) = "neighbour"; end
end

% --- columns by central FD ----------------------------------------------
JP = zeros(ns,ns); JF = zeros(ns,ns);
excursion = 0; t0 = tic;
for k = 1:ns
    j = sel(k);
    rp = rho; rp(j) = rp(j) + de;
    rm = rho; rm(j) = rm(j) - de;
    excursion = max(excursion, max(S.rhomin - min(rm), 0));
    Ep = fi_eval(S, rp, 'mode1');
    Em = fi_eval(S, rm, 'mode1');
    JP(:,k) = (Ep.gPhys(sel) - Em.gPhys(sel))/(2*de);
    JF(:,k) = (Ep.gFilt(sel) - Em.gFilt(sel))/(2*de);
end
wall = toc(t0);

% --- analytic term-1 skew for the same pairs ----------------------------
H = S.flt.H; Hs = full(S.flt.Hs(:));
E0 = fi_eval(S, rho, 'mode1'); g = E0.gPhys;
Hsub = full(H(sel,sel));
T1 = zeros(ns);
for a = 1:ns
    for b = 1:ns
        T1(a,b) = Hsub(a,b)*g(sel(b))/(Hs(sel(a))*rho(sel(a)));
    end
end
T1skew = T1 - T1.';

skewP = JP - JP.';
skewF = JF - JF.';
den = max(abs(JF), abs(JF.'));
relF = abs(skewF)./max(den, realmin);
denP = max(abs(JP), abs(JP.'));
relP = abs(skewP)./max(denP, realmin);
offd = ~eye(ns);
inStencil = Hsub ~= 0 & offd;

out = struct();
out.n_elements = ns;
out.classes = cellstr(cls).';
out.elements = sel(:).';
out.rho_of_elements = rho(sel).';
out.delta = de;
out.max_box_excursion = excursion;
out.wall_s = wall;
out.n_evaluations = 2*ns + 1;
out.physical = struct('median_rel_asym', median(relP(offd)), ...
    'p90_rel_asym', quantile(relP(offd),0.9), 'max_rel_asym', max(relP(offd)), ...
    'max_abs_skew', max(abs(skewP(offd))), 'fro_skew_ratio', norm(skewP,'fro')/norm(JP,'fro'));
out.filtered = struct('median_rel_asym', median(relF(offd)), ...
    'p90_rel_asym', quantile(relF(offd),0.9), 'max_rel_asym', max(relF(offd)), ...
    'max_abs_skew', max(abs(skewF(offd))), 'fro_skew_ratio', norm(skewF,'fro')/norm(JF,'fro'));
out.stencil = struct('n_in_stencil', nnz(inStencil), 'n_offdiag', nnz(offd), ...
    'median_rel_asym_in_stencil', median(relF(inStencil)), ...
    'median_rel_asym_out_stencil', median(relF(offd & ~inStencil)));

% ---- does the analytic term-1 skew explain the measured skew? ----------
res = skewF - T1skew;
out.analytic_check = struct( ...
    'statement','measured skew(J_filt) = T1skew (closed form) + skew(A*Hess)', ...
    'T1skew_fro_in_stencil', norm(T1skew(inStencil)), ...
    'measured_skew_fro_in_stencil', norm(skewF(inStencil)), ...
    'residual_fro_in_stencil', norm(res(inStencil)), ...
    'residual_over_measured_in_stencil', norm(res(inStencil))/max(norm(skewF(inStencil)),realmin), ...
    'T1_explains_fraction_in_stencil', 1 - norm(res(inStencil))/max(norm(skewF(inStencil)),realmin), ...
    'measured_skew_fro_out_stencil', norm(skewF(offd & ~inStencil)), ...
    'T1skew_fro_out_stencil', norm(T1skew(offd & ~inStencil)));

% class-resolved summary
uc = unique(cls);
cr = struct('class',{},'median_rel_filt',{},'median_rel_phys',{},'n',{});
for k = 1:numel(uc)
    m = (cls == uc(k));
    M = false(ns); M(m,:) = true; M = M & offd;
    q = numel(cr)+1; cr(q).class = char(uc(k));
    cr(q).median_rel_filt = median(relF(M));
    cr(q).median_rel_phys = median(relP(M));
    cr(q).n = nnz(m);
end
out.by_class = cr;

save(fullfile(study,'evaluations','mixed_partials.mat'), ...
     'JP','JF','T1','T1skew','skewP','skewF','relP','relF','sel','cls','Hsub','-v7.3');
f = fullfile(study,'evaluations','mixed_partials.json');
fid = fopen(f,'w'); fprintf(fid,'%s',jsonencode(out,'PrettyPrint',true)); fclose(fid);

fprintf('[fi_mixed] %d elements, %d evaluations, %.1f s, max box excursion %.2e\n', ...
    ns, out.n_evaluations, wall, excursion);
fprintf('  PHYSICAL  median rel asym = %.3e  p90 = %.3e  max = %.3e  fro ratio = %.3e\n', ...
    out.physical.median_rel_asym, out.physical.p90_rel_asym, out.physical.max_rel_asym, out.physical.fro_skew_ratio);
fprintf('  FILTERED  median rel asym = %.3e  p90 = %.3e  max = %.3e  fro ratio = %.3e\n', ...
    out.filtered.median_rel_asym, out.filtered.p90_rel_asym, out.filtered.max_rel_asym, out.filtered.fro_skew_ratio);
fprintf('  in-stencil pairs %d/%d: median rel asym %.3e ; out-of-stencil %.3e\n', ...
    out.stencil.n_in_stencil, out.stencil.n_offdiag, ...
    out.stencil.median_rel_asym_in_stencil, out.stencil.median_rel_asym_out_stencil);
fprintf('  ANALYTIC: T1 explains %.4f of the in-stencil skew (residual/measured = %.4f)\n', ...
    out.analytic_check.T1_explains_fraction_in_stencil, out.analytic_check.residual_over_measured_in_stencil);
fprintf('  out-of-stencil measured skew %.3e vs T1 %.3e (T1 must be exactly 0 there)\n', ...
    out.analytic_check.measured_skew_fro_out_stencil, out.analytic_check.T1skew_fro_out_stencil);
for q = 1:numel(cr)
    fprintf('  class %-10s n=%2d  median rel asym: filtered %.3e  physical %.3e\n', ...
        cr(q).class, cr(q).n, cr(q).median_rel_filt, cr(q).median_rel_phys);
end
end
