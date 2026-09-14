function out = fi_symmetry()
%FI_SYMMETRY  Parts 5 and 10: Jacobian symmetry of g_filt, with the mandatory
%   g_phys positive control and multiplicity screening.
%
%   J is never assembled.  Jv is estimated by central differences about the
%   FROZEN rho386.  No density is updated: every perturbed rho is a temporary
%   evaluation point and is discarded.

S = fi_setup();
study = fileparts(fileparts(mfilename('fullpath')));
D = fi_directions(S);
rho = S.rho386(:);

deltas = [1e-3 3e-4 1e-4 3e-5];          % preregistered
dirNames = unique([D.pairs(:,1); D.pairs(:,2)]);

% ---- base evaluation ----------------------------------------------------
E0 = fi_eval(S, rho, 'mode1');
base = struct('gap12',E0.gap12,'omega',E0.omega(:).');

% ---- J*d for every unique direction and delta, by central differences ---
JP = containers.Map(); JF = containers.Map();
screen = struct('n',0,'excluded',0,'min_gap12',inf,'order_changes',0);
nev = 0; t0 = tic;
for k = 1:numel(dirNames)
    d = D.(dirNames{k});
    for t = 1:numel(deltas)
        dl = deltas(t);
        Ep = fi_eval(S, rho + dl*d, 'mode1');
        Em = fi_eval(S, rho - dl*d, 'mode1');
        nev = nev + 2;
        for E = [Ep Em]
            screen.n = screen.n + 1;
            screen.min_gap12 = min(screen.min_gap12, E.gap12);
            if E.gap12 < 0.05, screen.excluded = screen.excluded + 1; end
            if any(diff(E.omega) < 0), screen.order_changes = screen.order_changes + 1; end
        end
        key = sprintf('%s|%g', dirNames{k}, dl);
        JP(key) = (Ep.gPhys - Em.gPhys)/(2*dl);
        JF(key) = (Ep.gFilt - Em.gFilt)/(2*dl);
    end
end
fprintf('[fi_symmetry] %d evaluations in %.1f s\n', nev, toc(t0));

% ---- the symmetry statistic --------------------------------------------
np = size(D.pairs,1);
res = struct('pair',{},'delta',{},'uJv_phys',{},'vJu_phys',{},'r_phys',{}, ...
             'uJv_filt',{},'vJu_filt',{},'r_filt',{},'a_filt',{},'a_phys',{});
for i = 1:np
    un = D.pairs{i,1}; vn = D.pairs{i,2};
    u = D.(un); v = D.(vn);
    for t = 1:numel(deltas)
        dl = deltas(t);
        ku = sprintf('%s|%g', un, dl); kv = sprintf('%s|%g', vn, dl);
        for fld = {'phys','filt'}
            if strcmp(fld{1},'phys'), Jv = JP(kv); Ju = JP(ku);
            else,                     Jv = JF(kv); Ju = JF(ku); end
            a = u.'*Jv; b = v.'*Ju;
            r = abs(a-b)/max(max(abs(a),abs(b)), realmin);
            if strcmp(fld{1},'phys'), ap=a; bp=b; rp=r; else, af=a; bf=b; rf=r; end
        end
        q = numel(res)+1;
        res(q).pair = sprintf('%s,%s',un,vn);  res(q).delta = dl;
        res(q).uJv_phys = ap; res(q).vJu_phys = bp; res(q).r_phys = rp;
        res(q).uJv_filt = af; res(q).vJu_filt = bf; res(q).r_filt = rf;
        res(q).a_phys = abs(ap-bp); res(q).a_filt = abs(af-bf);
    end
end

out = struct();
out.base = base;
out.screen = screen;
out.deltas = deltas;
out.n_evaluations = nev;
out.results = res;
out.direction_info = D.info;
out.core_fraction = D.core_fraction;
out.core_maxdepth = D.core_maxdepth;

% ---- medians per delta --------------------------------------------------
med = zeros(numel(deltas),4);
for t = 1:numel(deltas)
    sel = [res.delta] == deltas(t);
    med(t,:) = [deltas(t), median([res(sel).r_filt]), median([res(sel).r_phys]), sum(sel)];
end
out.median_cols = {'delta','median_r_filt','median_r_phys','nPairs'};
out.median_by_delta = med;

f = fullfile(study,'evaluations','jacobian_symmetry.json');
fid = fopen(f,'w'); fprintf(fid,'%s',jsonencode(out,'PrettyPrint',true)); fclose(fid);

fprintf('\n  multiplicity screen: %d samples, %d excluded (gap12<0.05), min gap12=%.5f, %d order changes\n', ...
    screen.n, screen.excluded, screen.min_gap12, screen.order_changes);
fprintf('\n  %-10s %-16s %-16s\n','delta','median r_filt','median r_phys');
for t = 1:numel(deltas)
    fprintf('  %-10.3g %-16.4e %-16.4e\n', med(t,1), med(t,2), med(t,3));
end
fprintf('\n  per-pair relative asymmetry r (filtered | physical):\n');
for i = 1:np
    un = D.pairs{i,1}; vn = D.pairs{i,2};
    sel = find(strcmp({res.pair}, sprintf('%s,%s',un,vn)));
    fprintf('  %-14s', sprintf('%s,%s',un,vn));
    for q = sel, fprintf(' %8.2e', res(q).r_filt); end
    fprintf('   |');
    for q = sel, fprintf(' %8.2e', res(q).r_phys); end
    fprintf('\n');
end
end
